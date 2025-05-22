import argparse
import json
import os
import random
import tempfile
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
)

_real_embedding = F.embedding


def _safe_embedding(weight, input, *args, **kwargs):
    if input.device != weight.device:
        input = input.to(weight.device)
    return _real_embedding(weight, input, *args, **kwargs)


F.embedding = _safe_embedding

_real_linear_forward = nn.Linear.forward


def _safe_linear_forward(self, input, *args, **kwargs):
    if input.device != self.weight.device:
        input = input.to(self.weight.device)
    return _real_linear_forward(self, input, *args, **kwargs)


nn.Linear.forward = _safe_linear_forward

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from inferencer import InterleaveInferencer


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _canonicalize_mem(mem: str | int | float) -> str:
    if isinstance(mem, str):
        return mem if mem.lower().endswith("gib") else f"{mem}GiB"
    return f"{mem}GiB"


def load_bagel(
        model_path: str,
        max_mem_per_gpu: str | int | float = "40GiB",
        dtype: torch.dtype = torch.bfloat16,
        offload_dir: Optional[str] = None,
):
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    bagel_cfg = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, bagel_cfg)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    max_mem_per_gpu = _canonicalize_mem(max_mem_per_gpu)
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    anchor_candidates = [k for k in device_map if "embed_tokens" in k]
    anchor = anchor_candidates[0] if anchor_candidates else next(iter(device_map))
    coupled = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    for k in coupled:
        if k in device_map:
            device_map[k] = device_map[anchor]

    needs_disk = any(v == "disk" for v in device_map.values())
    if needs_disk and offload_dir is None:
        offload_dir = tempfile.mkdtemp(prefix="bagel_offload_")
        print(f"[bagel_cli] Some layers set to 'disk'; using offload dir: {offload_dir}")
    if offload_dir is not None:
        os.makedirs(offload_dir, exist_ok=True)

    device_map = {k: ("cpu" if v == "disk" else v) for k, v in device_map.items()}

    load_kwargs = dict(
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=dtype,
    )
    if offload_dir is not None:
        load_kwargs["offload_folder"] = offload_dir

    model = load_checkpoint_and_dispatch(model, **load_kwargs).eval()

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    return InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )


_DEFAULT_INFER_HYPER = dict(
    cfg_text_scale=4.0,
    cfg_img_scale=1.0,
    cfg_interval=[0.4, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=1.0,
    cfg_renorm_type="global",
)


def build_infer_kwargs(args: argparse.Namespace, extra: Optional[Dict] = None) -> Dict:
    cfg = _DEFAULT_INFER_HYPER.copy()
    if extra:
        cfg.update(extra)
    if args.hparams_json:
        cfg.update(json.loads(args.hparams_json))
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BAGEL‑7B‑MoT CLI")
    p.add_argument("--model_path", required=True, help="Directory with BAGEL weights")
    p.add_argument("--mode", required=True, choices=["generate", "edit", "understand"], help="Task mode")
    p.add_argument("--prompt", required=True, help="Text prompt")
    p.add_argument("--image", help="Input image path for edit/understand modes")
    p.add_argument("--output", default="output.png", help="Path to save resulting image (if any)")
    p.add_argument("--think", action="store_true", help="Enable chain‑of‑thought reasoning")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument('--max_mem_per_gpu', type=str, default="40GiB")
    p.add_argument("--hparams_json", help="JSON string or file with inference hyper‑parameters")
    p.add_argument(
        "--offload_dir",
        help="Folder used by Accelerate to offload layers when VRAM is insufficient. If omitted, a temp dir is created.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Allow JSON file path for hyper‑params
    if args.hparams_json and os.path.isfile(args.hparams_json):
        with open(args.hparams_json) as f:
            args.hparams_json = f.read()

    inferencer = load_bagel(
        model_path=args.model_path,
        offload_dir=args.offload_dir,
        max_mem_per_gpu=args.max_mem_per_gpu,
    )

    kwargs = build_infer_kwargs(args)

    if args.mode == "generate":
        out = inferencer(text=args.prompt, think=args.think, **kwargs)
        out["image"].save(args.output)
        print(f"Generated image saved to {args.output}")

    elif args.mode == "edit":
        if not args.image:
            raise ValueError("--image required for edit mode")
        img = Image.open(args.image).convert("RGB")
        edit_cfg = dict(cfg_img_scale=2.0, cfg_interval=[0.0, 1.0])
        kwargs.update(edit_cfg)
        out = inferencer(image=img, text=args.prompt, think=args.think, **kwargs)
        out["image"].save(args.output)
        print(f"Edited image saved to {args.output}")

    elif args.mode == "understand":
        if not args.image:
            raise ValueError("--image required for understand mode")
        img = Image.open(args.image).convert("RGB")
        out = inferencer(
            image=img,
            text=args.prompt,
            understanding_output=True,
            think=args.think,
            **kwargs,
        )
        print("Model answer:\n" + out["text"])


if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    print(f"Total time: {t1 - t0:.2f} seconds")
