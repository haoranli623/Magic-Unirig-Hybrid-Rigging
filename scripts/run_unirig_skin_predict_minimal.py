#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from math import ceil
from pathlib import Path
from typing import Any

import lightning as L
import torch
import yaml
from box import Box


def _patch_typing_for_unirig() -> None:
    # UniRig-main has one annotation Dict[str, ...] that fails on Python 3.10.
    import typing

    original_type_check = typing._type_check

    def patched_type_check(arg: object, msg: str, *args: object, **kwargs: object) -> object:
        if arg is Ellipsis:
            return Any
        return original_type_check(arg, msg, *args, **kwargs)

    typing._type_check = patched_type_check  # type: ignore[attr-defined]


def _load_yaml(path: Path) -> Box:
    return Box(yaml.safe_load(path.read_text(encoding="utf-8")))


def _patch_torch_load_for_checkpoint_compat() -> None:
    """
    PyTorch 2.6 defaults torch.load(weights_only=True), while UniRig Lightning
    checkpoints require full-object unpickling metadata.
    """
    original_torch_load = torch.load

    def patched_torch_load(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load  # type: ignore[assignment]


def _install_flash_attn_fallback() -> None:
    """
    UniRig skin model imports `flash_attn.modules.mha.MHA`.
    If flash-attn is unavailable, install a minimal API-compatible fallback
    backed by torch.nn.MultiheadAttention so prediction can still run.
    """
    try:
        from flash_attn.modules.mha import MHA as _mha  # noqa: F401
        return
    except Exception:
        pass

    import types
    from torch import nn
    import torch.nn.functional as F

    class FallbackMHA(nn.Module):
        def __init__(self, embed_dim: int, num_heads: int, cross_attn: bool = False, **_: object):
            super().__init__()
            if embed_dim % num_heads != 0:
                raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.cross_attn = cross_attn
            if cross_attn:
                # Name these layers to match flash_attn checkpoint keys.
                self.Wq = nn.Linear(embed_dim, embed_dim, bias=True)
                self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
            else:
                self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
            b, n, c = x.shape
            return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
            b, h, n, d = x.shape
            return x.transpose(1, 2).contiguous().view(b, n, h * d)

        def forward(
            self,
            x: torch.Tensor,
            x_kv: torch.Tensor | None = None,
            **_: object,
        ) -> torch.Tensor:
            if self.cross_attn:
                kv_src = x_kv if x_kv is not None else x
                q = self.Wq(x)
                k, v = self.Wkv(kv_src).chunk(2, dim=-1)
            else:
                q, k, v = self.Wqkv(x).chunk(3, dim=-1)

            qh = self._reshape_heads(q)
            kh = self._reshape_heads(k)
            vh = self._reshape_heads(v)
            ah = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0, is_causal=False)
            out = self._merge_heads(ah)
            return self.out_proj(out)

    flash_attn_mod = types.ModuleType("flash_attn")
    modules_mod = types.ModuleType("flash_attn.modules")
    mha_mod = types.ModuleType("flash_attn.modules.mha")
    mha_mod.MHA = FallbackMHA
    modules_mod.mha = mha_mod
    flash_attn_mod.modules = modules_mod

    sys.modules["flash_attn"] = flash_attn_mod
    sys.modules["flash_attn.modules"] = modules_mod
    sys.modules["flash_attn.modules.mha"] = mha_mod


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal UniRig learned-skin predictor runner")
    parser.add_argument("--unirig-repo", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--data-name", type=str, default="predict_skeleton.npz")
    parser.add_argument("--cls", type=str, default="dt4d")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-fbx", type=Path, default=None)
    args = parser.parse_args()

    _patch_typing_for_unirig()
    _patch_torch_load_for_checkpoint_compat()
    _install_flash_attn_fallback()
    torch.set_float32_matmul_precision("high")
    L.seed_everything(args.seed, workers=True)

    unirig_repo = args.unirig_repo.resolve()
    if not unirig_repo.exists():
        raise FileNotFoundError(f"UniRig repo not found: {unirig_repo}")
    sys.path.insert(0, str(unirig_repo))

    from src.data.dataset import UniRigDatasetModule
    from src.data.datapath import Datapath
    from src.data.transform import TransformConfig
    from src.inference.download import download
    from src.model.parse import get_model
    from src.system.parse import get_system, get_writer
    from src.tokenizer.parse import get_tokenizer
    from src.tokenizer.spec import TokenizerConfig

    task = _load_yaml(args.task_config)
    if task.mode != "predict":
        raise ValueError(f"Expected predict mode in task config, got: {task.mode}")

    data_cfg = _load_yaml(unirig_repo / "configs" / "data" / f"{task.components.data}.yaml")
    transform_cfg = _load_yaml(unirig_repo / "configs" / "transform" / f"{task.components.transform}.yaml")
    predict_transform = TransformConfig.parse(config=transform_cfg.predict_transform_config)
    # Magic skeleton topology/order may not match UniRig's category skeleton templates.
    # Disable category-specific reordering and keep Magic's native joint order.
    predict_transform.order_config = None

    tokenizer_cfg = None
    tokenizer = None
    tokenizer_name = task.components.get("tokenizer", None)
    if tokenizer_name is not None:
        tokenizer_cfg = _load_yaml(unirig_repo / "configs" / "tokenizer" / f"{tokenizer_name}.yaml")
        tokenizer_cfg = TokenizerConfig.parse(config=tokenizer_cfg)
        tokenizer = get_tokenizer(config=tokenizer_cfg)

    model_cfg = _load_yaml(unirig_repo / "configs" / "model" / f"{task.components.model}.yaml")
    # Force non-flash code paths for environment portability.
    if "mesh_encoder" in model_cfg:
        model_cfg.mesh_encoder.enable_flash = False
    if "global_encoder" in model_cfg and "flash" in model_cfg.global_encoder:
        model_cfg.global_encoder.flash = False
    model = get_model(tokenizer=tokenizer, **model_cfg)

    datapath = Datapath(files=[str(args.sample_dir.resolve())], cls=args.cls)
    data_module = UniRigDatasetModule(
        process_fn=model._process_fn,
        train_dataset_config=None,
        predict_dataset_config=None,
        validate_dataset_config=None,
        train_transform_config=None,
        predict_transform_config=predict_transform,
        validate_transform_config=None,
        tokenizer_config=tokenizer_cfg,
        debug=False,
        data_name=args.data_name,
        datapath=datapath,
        cls=args.cls,
    )

    system_cfg = _load_yaml(unirig_repo / "configs" / "system" / f"{task.components.system}.yaml")
    system = get_system(
        **system_cfg,
        model=model,
        optimizer_config=task.get("optimizer", None),
        loss_config=task.get("loss", None),
        scheduler_config=task.get("scheduler", None),
        steps_per_epoch=ceil(1),
    )

    writer_cfg = dict(task.get("writer", {}))
    writer_cfg["npz_dir"] = str(args.npz_dir.resolve())
    writer_cfg["output_dir"] = str(args.output_dir.resolve())
    writer_cfg["output_name"] = None if args.output_fbx is None else str(args.output_fbx.resolve())
    writer_cfg["user_mode"] = True
    writer = get_writer(**writer_cfg, order_config=predict_transform.order_config)

    trainer_cfg = dict(task.get("trainer", {}))
    trainer = L.Trainer(
        callbacks=[writer],
        logger=None,
        **trainer_cfg,
    )

    ckpt_path = download(task.get("resume_from_checkpoint", None))
    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", ""))
    trainer.predict(system, datamodule=data_module, ckpt_path=ckpt_path, return_predictions=False)


if __name__ == "__main__":
    main()
