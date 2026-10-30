"""
Robust loading of the trained VideoMAE checkpoint.

Compatibility background
------------------------
The checkpoint in ``models/`` was exported with transformers 5.18, whose
VideoMAE attention stores fused query/value biases as ``q_bias``/``v_bias``
(T5-style; the key projection has no bias). The installed transformers 5.8.1
instead expects plain ``nn.Linear`` biases named ``query.bias``/``value.bias``.
A direct ``from_pretrained`` load on 5.x therefore reports ``q_bias``/``v_bias``
as UNEXPECTED and re-initializes the attention biases, which would silently
degrade inference.

This module maps the checkpoint keys onto the installed model's parameter
names (``q_bias -> query.bias``, ``v_bias -> value.bias``; key bias stays 0,
exactly matching the source architecture) before loading. Nothing is
downloaded and no weights are modified on disk.
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.pipeline import paths


def _checkpoint_needs_remap(state_dict_keys) -> bool:
    """True if the checkpoint uses fused q_bias/v_bias naming."""
    return any(k.endswith(".q_bias") or k.endswith(".v_bias") for k in state_dict_keys)


def load_videomae_model_and_processor(model_dir: Path | None = None):
    """Load trained VideoMAE model + image processor from the local directory.

    Never downloads: HF Hub access is disabled for the duration of the load
    and ``local_files_only=True`` is used.

    Returns:
        (model, processor) tuple. ``model`` is in eval mode on CPU.

    Raises:
        FileNotFoundError: If the model directory/weights are missing.
        RuntimeError: If trained weights could not be restored.
    """
    from transformers import (
        AutoImageProcessor,
        VideoMAEForVideoClassification,
        VideoMAEConfig,
    )

    model_dir = Path(model_dir) if model_dir else paths.resolve_videomae_dir()

    processor = AutoImageProcessor.from_pretrained(
        str(model_dir), local_files_only=True
    )

    config = VideoMAEConfig.from_pretrained(str(model_dir), local_files_only=True)

    # Inspect the checkpoint keys to decide whether remapping is required.
    weights_file = None
    for candidate in ("model.safetensors", "pytorch_model.bin"):
        p = model_dir / candidate
        if p.is_file():
            weights_file = p
            break
    if weights_file is None:
        raise FileNotFoundError(
            f"No VideoMAE weights found in: {model_dir} "
            f"(expected model.safetensors or pytorch_model.bin)"
        )

    if weights_file.suffix == ".safetensors":
        from safetensors.torch import load_file
        ckpt_state = load_file(str(weights_file))
    else:
        ckpt_state = torch.load(str(weights_file), map_location="cpu", weights_only=True)
        ckpt_state = ckpt_state.get("state_dict", ckpt_state) if isinstance(ckpt_state, dict) else ckpt_state

    remap_needed = _checkpoint_needs_remap(ckpt_state.keys())

    # Build the model architecture from the LOCAL config only, then load the
    # (possibly remapped) state dict strictly so nothing is silently skipped.
    model = VideoMAEForVideoClassification(config)
    model.load_state_dict(_remap_state_dict(ckpt_state) if remap_needed else ckpt_state,
                          strict=True)
    model.eval()
    return model, processor


def _remap_state_dict(state_dict: dict) -> dict:
    """Map transformers>=5.10 fused-bias names to plain Linear bias names.

    q_bias -> query.bias, v_bias -> value.bias. The key projection is
    bias-free in the source architecture, so key.bias is filled with zeros.
    """
    mapped = {}
    remapped_q = remapped_v = 0
    for key, tensor in state_dict.items():
        if key.endswith(".q_bias"):
            new_key = key[: -len("q_bias")] + "query.bias"
            mapped[new_key] = tensor
            remapped_q += 1
        elif key.endswith(".v_bias"):
            new_key = key[: -len("v_bias")] + "value.bias"
            mapped[new_key] = tensor
            remapped_v += 1
        else:
            mapped[key] = tensor

    if remapped_q == 0:
        raise RuntimeError("Remapping requested but no q_bias keys found")

    # Key projections: bias exists on the installed model but not in the
    # checkpoint (source used bias=False) -> zeros keep the math identical.
    for key, tensor in list(mapped.items()):
        if key.endswith("attention.query.bias"):
            key_bias_name = key.replace("query.bias", "key.bias")
            if key_bias_name not in mapped:
                mapped[key_bias_name] = torch.zeros_like(tensor)
    return mapped
