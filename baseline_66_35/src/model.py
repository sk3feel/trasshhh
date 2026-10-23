"""Unchanged Hugging Face BART architecture and local-only model loading."""

import json
from pathlib import Path

import torch
from transformers import BartConfig, BartForConditionalGeneration


def configure_runtime():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def create_model(config):
    """Random initialization, used only when explicitly starting a new training run."""
    configuration = BartConfig.from_dict(config)
    configuration._attn_implementation = "sdpa"
    return BartForConditionalGeneration(configuration)


def load_model(bundle, device="cuda"):
    """Read our own exported weights; never download pretrained weights."""
    model = BartForConditionalGeneration.from_pretrained(
        Path(bundle) / "model",
        local_files_only=True,
        attn_implementation="sdpa",
    )
    return model.to(device).eval()


def load_resume(path, device="cpu"):
    """The caller must supply a trusted local training checkpoint."""
    return torch.load(path, map_location=device, weights_only=False)


def verify_model_state(model, checkpoint):
    actual = model.state_dict()
    expected = checkpoint["model"]
    if actual.keys() != expected.keys():
        raise ValueError("Model state keys differ from the frozen baseline")
    for name, tensor in actual.items():
        if not torch.equal(tensor.cpu(), expected[name].cpu()):
            raise ValueError(f"Model tensor differs: {name}")
    return {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "tensor_entries_identical": len(actual),
        "embeddings_tied": model.model.shared.weight.data_ptr()
        == model.lm_head.weight.data_ptr(),
    }
