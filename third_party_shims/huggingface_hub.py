"""Minimal local shim for environments without huggingface_hub."""


def hf_hub_download(*args, **kwargs):
    raise RuntimeError("huggingface_hub is not installed; pass a local checkpoint path instead")
