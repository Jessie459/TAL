import importlib
from typing import Any

import torch


def is_torch_npu_available():
    return importlib.util.find_spec("torch_npu") is not None


IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_NPU_AVAILABLE = is_torch_npu_available() and torch.npu.is_available()

if IS_NPU_AVAILABLE:
    import torch_npu

    torch.npu.config.allow_internal_format = False


def get_device_type() -> str:
    """Get device type based on current machine, currently only support CPU, CUDA, NPU."""
    if IS_CUDA_AVAILABLE:
        device = "cuda"
    elif IS_NPU_AVAILABLE:
        device = "npu"
    else:
        device = "cpu"

    return device


def get_torch_device() -> Any:
    """Get torch attribute based on device type, e.g. torch.cuda or torch.npu"""
    device_name = get_device_type()

    try:
        return getattr(torch, device_name)
    except AttributeError:
        print(f"Device namespace '{device_name}' not found in torch, try to load 'torch.cuda'.")
        return torch.cuda


def get_device_id() -> int:
    """Get current device id based on device type."""
    return get_torch_device().current_device()


def get_device_name() -> str:
    """Get current device name based on device type."""
    return f"{get_device_type()}:{get_device_id()}"


def parse_device_type(device):
    if isinstance(device, str):
        if device.startswith("cuda"):
            return "cuda"
        elif device.startswith("npu"):
            return "npu"
        else:
            return "cpu"
    elif isinstance(device, torch.device):
        return device.type
