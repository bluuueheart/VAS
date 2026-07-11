"""Device detection and configuration for GPU/NPU support.

Supports:
- NVIDIA CUDA GPUs
- Huawei Ascend NPU (torch_npu)
- CPU fallback
"""
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)
_DETECTED_DEVICE_CACHE: Optional[Tuple[str, Optional[int]]] = None
_DEVICE_LOGGED = False


def detect_device() -> Tuple[str, Optional[int]]:
    """
    Detect available accelerator device.
    
    Returns:
        Tuple of (device_type, device_count)
        - device_type: 'cuda', 'npu', or 'cpu'
        - device_count: Number of available devices (None for CPU)
    """
    global _DETECTED_DEVICE_CACHE, _DEVICE_LOGGED
    if _DETECTED_DEVICE_CACHE is not None:
        return _DETECTED_DEVICE_CACHE

    # Try CUDA first (NVIDIA GPUs)
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if not _DEVICE_LOGGED:
                logger.info(f"CUDA detected: {device_count} GPU(s) available")
                _DEVICE_LOGGED = True
            _DETECTED_DEVICE_CACHE = ("cuda", device_count)
            return _DETECTED_DEVICE_CACHE
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"CUDA check failed: {e}")
    
    # Try NPU (Huawei Ascend)
    try:
        import torch_npu  # noqa: F401
        import torch
        if hasattr(torch, 'npu') and torch.npu.is_available():
            device_count = torch.npu.device_count()
            if not _DEVICE_LOGGED:
                logger.info(f"Huawei Ascend NPU detected: {device_count} NPU(s) available")
                _DEVICE_LOGGED = True
            _DETECTED_DEVICE_CACHE = ("npu", device_count)
            return _DETECTED_DEVICE_CACHE
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"NPU check failed: {e}")
    
    # Check environment variable for forced device selection
    force_device = os.environ.get("DEVICE_TYPE", "").lower()
    if force_device in ("cuda", "npu", "cpu"):
        if force_device != "cpu":
            logger.warning(f"Forced to use {force_device} via DEVICE_TYPE env var, but device not available")
        else:
            logger.info("CPU mode forced via DEVICE_TYPE env var")
    
    # Default to CPU
    if not _DEVICE_LOGGED:
        logger.warning("No accelerator (CUDA/NPU) detected, falling back to CPU. Performance will be reduced.")
        _DEVICE_LOGGED = True
    _DETECTED_DEVICE_CACHE = ("cpu", None)
    return _DETECTED_DEVICE_CACHE


def get_device_string(device_type: str = None, device_id: int = 0) -> str:
    """
    Get PyTorch device string.
    
    Args:
        device_type: Type of device ('cuda', 'npu', 'cpu'). Auto-detect if None.
        device_id: Device index (0-based)
        
    Returns:
        Device string compatible with torch (e.g., 'cuda:0', 'npu:0', 'cpu')
    """
    if device_type is None:
        device_type, _ = detect_device()
    
    if device_type == "cpu":
        return "cpu"
    
    return f"{device_type}:{device_id}"


def select_preferred_device_id(device_type: str = None, purpose: str = "aux") -> int:
    """
    Select the best accelerator index for auxiliary workloads.

    Priority:
    1. Explicit environment override
    2. CUDA GPU with the most free memory
    3. Device 0 as a conservative fallback
    """
    if device_type is None:
        device_type, _ = detect_device()

    if device_type == "cpu":
        return 0

    normalized_purpose = str(purpose or "aux").strip().upper()
    override_keys = [
        f"TOOLSTAR_{normalized_purpose}_DEVICE_ID",
        "TOOLSTAR_DEVICE_ID",
    ]
    for key in override_keys:
        raw_value = os.environ.get(key, "").strip()
        if raw_value.isdigit():
            return max(0, int(raw_value))

    if device_type == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                best_idx = 0
                best_free = -1
                best_allocated = float("inf")
                for idx in range(torch.cuda.device_count()):
                    free_bytes = -1
                    try:
                        with torch.cuda.device(idx):
                            free_bytes, _ = torch.cuda.mem_get_info()
                    except Exception:
                        free_bytes = -1
                    try:
                        allocated = float(torch.cuda.memory_allocated(idx))
                    except Exception:
                        allocated = float("inf")
                    if free_bytes > best_free or (free_bytes == best_free and allocated < best_allocated):
                        best_idx = idx
                        best_free = free_bytes
                        best_allocated = allocated
                return best_idx
        except Exception as exc:
            logger.debug("Failed to auto-select CUDA device for %s: %s", purpose, exc)

    return 0


def get_preferred_device_string(device_type: str = None, purpose: str = "aux") -> str:
    """Return the preferred device string for auxiliary workloads."""
    if device_type is None:
        device_type, _ = detect_device()
    if device_type == "cpu":
        return "cpu"
    return get_device_string(device_type, select_preferred_device_id(device_type, purpose=purpose))


def setup_npu_env():
    """Setup environment variables for Huawei Ascend NPU."""
    # Recommended NPU environment variables
    # These help torch_npu and ascend toolkit work optimally
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")
    npu_env_vars = {
        "ASCEND_RT_VISIBLE_DEVICES": visible_devices,
        # Disable profiling and debug output for production
        "ASCEND_LAUNCH_BLOCKING": os.environ.get("ASCEND_LAUNCH_BLOCKING", "0"),
    }

    if "ASCEND_DEVICE_ID" in os.environ:
        npu_env_vars["ASCEND_DEVICE_ID"] = os.environ["ASCEND_DEVICE_ID"]
    
    for key, value in npu_env_vars.items():
        os.environ[key] = value
        logger.debug(f"Set {key}={value}")


def setup_cuda_env():
    """Setup environment variables for NVIDIA CUDA."""
    # These help CUDA memory management
    cuda_env_vars = {
        "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
        # Set via CUDA_VISIBLE_DEVICES env var before Python starts
    }
    
    for key, value in cuda_env_vars.items():
        os.environ[key] = value
        logger.debug(f"Set {key}={value}")


def initialize_device(device_type: str = None):
    """
    Initialize device and setup environment.
    
    Args:
        device_type: Force specific device type ('cuda', 'npu', 'cpu'). Auto-detect if None.
    """
    if device_type is None:
        device_type, _ = detect_device()
    
    logger.info(f"Initializing device: {device_type}")
    
    if device_type == "cuda":
        setup_cuda_env()
    elif device_type == "npu":
        setup_npu_env()
    else:
        logger.info("Using CPU device")


def get_accelerator_snapshot(device_type: str = None) -> str:
    """Return a compact runtime snapshot for the active accelerator backend."""
    if device_type is None:
        device_type, _ = detect_device()

    if device_type == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                return "cuda unavailable"

            parts = []
            for idx in range(torch.cuda.device_count()):
                try:
                    name = torch.cuda.get_device_name(idx)
                except Exception:
                    name = "unknown"
                try:
                    allocated_gb = torch.cuda.memory_allocated(idx) / (1024 ** 3)
                    reserved_gb = torch.cuda.memory_reserved(idx) / (1024 ** 3)
                    parts.append(
                        f"cuda:{idx} name={name} allocated={allocated_gb:.2f}GB reserved={reserved_gb:.2f}GB"
                    )
                except Exception:
                    parts.append(f"cuda:{idx} name={name} memory=unavailable")
            return " | ".join(parts) if parts else "cuda detected but no visible devices"
        except Exception as exc:
            return f"cuda snapshot unavailable: {exc}"

    if device_type == "npu":
        try:
            import torch

            if not hasattr(torch, "npu") or not torch.npu.is_available():
                return "npu unavailable"

            parts = []
            device_count = torch.npu.device_count()
            for idx in range(device_count):
                parts.append(f"npu:{idx}")
            return " | ".join(parts) if parts else "npu detected but no visible devices"
        except Exception as exc:
            return f"npu snapshot unavailable: {exc}"

    return "cpu"


def is_npu_runtime_error(error: Exception | str) -> bool:
    """Detect common Huawei Ascend NPU runtime errors by message pattern."""
    message = str(error).lower()
    patterns = [
        "507001",
        "acl stream synchronize failed",
        "acl api failed",
        "sdma error",
        "rtdevicesynchronize",
        "rtstreamsynchronize",
        "c10_npu",
        "ascend",
    ]
    return any(p in message for p in patterns)


__all__ = [
    "detect_device",
    "get_device_string",
    "select_preferred_device_id",
    "get_preferred_device_string",
    "setup_npu_env",
    "setup_cuda_env",
    "initialize_device",
    "get_accelerator_snapshot",
    "is_npu_runtime_error",
]
