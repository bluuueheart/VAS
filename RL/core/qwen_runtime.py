"""Runtime switches for Qwen3.5 optional fused kernels.

The local Orion CUDA runtime can abort inside LLVM while compiling one of the
optional Qwen3.5 extensions. Keep the switch in a small dependency-free
module so every real Transformers entry point (text and video) applies the
same safety policy before loading or executing the model.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


def disable_qwen35_fused_kernels_if_requested(
    transformers_module: Optional[Any] = None,
) -> Dict[str, Any]:
    """Disable selected Qwen3.5 fast-path symbols when explicitly requested."""
    raw_flag = os.environ.get("TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS", "").strip().lower()
    if raw_flag in {"", "0", "false", "no", "off"}:
        return {"requested": False, "mode": "off", "disabled_symbols": []}

    if raw_flag in {"1", "true", "yes", "on", "full", "all"}:
        mode = "full"
    elif raw_flag in {"gdn", "linear", "partial"}:
        mode = "gdn"
    elif raw_flag in {"conv", "causal_conv"}:
        mode = "conv"
    elif raw_flag in {"norm", "rmsnorm", "rms_norm"}:
        mode = "norm"
    elif raw_flag in {"convnorm", "conv_norm", "conv+norm", "causal_conv_norm"}:
        mode = "convnorm"
    else:
        logger.warning(
            "Ignoring unknown TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS=%r; "
            "expected full, gdn, conv, norm, or convnorm.",
            raw_flag,
        )
        return {"requested": True, "mode": "unknown", "disabled_symbols": []}

    module = transformers_module
    if module is None:
        try:
            import transformers as module  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Could not import Transformers for Qwen3.5 safety switch: %s", exc)
            return {"requested": True, "mode": mode, "disabled_symbols": [], "error": str(exc)}

    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5  # type: ignore
    except Exception as exc:  # pragma: no cover - optional model family
        logger.warning("Could not inspect Qwen3.5 fused-kernel globals: %s", exc)
        return {"requested": True, "mode": mode, "disabled_symbols": [], "error": str(exc)}

    disabled_by_mode = {
        "full": (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
            "FusedRMSNormGated",
        ),
        "gdn": (
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
        ),
        "conv": (
            "causal_conv1d_fn",
            "causal_conv1d_update",
        ),
        "norm": ("FusedRMSNormGated",),
        "convnorm": (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "FusedRMSNormGated",
        ),
    }
    disabled_symbols = []
    for symbol in disabled_by_mode[mode]:
        if getattr(modeling_qwen3_5, symbol, None) is not None:
            setattr(modeling_qwen3_5, symbol, None)
            disabled_symbols.append(symbol)
    if hasattr(modeling_qwen3_5, "is_fast_path_available"):
        modeling_qwen3_5.is_fast_path_available = False
    logger.warning(
        "Disabled Qwen3.5 fused kernels via TOOLSTAR_DISABLE_QWEN_FUSED_KERNELS=%s; "
        "using Transformers fallback for selected symbols (symbols=%s).",
        mode,
        ",".join(disabled_symbols) or "already-disabled",
    )
    return {"requested": True, "mode": mode, "disabled_symbols": disabled_symbols}
