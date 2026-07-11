# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compatibility utilities for different versions of transformers library.
"""

import importlib.metadata
import inspect
from functools import wraps
from functools import lru_cache
from typing import Optional

from packaging import version

# Handle version compatibility for flash_attn_supports_top_left_mask
# This function was added in newer versions of transformers
try:
    from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
except ImportError:
    # For older versions of transformers that don't have this function
    # Default to False as a safe fallback for older versions
    def flash_attn_supports_top_left_mask():
        """Fallback implementation for older transformers versions.
        Returns False to disable features that require this function.
        """
        return False


@lru_cache
def is_transformers_version_in_range(min_version: Optional[str] = None, max_version: Optional[str] = None) -> bool:
    try:
        # Get the installed version of the transformers library
        transformers_version_str = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModuleNotFoundError("The `transformers` package is not installed.") from e

    transformers_version = version.parse(transformers_version_str)

    lower_bound_check = True
    if min_version is not None:
        lower_bound_check = version.parse(min_version) <= transformers_version

    upper_bound_check = True
    if max_version is not None:
        upper_bound_check = transformers_version <= version.parse(max_version)

    return lower_bound_check and upper_bound_check


@lru_cache
def get_auto_model_for_vision2seq():
    """Return the available VL auto model class across transformers versions."""

    try:
        # Prefer the newer class when available. In transformers 4.x this class has
        # a broader mapping than AutoModelForVision2Seq, and AutoModelForVision2Seq
        # is deprecated for removal in v5.
        from transformers import AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq

    return AutoModelForImageTextToText


def unpack_visual_output(visual_output):
    """Unpack the output from the visual encoder, handling both tuple and object return types.

    Newer versions of transformers return an object with `pooler_output` and `deepstack_features`
    attributes instead of a plain tuple.
    """
    if hasattr(visual_output, "pooler_output"):
        # For newer versions(>=5.0.0) of transformers, return the pooler_output and deepstack_features
        if hasattr(visual_output, "deepstack_features"):
            return visual_output.pooler_output, visual_output.deepstack_features
        else:
            return visual_output.pooler_output, None
    if isinstance(visual_output, tuple):
        return visual_output
    else:
        return visual_output, None


@lru_cache
def patch_accelerate_parameter_init_compat() -> bool:
    """Patch accelerate tensor assignment for torch versions without `_is_hf_initialized` support."""

    try:
        import torch
    except Exception:
        return False

    try:
        parameter_new_sig = inspect.signature(torch.nn.Parameter.__new__)
    except Exception:
        return False

    # Newer torch versions already support this kwargs path.
    if "_is_hf_initialized" in parameter_new_sig.parameters:
        return False

    patched_any = False

    original_parameter_new = torch.nn.Parameter.__new__
    if not getattr(original_parameter_new, "_verl_hf_param_init_compat_patch", False):

        @wraps(original_parameter_new)
        def _patched_parameter_new(cls, data=None, requires_grad=True, *args, **kwargs):
            # For mixed torch/transformers/accelerate versions, this internal flag can
            # be forwarded into Parameter.__new__, while older torch doesn't accept it.
            kwargs.pop("_is_hf_initialized", None)

            if "requires_grad" in kwargs:
                requires_grad = kwargs.pop("requires_grad")

            if args or kwargs:
                try:
                    return original_parameter_new(cls, data, *args, requires_grad=requires_grad, **kwargs)
                except TypeError:
                    # Keep backward compatibility with older torch signatures.
                    pass

            return original_parameter_new(cls, data, requires_grad)

        _patched_parameter_new._verl_hf_param_init_compat_patch = True
        torch.nn.Parameter.__new__ = staticmethod(_patched_parameter_new)
        patched_any = True

    try:
        from accelerate import big_modeling
    except Exception:
        return patched_any

    original_fn = getattr(big_modeling, "set_module_tensor_to_device", None)
    if original_fn is None:
        return patched_any
    if getattr(original_fn, "_verl_hf_param_init_compat_patch", False):
        return True

    @wraps(original_fn)
    def _patched_set_module_tensor_to_device(module, tensor_name, *args, **kwargs):
        target_module = module
        target_name = tensor_name
        if isinstance(tensor_name, str) and "." in tensor_name:
            parts = tensor_name.split(".")
            for part in parts[:-1]:
                target_module = getattr(target_module, part)
            target_name = parts[-1]

        removed_marker = None
        target_param = None
        parameters = getattr(target_module, "_parameters", None)
        if isinstance(parameters, dict):
            target_param = parameters.get(target_name)
            if isinstance(target_param, torch.nn.Parameter):
                param_dict = getattr(target_param, "__dict__", None)
                if isinstance(param_dict, dict) and "_is_hf_initialized" in param_dict:
                    removed_marker = param_dict.pop("_is_hf_initialized")

        try:
            return original_fn(module, tensor_name, *args, **kwargs)
        finally:
            if removed_marker is not None and target_param is not None:
                param_dict = getattr(target_param, "__dict__", None)
                if isinstance(param_dict, dict) and "_is_hf_initialized" not in param_dict:
                    param_dict["_is_hf_initialized"] = removed_marker

    _patched_set_module_tensor_to_device._verl_hf_param_init_compat_patch = True
    big_modeling.set_module_tensor_to_device = _patched_set_module_tensor_to_device
    patched_any = True

    # Keep potential aliases consistent when modules imported function directly.
    try:
        from accelerate.utils import modeling as accelerate_modeling

        if getattr(accelerate_modeling, "set_module_tensor_to_device", None) is original_fn:
            accelerate_modeling.set_module_tensor_to_device = _patched_set_module_tensor_to_device
    except Exception:
        pass

    try:
        import transformers.core_model_loading as core_model_loading

        if getattr(core_model_loading, "set_module_tensor_to_device", None) is original_fn:
            core_model_loading.set_module_tensor_to_device = _patched_set_module_tensor_to_device
    except Exception:
        pass

    return patched_any
