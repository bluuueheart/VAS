"""Video processing utilities."""

from importlib import import_module

__all__ = ["VideoSegmenter", "ASRProcessor", "VisualProcessor"]

_EXPORTS = {
    "VideoSegmenter": (".segmentation", "VideoSegmenter"),
    "ASRProcessor": (".asr", "ASRProcessor"),
    "VisualProcessor": (".caption", "VisualProcessor"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
