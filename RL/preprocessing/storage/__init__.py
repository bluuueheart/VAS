"""Storage modules for preprocessed data.

Keep this package import-light.  The embedding runtime is used by planner
grounding and should not import the vector-index implementation (and its
optional service clients) merely because ``preprocessing.storage`` is
referenced.  Heavy modules are resolved only when their public export is
actually requested.
"""

from importlib import import_module

__all__ = [
    "Qwen3VLEmbeddingRuntimeEncoder",
    "Qwen3VLEmbeddingGenerator",
    "VideoSegmentVectorStore",
]

_EXPORTS = {
    "Qwen3VLEmbeddingRuntimeEncoder": (".qwen3_vl_embedding_runtime", "Qwen3VLEmbeddingRuntimeEncoder"),
    "Qwen3VLEmbeddingGenerator": (".vector_store", "Qwen3VLEmbeddingGenerator"),
    "VideoSegmentVectorStore": (".vector_store", "VideoSegmentVectorStore"),
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
