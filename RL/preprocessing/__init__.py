"""
Preprocessing module for video data ingestion.
"""

from importlib import import_module

__all__ = [
    "VideoPreprocessingPipeline",
    "PipelineConfig",
    "VideoSegmenter",
    "ASRProcessor",
    "VisualProcessor",
    "Qwen3VLEmbeddingRuntimeEncoder",
    "KnowledgeGraphBuilder",
    "Entity",
    "Relation",
    "MetadataSynthesizer",
    "Qwen3VLEmbeddingGenerator",
    "VideoSegmentVectorStore",
]

_EXPORTS = {
    "VideoPreprocessingPipeline": (".pipeline", "VideoPreprocessingPipeline"),
    "PipelineConfig": (".pipeline", "PipelineConfig"),
    "VideoSegmenter": (".video.segmentation", "VideoSegmenter"),
    "ASRProcessor": (".video.asr", "ASRProcessor"),
    "VisualProcessor": (".video.caption", "VisualProcessor"),
    "Qwen3VLEmbeddingRuntimeEncoder": (".storage.qwen3_vl_embedding_runtime", "Qwen3VLEmbeddingRuntimeEncoder"),
    "KnowledgeGraphBuilder": (".knowledge_graph", "KnowledgeGraphBuilder"),
    "Entity": (".knowledge_graph", "Entity"),
    "Relation": (".knowledge_graph", "Relation"),
    "MetadataSynthesizer": (".metadata_synthesis", "MetadataSynthesizer"),
    "Qwen3VLEmbeddingGenerator": (".storage.vector_store", "Qwen3VLEmbeddingGenerator"),
    "VideoSegmentVectorStore": (".storage.vector_store", "VideoSegmentVectorStore"),
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
