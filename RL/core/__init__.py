"""Core module exports.

Keep infrastructure modules such as ``core.accelerator_queue`` lightweight:
the language-model stack is imported only when callers request its public
symbols.  This prevents a CPU-side scheduler or queue probe from loading
transformers and torch before any inference worker starts.
"""

__all__ = ["BaseLLM", "get_llm_provider"]


def __getattr__(name: str):
    if name in __all__:
        from .llm import BaseLLM, get_llm_provider

        return {"BaseLLM": BaseLLM, "get_llm_provider": get_llm_provider}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
