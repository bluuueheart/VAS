"""Abstract base class for LLM providers."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator


class GenerationTimeoutError(TimeoutError):
    """Raised when one LLM generation turn exceeds its allotted runtime."""

    def __init__(
        self,
        message: str,
        *,
        partial_response: str = "",
        elapsed_seconds: Optional[float] = None,
        role: str = "",
        phase: str = "",
    ) -> None:
        super().__init__(str(message))
        self.partial_response = str(partial_response or "")
        self.elapsed_seconds = (
            float(elapsed_seconds) if elapsed_seconds not in (None, "") else None
        )
        self.role = str(role or "").strip()
        self.phase = str(phase or "").strip()


class AcceleratorFatalError(RuntimeError):
    """Raised when the accelerator runtime is no longer safe to reuse."""


def is_accelerator_fatal_error(exc: BaseException) -> bool:
    """Return True for CUDA/NCCL failures that poison the current process."""
    message = str(exc or "").lower()
    fatal_markers = (
        "cuda error: an illegal memory access was encountered",
        "cudaerrorillegaladdress",
        "illegal memory access",
        "device-side assert",
        "device side assert",
        "cublas_status_execution_failed",
        "cusolver_status_execution_failed",
        "nccl watchdog",
        "processgroupnccl",
    )
    return any(marker in message for marker in fatal_markers)


class BaseLLM(ABC):
    """Abstract base class for Language Model providers."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize LLM with model name and configuration.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """Generate a streaming response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Yields:
            Text chunks as they are generated
        """
        pass
    
    def format_messages(
        self,
        system_prompt: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Format messages for API call.
        
        Args:
            system_prompt: System-level instructions
            user_message: User query
            history: Optional conversation history
            
        Returns:
            Formatted message list
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": user_message})
        
        return messages

    # Context management helpers

    def count_tokens(self, text: str) -> int:
        """Estimate token count for *text*.

        Subclasses with an actual tokenizer should override for accuracy.
        Default heuristic: ~4 chars per token (mixed English / Chinese).
        """
        return max(1, len(text) // 4)

    def count_message_tokens(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> int:
        """Estimate token count for a structured chat payload.

        Providers with chat templates should override this to count the exact
        rendered prompt instead of a flat role/content approximation.
        """
        parts: List[str] = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            parts.append(f"{role}: {content}" if role else content)
        return self.count_tokens("\n".join(parts))

    def get_max_context_length(self) -> int:
        """Return the model's maximum context window in tokens.

        Subclasses should override to return the real value.
        """
        return 8192

    def prepare(self) -> None:
        """Optional eager initialization hook for providers with heavy lazy startup."""
        return None
