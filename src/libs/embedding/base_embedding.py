"""Abstract base class for Embedding providers.

This module defines the pluggable interface for Embedding service providers,
enabling seamless switching between different backends (OpenAI, Local/BGE, etc.)
through configuration-driven instantiation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Optional

class BaseEmbedding(ABC):
    """
    嵌入提供者的抽象基类。
    所有嵌入实现都必须继承自此类并实现embed()方法。
    这确保了不同提供者（OpenAI、本地/BGE、Ollama等）之间的一致接口。

    应用的设计原则：
    可插拔：子类可以在不修改上游代码的情况下进行替换
    可观测：接受可选的TraceContext以实现可观测性集成
    配置驱动：通过基于设置的工厂创建实例
    批量优先：为批量处理设计以最大化效率
    """

    @abstractmethod
    def embed(self, texts: List[str], trace: Optional[Any] = None, **kwargs: Any) -> List[List[float]]:
        """Generate embeddings for a batch of texts.
            
            Args:
                texts: List of text strings to embed. Must not be empty.
                trace: Optional TraceContext for observability (reserved for Stage F).
                **kwargs: Provider-specific parameters (batch_size, dimensions, etc.).
            
            Returns:
                List of embedding vectors, where each vector is a list of floats.
                The length of the outer list matches len(texts).
                The length of each inner list (vector dimension) is provider-dependent.
            
            Raises:
                ValueError: If texts list is empty or contains invalid entries.
                RuntimeError: If the embedding provider call fails.
            
            Example:
                >>> embeddings = embedding.embed(["hello", "world"])
                >>> len(embeddings)  # 2 vectors
                >>> len(embeddings[0])  # dimension (e.g., 1536 for OpenAI)
        """
        pass

    def validate_texts(self, texts: List[str]) -> None:
        """Validate input text list.
        
        Args:
            texts: List of texts to validate.
        
        Raises:
        """
        if not texts:
            raise ValueError("Texts can not be empty")

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError("Text at index {i} is not a string (type:{type(text).__name__})")
            if not text.strip():
                raise ValueError(
                    f"Text at index {i} is empty or whitespace-only. "
                    "Embedding providers typically reject empty strings."
                )
            
    def getDimendion(self) -> int:
        """Get the dimensionality of embeddings produced by this provider.
        
        Returns:
            The vector dimension (e.g., 1536 for OpenAI text-embedding-3-small).
        
        Raises:
            NotImplementedError: If the subclass doesn't override this method.
        
        Note:
            Subclasses should override this method to return their specific dimension.
            This is useful for validation and storage configuration.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_dimension() method"
        )