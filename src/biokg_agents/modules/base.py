from abc import ABC, abstractmethod
from typing import List


class LLMClient(ABC):
    """
    Abstract interface for chat/completion models.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate text given a prompt.
        """
        raise NotImplementedError


class EmbeddingsClient(ABC):
    """
    Abstract interface for embedding models.
    """

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Return a vector embedding for the given text.
        """
        raise NotImplementedError