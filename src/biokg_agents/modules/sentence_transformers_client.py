# src/biokg_agents/modules/sentence_transformers_client.py

from typing import List

from sentence_transformers import SentenceTransformer

from .base import EmbeddingsClient


class SentenceTransformersEmbeddingsClient(EmbeddingsClient):
    """
    Embeddings client using sentence-transformers.

    This is used when a local LLM (e.g., Ollama) is configured,
    so that embeddings are computed locally instead of via an API.
    """

    def __init__(self, model: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        :param model: sentence-transformers model name
                      e.g. "sentence-transformers/all-mpnet-base-v2"
        """
        self.model_name = model
        self.model = SentenceTransformer(model, trust_remote_code=True)

    def embed(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()