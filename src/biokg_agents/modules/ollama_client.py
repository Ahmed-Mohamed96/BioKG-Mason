import requests
from typing import List

from .base import LLMClient, EmbeddingsClient


class OllamaLLMClient(LLMClient):
    """
    LLM client for local Ollama models.
    Assumes Ollama running at http://localhost:11434.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns list of messages; adapt if needed to actual API format
        return data.get("message", {}).get("content", "")


class OllamaEmbeddingsClient(EmbeddingsClient):
    """
    Embeddings using Ollama /api/embeddings.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data.get("embedding", [])