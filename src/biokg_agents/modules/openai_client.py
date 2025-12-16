# src/biokg_agents/modules/openai_client.py

import os
from typing import List
from openai import OpenAI

from .base import LLMClient, EmbeddingsClient


class OpenAILLMClient(LLMClient):
    """
    LLM client using OpenAI chat/completion models.
    """

    def __init__(self, model: str):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You extract structured biomedical knowledge."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content


class OpenAIEmbeddingsClient(EmbeddingsClient):
    """
    Embeddings client using OpenAI embedding models.
    """

    def __init__(self, model: str):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return resp.data[0].embedding