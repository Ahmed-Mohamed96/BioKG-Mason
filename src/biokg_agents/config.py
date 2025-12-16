import os
from dataclasses import dataclass
from typing import Any, Dict
import yaml


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.0


@dataclass
class EmbeddingsConfig:
    provider: str
    model: str


@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str


@dataclass
class PDFConfig:
    input_dir: str


@dataclass
class PipelineConfig:
    similarity_threshold: float = 0.82

@dataclass
class SchemaConfig:
    file: str


@dataclass
class Config:
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    neo4j: Neo4jConfig
    pdf: PDFConfig
    pipeline: PipelineConfig


def load_config(path: str) -> Config:
    """
    Load YAML config file and return a Config object.
    """
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    llm_cfg = LLMConfig(**raw["llm"])
    emb_cfg = EmbeddingsConfig(**raw["embeddings"])
    neo4j_cfg = Neo4jConfig(**raw["neo4j"])
    pdf_cfg = PDFConfig(**raw["pdf"])
    pipeline_cfg = PipelineConfig(**raw.get("pipeline", {}))

    return Config(
        llm=llm_cfg,
        embeddings=emb_cfg,
        neo4j=neo4j_cfg,
        pdf=pdf_cfg,
        pipeline=pipeline_cfg,
    )

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    llm_cfg = LLMConfig(**raw["llm"])
    emb_cfg = EmbeddingsConfig(**raw["embeddings"])
    neo4j_cfg = Neo4jConfig(**raw["neo4j"])
    pdf_cfg = PDFConfig(**raw["pdf"])
    pipeline_cfg = PipelineConfig(**raw.get("pipeline", {}))

    schema_cfg = None
    if "schema" in raw:
        schema_cfg = SchemaConfig(**raw["schema"])

    return Config(
        llm=llm_cfg,
        embeddings=emb_cfg,
        neo4j=neo4j_cfg,
        pdf=pdf_cfg,
        pipeline=pipeline_cfg,
        schema=schema_cfg,
    )