from pathlib import Path
from typing import List, Optional, Dict

from .config import Config
from .llm import (
    LLMClient,
    EmbeddingsClient,
    OpenAILLMClient,
    OpenAIEmbeddingsClient,
    OllamaLLMClient,
    SentenceTransformersEmbeddingsClient,
)
from .pdf import PDFReader
from .kg import Neo4jClient, EntityMatcher, SchemaMatcher
from .agents import TripletExtractor, EntityMatchValidator
from .models import Triplet


def _build_llm_client(cfg: Config) -> LLMClient:
    provider = cfg.llm.provider.lower()
    if provider == "openai":
        return OpenAILLMClient(model=cfg.llm.model)
    elif provider == "ollama":
        return OllamaLLMClient(model=cfg.llm.model)
    else:
        raise ValueError(f"Unsupported LLM provider: {cfg.llm.provider}")


def _build_embeddings_client(cfg: Config) -> EmbeddingsClient:
    provider = cfg.embeddings.provider.lower()
    if provider == "openai":
        return OpenAIEmbeddingsClient(model=cfg.embeddings.model)
    elif provider == "ollama":
        # When using Ollama (local LLM), embeddings are computed locally
        # using sentence-transformers.
        return SentenceTransformersEmbeddingsClient(model=cfg.embeddings.model)
    else:
        raise ValueError(f"Unsupported embeddings provider: {cfg.embeddings.provider}")


class PDFToKGPipeline:
    """
    Orchestrates:
    PDF -> paragraphs -> triplets -> entity resolution (embeddings + LLM check)
    -> schema matching via LLM -> Neo4j.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.llm_client = _build_llm_client(cfg)
        self.embeddings_client = _build_embeddings_client(cfg)

        self.kg_client = Neo4jClient(
            uri=cfg.neo4j.uri,
            user=cfg.neo4j.user,
            password=cfg.neo4j.password,
        )

        self.pdf_reader = PDFReader()
        self.triplet_extractor = TripletExtractor(self.llm_client)

        # Embedding-based matcher
        self.entity_matcher = EntityMatcher(
            kg_client=self.kg_client,
            embeddings_client=self.embeddings_client,
            similarity_threshold=cfg.pipeline.similarity_threshold,
        )

        # LLM-based validator for entity matches
        self.entity_match_validator = EntityMatchValidator(self.llm_client)

        # LLM-based schema matcher
        self.schema_matcher = SchemaMatcher(self.kg_client, self.llm_client)

    def _validate_entity_match(
        self,
        role: str,
        triplet: Triplet,
        paragraph: str,
        match_info: Optional[Dict],
    ) -> None:
        """
        Use the LLM-based EntityMatchValidator to confirm or reject an
        embedding-based match.

        If rejected, sets triplet.subject/entity_id or triplet.obj/entity_id
        to None so a new node will be created.
        """
        if match_info is None:
            return  # no candidate match, nothing to validate

        if role == "subject":
            entity = triplet.subject
        else:
            entity = triplet.obj

        mention = entity.name
        candidate_name = match_info.get("name", "")
        candidate_labels = match_info.get("labels") or []

        is_same = self.entity_match_validator.validate_match(
            mention=mention,
            paragraph=paragraph,
            candidate_name=candidate_name,
            candidate_labels=candidate_labels,
        )

        if not is_same:
            # Reject the match: treat as new entity
            entity.entity_id = None

    def _process_triplet(self, triplet: Triplet, pmid: str, paragraph: str):
        """
        Resolve entities, validate matches via LLM, normalize schema via LLM,
        and persist triplet to Neo4j with full provenance.
        """
        # 1. Resolve entities using embeddings against existing nodes
        resolved_triplet, embeddings_map, matches_map = self.entity_matcher.resolve_triplet(
            triplet
        )

        # 2. Let LLM validate whether the embedding-based matches are correct
        self._validate_entity_match(
            role="subject",
            triplet=resolved_triplet,
            paragraph=paragraph,
            match_info=matches_map.get("subject"),
        )
        self._validate_entity_match(
            role="object",
            triplet=resolved_triplet,
            paragraph=paragraph,
            match_info=matches_map.get("object"),
        )

        # 3. Normalize schema (entity labels and relationship type) via LLM
        normalized_triplet = self.schema_matcher.normalize_triplet(resolved_triplet)

        # 4. Upsert subject and object nodes with final labels
        subj = normalized_triplet.subject
        obj = normalized_triplet.obj

        subj_emb = embeddings_map["subject"]
        obj_emb = embeddings_map["object"]

        subj_entity_id = self.kg_client.upsert_entity(
            name=subj.name,
            label=subj.type or "BIO_ENTITY",
            embedding=subj_emb,
            entity_id=subj.entity_id,  # None => new node
        )
        obj_entity_id = self.kg_client.upsert_entity(
            name=obj.name,
            label=obj.type or "BIO_ENTITY",
            embedding=obj_emb,
            entity_id=obj.entity_id,  # None => new node
        )

        # Update local entity cache in EntityMatcher for future matches
        self.entity_matcher.register_new_entity(
            {
                "entity_id": subj_entity_id,
                "name": subj.name,
                "labels": [subj.type or "BIO_ENTITY"],
                "embedding": subj_emb,
            }
        )
        self.entity_matcher.register_new_entity(
            {
                "entity_id": obj_entity_id,
                "name": obj.name,
                "labels": [obj.type or "BIO_ENTITY"],
                "embedding": obj_emb,
            }
        )

        # 5. Create relationship with provenance (PMID, paragraph)
        self.kg_client.create_relationship(
            subj_entity_id=subj_entity_id,
            obj_entity_id=obj_entity_id,
            rel_type=normalized_triplet.predicate,
            pmid=pmid,
            paragraph=paragraph,
        )

    def process_pdf(self, pdf_path: str):
        """
        Process a single PDF file into the KG.
        Assumes filename (without extension) is the PMID.
        """
        pmid = Path(pdf_path).stem
        paragraphs = self.pdf_reader.extract_paragraphs(pdf_path)

        for paragraph in paragraphs:
            triplets: List[Triplet] = self.triplet_extractor.extract_triplets(paragraph)
            for triplet in triplets:
                # Attach provenance to triplet metadata (optional)
                triplet.metadata["pmid"] = pmid
                triplet.metadata["paragraph"] = paragraph
                self._process_triplet(triplet, pmid, paragraph)

    def process_all_pdfs(self):
        """
        Process all PDF files in the configured PDF directory.
        """
        pdf_dir = Path(self.cfg.pdf.input_dir)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))

        for pdf_file in pdf_files:
            self.process_pdf(str(pdf_file))

    def close(self):
        """
        Close Neo4j connection.
        """
        self.kg_client.close()


if __name__ == "__main__":
    from .config import load_config

    cfg = load_config("config.yaml")
    pipeline = PDFToKGPipeline(cfg)
    try:
        pipeline.process_all_pdfs()
    finally:
        pipeline.close()