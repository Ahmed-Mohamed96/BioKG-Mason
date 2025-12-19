from pathlib import Path
from typing import List, Optional, Dict

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from .config import Config
from .modules.base import EmbeddingsClient
from .modules.sentence_transformers_client import SentenceTransformersEmbeddingsClient
from .modules.openai_client import OpenAIEmbeddingsClient
from .modules.pdf_reader import PDFReader, DocSegment
from .modules.neo4j_client import Neo4jClient
from .modules.entity_matcher import EntityMatcher
from .modules.schema_matcher import SchemaMatcher
from .modules.triplet_extractor import TripletExtractor
from .modules.entity_match_validator import EntityMatchValidator
from .modules.triplets import Triplet

from tqdm import tqdm


def _build_chat_model(cfg: Config) -> BaseChatModel:
    """
    Build a LangChain chat model (ChatOpenAI or ChatOllama)
    based on config.llm.provider and config.llm.model.
    """
    provider = cfg.llm.provider.lower()
    temp = cfg.llm.temperature
    if provider == "openai":
        # OPENAI_API_KEY is read from environment by ChatOpenAI
        return ChatOpenAI(model=cfg.llm.model, temperature=temp)
    elif provider == "ollama":
        return ChatOllama(model=cfg.llm.model, temperature=temp)
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

        # LangChain chat model (OpenAI or Ollama)
        self.chat_model = _build_chat_model(cfg)

        # Embeddings (OpenAI or sentence-transformers)
        self.embeddings_client = _build_embeddings_client(cfg)

        self.kg_client = Neo4jClient(
            uri=cfg.neo4j.uri,
            user=cfg.neo4j.user,
            password=cfg.neo4j.password,
            database=cfg.neo4j.database,
        )

        self.pdf_reader = PDFReader()

        # LLM-based agents
        self.triplet_extractor = TripletExtractor(self.chat_model)
        self.entity_match_validator = EntityMatchValidator(self.chat_model)

        schema_path = self.cfg.schema.file if self.cfg.schema else None
        self.schema_matcher = SchemaMatcher(self.kg_client, self.chat_model, schema_path=schema_path,)

        # Embedding-based entity matcher
        self.entity_matcher = EntityMatcher(
            kg_client=self.kg_client,
            embeddings_client=self.embeddings_client,
            similarity_threshold=self.cfg.pipeline.similarity_threshold,
        )

    @staticmethod
    def _segment_to_prompt_text(segment: DocSegment) -> str:
        """
        Convert a DocSegment (paragraph or table) to the text that will be
        fed to the TripletExtractor.
        """
        if segment.kind == "paragraph":
            return segment.text
        else:
            # For tables, prepend a hint so the LLM understands context
            return (
                "The following is a biomedical results table flattened into text.\n"
                "Each line is a row, and each 'column: value' is a cell.\n\n"
                + segment.text

            )

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

        mention = f"{entity.name} (type: {entity.type})"
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

    def _process_triplet(
        self,
        triplet: Triplet,
        pmid: str,
        source_text: str,
        source_kind: str,
        page: int,
        table_index: Optional[int] = None,
    ):
        """
        Resolve entities, validate matches via LLM, normalize schema via LLM,
        and persist triplet to Neo4j with full provenance.
        `source_text` can be a paragraph or a flattened table.
        """
        # 1. Resolve entities using embeddings
        resolved_triplet, embeddings_map, matches_map = self.entity_matcher.resolve_triplet(
            triplet
        )

        print(f"[Triplet] [Resolved] {resolved_triplet.subject.name} - {resolved_triplet.predicate} >> {resolved_triplet.obj.name}")

        # 2. LLM validates whether embedding-based matches are correct.
        #    Use source_text as "context" (paragraph or table text).
        self._validate_entity_match(
            role="subject",
            triplet=resolved_triplet,
            paragraph=source_text,
            match_info=matches_map.get("subject"),
        )
        self._validate_entity_match(
            role="object",
            triplet=resolved_triplet,
            paragraph=source_text,
            match_info=matches_map.get("object"),
        )

        # 3. Normalize schema via LLM
        normalized_triplet = self.schema_matcher.normalize_triplet(resolved_triplet, source_text)

        print(f"[Triplet] [Schema-Normalized] {normalized_triplet.subject.name} - {normalized_triplet.predicate} >> {normalized_triplet.obj.name}")

        # 4. Upsert subject and object nodes
        subj = normalized_triplet.subject
        obj = normalized_triplet.obj

        subj_emb = embeddings_map["subject"]
        obj_emb = embeddings_map["object"]

        subj_entity_id = self.kg_client.upsert_entity(
            name=subj.name,
            label=subj.type,
            embedding=subj_emb,
            entity_id=subj.entity_id,
        )
        obj_entity_id = self.kg_client.upsert_entity(
            name=obj.name,
            label=obj.type,
            embedding=obj_emb,
            entity_id=obj.entity_id,
        )

        # Update local cache
        self.entity_matcher.register_new_entity(
            {
                "entity_id": subj_entity_id,
                "name": subj.name,
                "labels": [subj.type],
                "embedding": subj_emb,
            }
        )
        self.entity_matcher.register_new_entity(
            {
                "entity_id": obj_entity_id,
                "name": obj.name,
                "labels": [obj.type],
                "embedding": obj_emb,
            }
        )

        # 5. Create relationship with extended provenance
        self.kg_client.create_relationship(
            subj_entity_id=subj_entity_id,
            obj_entity_id=obj_entity_id,
            rel_type=normalized_triplet.predicate,
            pmid=pmid,
            source_text=source_text,
            source_kind=source_kind,
            page=page,
            table_index=table_index,
        )

    def process_pdf(self, pdf_path: str):
        """
        Process a single PDF file into the KG.
        Assumes filename (without extension) is the PMID.
        Now processes both paragraphs and tables.
        """
        pmid = Path(pdf_path).stem
        segments = self.pdf_reader.extract_segments(pdf_path)

        counter = 1
        triplet_counter = 1
        for seg in tqdm(segments, desc="Processing Text Segments", unit="Segment"):
            
            print("\n")
            print("\n")
            print(f"========== Segment {counter}/{len(segments)} ==========")
            print(seg.text)
            print(f"-----------------------------------------")
            print("\n")
            source_text = self._segment_to_prompt_text(seg)
            triplets: List[Triplet] = self.triplet_extractor.extract_triplets(source_text)
            print(f"[SEG {counter}] Number of Triplets = {len(triplets)}")

            for triplet in triplets:
                try:
                    print(f"[Triplet {triplet_counter}] [Original] {triplet.subject.name} - {triplet.predicate} >> {triplet.obj.name}")
                    # Attach provenance to triplet metadata (optional)
                    triplet.metadata["pmid"] = pmid
                    triplet.metadata["source_kind"] = seg.kind
                    triplet.metadata["page"] = seg.page
                    triplet.metadata["table_index"] = seg.table_index

                    self._process_triplet(
                        triplet=triplet,
                        pmid=pmid,
                        source_text=source_text,
                        source_kind=seg.kind,
                        page=seg.page,
                        table_index=seg.table_index,
                    )

                    triplet_counter += 1
                    print("\n")
                    print(f"-----------------------------------------")
                except Exception as e:
                    print(f"[FAIL] {e}")
                    continue
            counter += 1
            print(f"=========================================")

    def process_all_pdfs(self):
        """
        Process all PDF files in the configured PDF directory.
        """
        pdf_dir = Path(self.cfg.pdf.input_dir)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))

        for pdf_file in pdf_files:
            print(f"[PDF] Process Article: {Path(pdf_file).stem}")
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