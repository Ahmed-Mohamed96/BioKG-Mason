# biokg/kg/schema_matcher.py

from typing import List, Set, Tuple, Optional
from pathlib import Path
import json

from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .triplets import Triplet
from .neo4j_client import Neo4jClient


class SchemaMatchResult(BaseModel):
    subject_label: str
    object_label: str
    relationship_type: str


SYSTEM_PROMPT = """
You are an expert biomedical ontology and schema normalization system.

You will be given:
1) A biomedical triplet (subject, predicate, object) with optional type hints.
2) The list of existing node labels in the current Neo4j graph.
3) The list of existing relationship types in the current Neo4j graph.
4) A Refernce text from which the triplet was extracted

Your tasks:
- Based on the Reference text, Try to choose a normalized label for the subject entity, a normalized label for the object entity, and a normalized relationship type for the predicate without losing the meaning and the representation of functional impact. It is important to preserve the meaning of the triplet.
- For relationship normalization, always favor labels and relations that describe a specific impact on the object from the subject. For example, 'increases' is favored over 'modulates', and 'activates' is favored over 'regulates', as they describe a causal connection.

Guidelines:
- Use broad but representative biomedical categories as labels, such as:

  PROTEIN, GENE, RNA, MICRO_RNA, DRUG, DISEASE, CELL, CELL_LINE,
  PATHWAY, TISSUE, HORMONE, CYTOKINE, etc.
- Avoid using General and not specific labels like Bio_entity
- When possible, reuse an existing label from the provided label list if it

  has the same meaning. For example:
  - "hormone" or "insulin" -> PROTEIN
  - "micro RNA", "miR-21" -> RNA or MICRO_RNA (choose the most appropriate)
- Avoid creating multiple labels with the same meaning.
- Relationship types should be uppercase with underscores, and mechanistic/functional/causal (e.g., ACTIVATES, INHIBITS, BINDS_TO, CAUSES).
- Don't favor relationship types with no specific causal effect, like modulates, regulates, and associated_with, even if such types exist in the list of existing relationship types, Unless there is no other choice. 
- When possible, reuse an existing relationship type from the given list if it has the same causal/functional/mechanestic meaning as the predicate.
- Normalization should be done considering the context of the given reference text.

You MUST output data that conforms exactly to the provided JSON schema.
If you are unsure, choose the closest reasonable labels and relationship type.
"""


class SchemaMatcher:
    """
    LLM-based schema matcher that:
    - Loads initial labels and relationship types from a JSON schema file.
    - Merges them with labels/types discovered in the KG.
    - Maintains an in-memory cache of known labels/types and updates it with every normalization.
    """

    def __init__(self, kg_client: Neo4jClient, chat_model: BaseChatModel, schema_path: Optional[str] = None):
        self.kg = kg_client
        self.chat_model = chat_model

        # Initialize cache of known labels and relationship types
        (
            self.known_entity_labels,
            self.known_relationship_types,
        ) = self._init_schema_cache(schema_path)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Existing node labels:\n{existing_labels}\n\n"
                    "Existing relationship types:\n{existing_rels}\n\n"
                    "Triplet:\n"
                    "- subject name: {subject_name}\n"
                    "- subject type hint: {subject_type_hint}\n"
                    "- predicate: {predicate}\n"
                    "- object name: {object_name}\n"
                    "- object type hint: {object_type_hint}\n\n"
                    "Reference Text:\n{reference_text}\n\n",
                ),
            ]
        )

        self.chain = self.prompt | self.chat_model.with_structured_output(SchemaMatchResult)


    def _init_schema_cache(
        self, schema_path: Optional[str]
    ) -> Tuple[Set[str], Set[str]]:
        labels: Set[str] = set()
        rel_types: Set[str] = set()

        # 1) Load from JSON schema file if provided
        if schema_path:
            path = Path(schema_path)
            if path.is_file():
                with path.open("r") as f:
                    data = json.load(f)
                labels.update(data.get("node_labels", []))
                rel_types.update(data.get("relationship_types", []))

        # 2) Merge with any existing labels/types from the KG
        labels.update(self.kg.get_existing_labels())
        rel_types.update(self.kg.get_existing_relationship_types())

        return labels, rel_types

    def normalize_triplet(self, triplet: Triplet) -> Triplet:
        """
        Ask the LLM to normalize the triplet's subject type, object type,
        and relationship type based on the current cache of known
        labels and relationship types.
        """
        result: SchemaMatchResult = self.chain.invoke(
            {
                "existing_labels": sorted(self.known_entity_labels),
                "existing_rels": sorted(self.known_relationship_types),
                "subject_name": triplet.subject.name,
                "subject_type_hint": triplet.subject.type or "",
                "predicate": triplet.predicate,
                "object_name": triplet.obj.name,
                "object_type_hint": triplet.obj.type or "",
            }
        )

        subj_label = result.subject_label or triplet.subject.type
        obj_label = result.object_label or triplet.obj.type
        rel_type = result.relationship_type or triplet.predicate

        triplet.subject.type = subj_label
        triplet.obj.type = obj_label
        triplet.predicate = rel_type

        # Update in-memory cache with any new labels/types the LLM introduces
        self.known_entity_labels.update([subj_label, obj_label])
        self.known_relationship_types.add(rel_type)

        return triplet