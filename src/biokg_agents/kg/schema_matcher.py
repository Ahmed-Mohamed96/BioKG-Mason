# biokg/kg/schema_matcher.py

from typing import List, Dict

from ..llm import LLMClient
from ..models import Triplet
from .neo4j_client import Neo4jClient


SCHEMA_MATCHER_PROMPT = """
You are an expert biomedical ontology and schema normalization system.

You will be given:
1) A biomedical triplet (subject, predicate, object) with optional type hints.
2) The list of existing node labels in the current Neo4j graph.
3) The list of existing relationship types in the current Neo4j graph.

Your tasks:
- Choose a normalized label for the subject entity.
- Choose a normalized label for the object entity.
- Choose a normalized relationship type for the predicate.

Guidelines:
- Use broad but representative biomedical categories as labels, such as:

  PROTEIN, GENE, RNA, MICRO_RNA, DRUG, DISEASE, CELL, CELL_LINE,
  PATHWAY, TISSUE, HORMONE, CYTOKINE, BIO_ENTITY, etc.
- When possible, reuse an existing label from the provided label list if it

  has the same meaning. For example:
  - "hormone" or "insulin" -> PROTEIN
  - "micro RNA", "miR-21" -> RNA or MICRO_RNA (choose the most appropriate)
- Avoid creating multiple labels with the same meaning.
- Relationship types should be uppercase with underscores (e.g., ACTIVATES, INHIBITS,

  BINDS_TO, ASSOCIATED_WITH, CAUSES).
- When possible, reuse an existing relationship type from the given list

  if it has the same meaning as the predicate.

Return ONLY valid JSON with this structure:

{{
  "subject_label": "PROTEIN",
  "object_label": "GENE",
  "relationship_type": "ACTIVATES"
}}

If you are unsure, choose the closest reasonable labels and relationship type.

Now process the following:

Existing node labels:
{existing_labels}

Existing relationship types:
{existing_rels}

Triplet:
- subject name: "{subject_name}"
- subject type hint: "{subject_type_hint}"
- predicate: "{predicate}"
- object name: "{object_name}"
- object type hint: "{object_type_hint}"

"""


class SchemaMatcher:
    """
    LLM-based schema matcher that normalizes entity labels and relationship types.
    It uses the current Neo4j schema (labels and relationship types) as context
    and asks an LLM to choose normalized labels/types that avoid duplicates and
    remain semantically broad and representative.
    """

    def __init__(self, kg_client: Neo4jClient, llm_client: LLMClient):
        self.kg = kg_client
        self.llm = llm_client

    def _build_prompt(self, triplet: Triplet) -> str:
        existing_labels = self.kg.get_existing_labels()
        existing_rels = self.kg.get_existing_relationship_types()

        prompt = SCHEMA_MATCHER_PROMPT.format(
            existing_labels=existing_labels,
            existing_rels=existing_rels,
            subject_name=triplet.subject.name,
            subject_type_hint=triplet.subject.type or "",
            predicate=triplet.predicate,
            object_name=triplet.obj.name,
            object_type_hint=triplet.obj.type or "",
        )
        return prompt

    def normalize_triplet(self, triplet: Triplet) -> Triplet:
        """
        Ask the LLM to normalize the triplet's subject type, object type,
        and relationship type based on existing labels and relationship types
        in the KG.
        """
        prompt = self._build_prompt(triplet)
        raw = self.llm.generate(prompt)

        # Attempt to extract JSON
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            raw_json = raw[start:end]
        except ValueError:
            # Could not find JSON; leave original types unchanged
            return triplet

        import json

        try:
            data: Dict = json.loads(raw_json)
        except json.JSONDecodeError:
            # Invalid JSON; leave original types unchanged
            return triplet

        subj_label = data.get("subject_label") or triplet.subject.type or "BIO_ENTITY"
        obj_label = data.get("object_label") or triplet.obj.type or "BIO_ENTITY"
        rel_type = data.get("relationship_type") or triplet.predicate

        triplet.subject.type = subj_label
        triplet.obj.type = obj_label
        triplet.predicate = rel_type

        return triplet