# biokg/kg/schema_matcher.py

from typing import List

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
    LLM-based schema matcher that normalizes entity labels and relationship types
    using a LangChain chat model (ChatOpenAI or ChatOllama) with structured output.
    """

    def __init__(self, kg_client: Neo4jClient, chat_model: BaseChatModel):
        self.kg = kg_client
        self.chat_model = chat_model

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

    def normalize_triplet(self, triplet: Triplet, reference_text: str) -> Triplet:
        """
        Ask the LLM to normalize the triplet's subject type, object type,
        and relationship type based on existing labels and relationship types
        in the KG.
        """
        existing_labels = self.kg.get_existing_labels()
        existing_rels = self.kg.get_existing_relationship_types()

        result: SchemaMatchResult = self.chain.invoke(
            {
                "existing_labels": existing_labels,
                "existing_rels": existing_rels,
                "subject_name": triplet.subject.name,
                "subject_type_hint": triplet.subject.type or "",
                "predicate": triplet.predicate,
                "object_name": triplet.obj.name,
                "object_type_hint": triplet.obj.type or "",
                "reference_text": reference_text,
            }
        )

        triplet.subject.type = result.subject_label or triplet.subject.type
        triplet.obj.type = result.object_label or triplet.obj.type
        triplet.predicate = result.relationship_type or triplet.predicate

        return triplet