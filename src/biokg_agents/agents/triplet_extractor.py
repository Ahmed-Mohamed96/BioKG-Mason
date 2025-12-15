import json
from typing import List

from ..llm import LLMClient
from ..models import Entity, Triplet


TRIPLET_EXTRACTION_PROMPT = """
You are an expert biomedical information extraction system.

Given the following paragraph from a biomedical research article,
extract *all* biomedical triplets of the form:

- subject (entity)
- predicate (relationship phrase)
- object (entity)
- subject_type (e.g., protein, gene, RNA, disease, drug, cell line, pathway, etc.)
- object_type (same as subject_type)

Return ONLY valid JSON with the following structure:

[
  {
    "subject": "...",
    "subject_type": "...",
    "predicate": "...",
    "object": "...",
    "object_type": "..."
  },
  ...
]

Rules:
- Only include triplets that are biomedical and meaningful.
- Use concise but specific names for entities (e.g., "TNF-alpha", "p53", "breast cancer").
- Use natural language predicates that are causal/mechanistic/functional (e.g., "inhibits", "activates", "increases_levels_of").
- If you are unsure about the type, guess the closest from: protein, gene, RNA, disease, drug, pathway, cell line, cell, tissue, hormone, micro RNA, cytokine.

Paragraph:
"""


class TripletExtractor:
    """
    Agent that extracts biomedical triplets from paragraphs using an LLM.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract_triplets(self, paragraph: str) -> List[Triplet]:
        """
        Run LLM to extract triplets from a paragraph.
        """
        prompt = TRIPLET_EXTRACTION_PROMPT + paragraph
        raw = self.llm.generate(prompt)

        # Try to parse JSON; handle wrapping text by finding first '['
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            raw_json = raw[start:end]
        except ValueError:
            return []

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            return []

        triplets: List[Triplet] = []
        if not isinstance(data, list):
            return triplets

        for item in data:
            try:
                subj = Entity(
                    name=item["subject"],
                    type=item.get("subject_type"),
                )
                obj = Entity(
                    name=item["object"],
                    type=item.get("object_type"),
                )
                pred = item["predicate"]
                triplets.append(Triplet(subject=subj, predicate=pred, obj=obj))
            except KeyError:
                continue

        return triplets