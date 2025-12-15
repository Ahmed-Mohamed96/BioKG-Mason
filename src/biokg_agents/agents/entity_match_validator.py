# biokg/agents/entity_match_validator.py

import json
from typing import List, Dict, Optional

from ..llm import LLMClient


ENTITY_MATCH_VALIDATION_PROMPT = """
You are an expert biomedical entity disambiguation system.

You will be given:
1) A paragraph from a biomedical article.
2) A mention string that appears in the paragraph.
3) A candidate existing node from a biomedical knowledge graph, with:
   - canonical name
   - one or more semantic labels (e.g., PROTEIN, GENE, DISEASE).

Your task:
Determine if, in the context of the paragraph, the mention refers to the SAME
biomedical entity as the candidate node.

Answer the question:
"Does the mention refer to this candidate entity?"

Rules:
- Consider biological meaning and context, not just string similarity.
- If you are unsure, answer "false" (i.e., do NOT merge them).

Return ONLY valid JSON in this format:

{{
  "is_same_entity": true,
  "reason": "short explanation"
}}

Now analyze:

Paragraph:
\"\"\"{paragraph}\"\"\"

Mention in the paragraph:
"{mention}"

Candidate node:
- name: "{candidate_name}"
- labels: {candidate_labels}

Is the mention the SAME biomedical entity as this candidate node?
"""


class EntityMatchValidator:
    """
    LLM-based validator to check whether an embedding-based entity match
    is semantically correct given the paragraph context.

    If the validator rejects the match (is_same_entity = false),
    the pipeline will treat the entity as NEW and create a new node.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def validate_match(
        self,
        mention: str,
        paragraph: str,
        candidate_name: str,
        candidate_labels: List[str],
    ) -> bool:
        """
        Return True if the LLM confirms that the mention and candidate node
        refer to the same entity; False otherwise.
        """
        prompt = ENTITY_MATCH_VALIDATION_PROMPT.format(
            paragraph=paragraph,
            mention=mention,
            candidate_name=candidate_name,
            candidate_labels=candidate_labels,
        )
        raw = self.llm.generate(prompt)

        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            raw_json = raw[start:end]
        except ValueError:
            # No JSON found; be conservative and reject the match
            return False

        try:
            data: Dict = json.loads(raw_json)
        except json.JSONDecodeError:
            # Invalid JSON; be conservative and reject the match
            return False

        is_same = data.get("is_same_entity")
        # If not explicitly true, treat as False
        return bool(is_same) is True