from typing import List

from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


class EntityMatchValidationResult(BaseModel):
    is_same_entity: bool
    reason: str


SYSTEM_PROMPT = """
You are an expert biomedical entity disambiguation system.

You will be given:
1) A paragraph or table from a biomedical article.
2) A mention string extracted from the paragraph or table with assigned type by a curator.
3) A candidate existing node from a biomedical knowledge graph, with:
   - canonical name
   - one or more semantic labels (e.g., PROTEIN, GENE, DISEASE).

Your task:
Determine if, in the context of the paragraph or table, the mention refers to the SAME
biomedical entity as the candidate node.

Rules:
- Consider biological meaning and context, not just string similarity.
- If the name and label strings match, answer "true" (i.e., Merge them).
- If you are unsure, answer "false" (i.e., do NOT merge them).

You MUST output data that conforms exactly to the provided JSON schema.
"""


class EntityMatchValidator:
    """
    LLM-based validator to check whether an embedding-based entity match
    is semantically correct given the paragraph context.

    If the validator rejects the match (is_same_entity = false),
    the pipeline should treat the entity as NEW and create a new node.
    """

    def __init__(self, chat_model: BaseChatModel):
        self.chat_model = chat_model

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Paragraph:\n\"\"\"{paragraph}\"\"\"\n\n"
                    "Mention in the paragraph:\n\"{mention}\"\n\n"
                    "Candidate node:\n"
                    "- name: \"{candidate_name}\"\n"
                    "- labels: {candidate_labels}\n\n"
                    "Does the mention refer to this candidate entity?\n",
                ),
            ]
        )

        self.chain = self.prompt | self.chat_model.with_structured_output(
            EntityMatchValidationResult
        )

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
        result: EntityMatchValidationResult = self.chain.invoke(
            {
                "paragraph": paragraph,
                "mention": mention,
                "candidate_name": candidate_name,
                "candidate_labels": candidate_labels,
            }
        )
        return bool(result.is_same_entity)