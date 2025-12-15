# biokg/agents/triplet_extractor.py

from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ..models import Entity, Triplet


class TripletItem(BaseModel):
    subject: str
    subject_type: Optional[str] = None
    predicate: str
    object: str
    object_type: Optional[str] = None


class TripletList(BaseModel):
    triplets: List[TripletItem] = Field(default_factory=list)


SYSTEM_PROMPT = """
You are an expert biomedical information extraction system.

Your task: Given a paragraph from a biomedical research article, extract
ALL biomedical triplets with fields:

- subject (entity name)
- subject_type (e.g., protein, gene, RNA, disease, drug, cell line, pathway, etc.)
- predicate (relationship phrase)
- object (entity name)
- object_type (same as subject_type)

Rules:
- Only include triplets that are biomedical and meaningful.
- Use concise but specific names for entities (e.g., "TNF-alpha", "p53", "breast cancer").
- Use natural language predicates that are causal/mechanistic (e.g., "inhibits", "activates", "increases").

You MUST output data that conforms exactly to the provided JSON schema.
"""


class TripletExtractor:
    """
    Agent that extracts biomedical triplets from paragraphs using a LangChain
    chat model (ChatOpenAI or ChatOllama) with structured output.
    """

    def __init__(self, chat_model: BaseChatModel):
        """
        :param chat_model: A LangChain BaseChatModel (ChatOpenAI or ChatOllama).
        """
        self.chat_model = chat_model

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "Paragraph:\n{paragraph}"),
            ]
        )

        # Wrap the chat model to enforce TripletList schema
        self.chain = self.prompt | self.chat_model.with_structured_output(TripletList)

    def extract_triplets(self, paragraph: str) -> List[Triplet]:
        """
        Extract structured triplets from a paragraph.

        Uses LangChain structured output to guarantee JSON schema compliance.
        """
        result: TripletList = self.chain.invoke({"paragraph": paragraph})

        triplets: List[Triplet] = []
        for item in result.triplets:
            subj = Entity(name=item.subject, type=item.subject_type)
            obj = Entity(name=item.object, type=item.object_type)
            triplets.append(Triplet(subject=subj, predicate=item.predicate, obj=obj))

        return triplets