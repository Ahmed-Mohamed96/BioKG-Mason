# src/biokg_agents/modules/triplets.py

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Entity:
    """
    Represents a biomedical entity in a triplet.
    """
    name: str
    type: Optional[str] = None        # semantic type, e.g. "PROTEIN", "GENE"
    entity_id: Optional[str] = None   # Neo4j entity_id if resolved
    metadata: Dict = field(default_factory=dict)


@dataclass
class Triplet:
    """
    Represents a (subject, predicate, object) fact.
    """
    subject: Entity
    predicate: str
    obj: Entity
    metadata: Dict = field(default_factory=dict)  # e.g. {"pmid":..., "paragraph":...}