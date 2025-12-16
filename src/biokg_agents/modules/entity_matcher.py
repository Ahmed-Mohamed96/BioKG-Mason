from typing import Dict, List, Optional, Tuple
import math

from .base import EmbeddingsClient
from .triplets import Entity, Triplet
from .neo4j_client import Neo4jClient


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EntityMatcher:
    """
    Resolves entities using embeddings to avoid duplicates.
    Uses Neo4j as source of existing nodes and their embeddings.

    This class performs the embedding-based candidate selection.
    A separate LLM-based EntityMatchValidator will check whether
    the selected candidate is semantically correct in context.
    """

    def __init__(
        self,
        kg_client: Neo4jClient,
        embeddings_client: EmbeddingsClient,
        similarity_threshold: float = 0.82,
    ):
        self.kg = kg_client
        self.emb = embeddings_client
        self.similarity_threshold = similarity_threshold
        # Cache of existing entities: list of dicts
        self.existing_entities: List[Dict] = self.kg.get_all_entities()

    def _find_best_match(self, name_embedding: List[float]) -> Optional[Dict]:
        """
        Find the best matching entity (by cosine similarity) above threshold.
        Returns the entity dict or None.
        """
        best_entity = None
        best_score = 0.0
        for ent in self.existing_entities:
            ent_emb = ent.get("embedding") or []
            score = cosine_similarity(name_embedding, ent_emb)
            if score > best_score:
                best_score = score
                best_entity = ent
        if best_entity and best_score >= self.similarity_threshold:
            return best_entity
        return None

    def resolve_entity(
        self, entity: Entity
    ) -> Tuple[Entity, List[float], Optional[Dict]]:
        """
        Resolve a single entity using embeddings.

        Returns:
        - updated_entity (with entity_id if matched, else None)
        - embedding of entity.name
        - match_info: the matched existing entity dict (if any), or None

        """
        embedding = self.emb.embed(entity.name)
        match = self._find_best_match(embedding)
        if match:
            # Use existing node's entity_id (canonical node)
            entity.entity_id = match["entity_id"]
        else:
            entity.entity_id = None  # new entity will be created later
        return entity, embedding, match

    def resolve_triplet(
        self, triplet: Triplet
    ) -> Tuple[Triplet, Dict[str, List[float]], Dict[str, Optional[Dict]]]:
        """
        Resolve both subject and object entities in a triplet.

        Returns:
        - updated triplet
        - embeddings_map: {"subject": subj_embedding, "object": obj_embedding}
        - matches_map: {"subject": subj_match_dict_or_None,

                        "object": obj_match_dict_or_None}
        """
        subj, subj_emb, subj_match = self.resolve_entity(triplet.subject)
        obj, obj_emb, obj_match = self.resolve_entity(triplet.obj)
        triplet.subject = subj
        triplet.obj = obj
        embeddings_map = {"subject": subj_emb, "object": obj_emb}
        matches_map = {"subject": subj_match, "object": obj_match}
        return triplet, embeddings_map, matches_map

    def register_new_entity(self, entity_dict: Dict):
        """
        After creating a new node in Neo4j, register it in local cache.
        entity_dict should contain entity_id, name, labels, embedding.
        """
        self.existing_entities.append(entity_dict)