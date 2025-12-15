from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from neo4j import GraphDatabase


def _sanitize_label(label: str) -> str:
    """
    Allow only [A-Z0-9_]; fallback to GENERIC if invalid.
    """
    import re

    label = label.strip().upper().replace(" ", "_")
    if not re.match(r"^[A-Z0-9_]+$", label):
        return "GENERIC_ENTITY"
    return label


def _sanitize_rel_type(rel_type: str) -> str:
    """
    Allow only [A-Z0-9_]; fallback to RELATED_TO if invalid.
    """
    import re

    rel_type = rel_type.strip().upper().replace(" ", "_")
    if not re.match(r"^[A-Z0-9_]+$", rel_type):
        return "RELATED_TO"
    return rel_type


class Neo4jClient:
    """
    Thin wrapper around Neo4j driver for entity/relationship operations.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """
        Return all entities with their id, name, labels, and embeddings.
        """
        query = """
        MATCH (e)
        RETURN e.entity_id AS entity_id,
               e.name AS name,
               labels(e) AS labels,
               e.embedding AS embedding
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

    def upsert_entity(
        self, name: str, label: str, embedding: List[float], entity_id: Optional[str] = None
    ) -> str:
        """
        Create or update an entity node. Returns entity_id.
        If entity_id is None, a new node is created.
        """
        label = _sanitize_label(label)
        with self.driver.session() as session:
            if entity_id is None:
                new_id = str(uuid4())
                query = f"""
                CREATE (e:`{label}` {{
                    entity_id: $entity_id,
                    name: $name,
                    embedding: $embedding
                }})
                RETURN e.entity_id AS entity_id
                """
                params = {
                    "entity_id": new_id,
                    "name": name,
                    "embedding": embedding,
                }
            else:
                # Add label if missing and update name/embedding
                query = f"""
                MATCH (e {{entity_id: $entity_id}})
                SET e:`{label}`,
                    e.name = $name,
                    e.embedding = $embedding
                RETURN e.entity_id AS entity_id
                """
                params = {
                    "entity_id": entity_id,
                    "name": name,
                    "embedding": embedding,
                }
            record = session.run(query, **params).single()
            return record["entity_id"]

    def create_relationship(
        self,
        subj_entity_id: str,
        obj_entity_id: str,
        rel_type: str,
        pmid: str,
        paragraph: str,
    ) -> None:
        """
        Create a relationship between two entities with provenance.
        """
        rel_type = _sanitize_rel_type(rel_type)
        query = f"""
        MATCH (s {{entity_id: $subj_id}})
        MATCH (o {{entity_id: $obj_id}})
        MERGE (s)-[r:`{rel_type}` {{
            pmid: $pmid,
            paragraph: $paragraph
        }}]->(o)
        RETURN id(r) AS rel_id
        """
        params = {
            "subj_id": subj_entity_id,
            "obj_id": obj_entity_id,
            "pmid": pmid,
            "paragraph": paragraph,
        }
        with self.driver.session() as session:
            session.run(query, **params)

    def get_existing_labels(self) -> List[str]:
        query = "CALL db.labels() YIELD label RETURN label"
        with self.driver.session() as session:
            result = session.run(query)
            return [record["label"] for record in result]

    def get_existing_relationship_types(self) -> List[str]:
        query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        with self.driver.session() as session:
            result = session.run(query)
            return [record["relationshipType"] for record in result]