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

    def __init__(self, uri: str, user: str, password: str, database: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), database=database)

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
        self,
        name: str,
        label: str,
        embedding: List[float],
        entity_id: Optional[str] = None,
    ) -> str:
        """
        Upsert an entity node using a priority:
        1) Reuse an existing node with the same label + name.
        2) If none found, and entity_id is provided, reuse node with that entity_id.
        3) Otherwise, create a new node (with entity_id = given or new UUID).

        Returns the entity_id of the node used/created.
        """
        label = _sanitize_label(label)

        with self.driver.session() as session:
            # 1) Try to match by (label, name)
            query_match_name = f"""
            MATCH (e:`{label}` {{name: $name}})
            RETURN e.entity_id AS entity_id
            LIMIT 1
            """
            rec = session.run(query_match_name, name=name).single()
            if rec:
                existing_id = rec["entity_id"]
                # Update this existing node
                query_update = f"""
                MATCH (e {{entity_id: $entity_id}})
                SET e:`{label}`,
                    e.name = $name,
                    e.embedding = $embedding
                RETURN e.entity_id AS entity_id
                """
                rec2 = session.run(
                    query_update,
                    entity_id=existing_id,
                    name=name,
                    embedding=embedding,
                ).single()
                return rec2["entity_id"]

            # 2) Fallback: if entity_id is provided, try to match by entity_id
            if entity_id is not None:
                query_match_id = f"""
                MATCH (e {{entity_id: $entity_id}})
                SET e:`{label}`,
                    e.name = $name,
                    e.embedding = $embedding
                RETURN e.entity_id AS entity_id
                """
                rec = session.run(
                    query_match_id,
                    entity_id=entity_id,
                    name=name,
                    embedding=embedding,
                ).single()
                if rec:
                    return rec["entity_id"]

            # 3) No existing node; create a new one
            new_id = entity_id or str(uuid4())
            query_create = f"""
            CREATE (e:`{label}` {{
                entity_id: $entity_id,
                name: $name,
                embedding: $embedding
            }})
            RETURN e.entity_id AS entity_id
            """
            rec = session.run(
                query_create,
                entity_id=new_id,
                name=name,
                embedding=embedding,
            ).single()
            return rec["entity_id"]

    def create_relationship(
        self,
        subj_entity_id: str,
        obj_entity_id: str,
        rel_type: str,
        pmid: str,
        source_text: str,
        source_kind: str = "paragraph",   # "paragraph" or "table"
        page: Optional[int] = None,
        table_index: Optional[int] = None,
    ) -> None:
        """
        Create or reuse a relationship between two entities with provenance.

        Note:
        - We MERGE only on the pattern (s)-[r:TYPE]->(o) without properties.
        - Then we SET properties, which can safely include nulls (they get removed).

        This avoids Neo4j's error about MERGE with null property values.
        """
        rel_type = _sanitize_rel_type(rel_type)

        query = f"""
        MATCH (s {{entity_id: $subj_id}})
        MATCH (o {{entity_id: $obj_id}})
        MERGE (s)-[r:`{rel_type}`]->(o)
        SET r.pmid = $pmid,
            r.source_text = $source_text,
            r.source_kind = $source_kind,
            r.page = $page,
            r.table_index = $table_index
        RETURN elementId(r) AS rel_id
        """

        params = {
            "subj_id": subj_entity_id,
            "obj_id": obj_entity_id,
            "pmid": pmid,
            "source_text": source_text,
            "source_kind": source_kind,
            "page": page,
            "table_index": table_index,
        }

        with self.driver.session() as session:
            result = session.run(query, **params)
            record = result.single()
            rel_id = record["rel_id"] if record is not None else None
            print(
                f"[Neo4j] create_relationship: subj={subj_entity_id}, "
                f"obj={obj_entity_id}, rel_type={rel_type}, rel_id={rel_id}"
            )

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