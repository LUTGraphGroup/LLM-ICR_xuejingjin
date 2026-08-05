from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .embeddings import Embedder
from .ontology import ICDOntology
from .schemas import Entity, GraphEdge, GraphNode, Relation


class HeterogeneousGraph:
    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self.outgoing: dict[str, list[GraphEdge]] = defaultdict(list)
        self.incoming: dict[str, list[GraphEdge]] = defaultdict(list)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError(f'Edge references unknown nodes: {edge.source} -> {edge.target}')
        self.edges.append(edge)
        self.outgoing[edge.source].append(edge)
        self.incoming[edge.target].append(edge)

    def neighbors(self, node_id: str, include_incoming: bool = True) -> list[tuple[str, GraphEdge]]:
        result = [(edge.target, edge) for edge in self.outgoing.get(node_id, [])]
        if include_incoming:
            result.extend((edge.source, edge) for edge in self.incoming.get(node_id, []))
        return result


@dataclass(slots=True)
class EntityLink:
    entity_id: str
    code: str
    similarity: float


class DualSourceGraphBuilder:
    def __init__(self, ontology: ICDOntology, embedder: Embedder, link_threshold: float = 0.15):
        self.ontology = ontology
        self.embedder = embedder
        self.link_threshold = float(link_threshold)
        self._codes, self._texts = ontology.search_texts()
        self._ontology_matrix = embedder.encode(self._texts)

    def link_entities(self, entities: list[Entity], top_k: int = 3) -> list[EntityLink]:
        if not entities:
            return []
        queries = [f'{e.normalized_name} {e.description}'.strip() for e in entities]
        query_matrix = self.embedder.encode(queries)
        similarities = query_matrix @ self._ontology_matrix.T
        links: list[EntityLink] = []
        for row, entity in enumerate(entities):
            order = np.argsort(-similarities[row])[:top_k]
            for index in order:
                score = float(similarities[row, index])
                if score >= self.link_threshold:
                    links.append(EntityLink(entity.entity_id, self._codes[int(index)], score))
        return links

    def build(
        self,
        entities: list[Entity],
        relations: list[Relation],
        external_edges_path: str | Path | None = None,
        use_external_kg: bool = True,
        topology_aware: bool = True,
    ) -> tuple[HeterogeneousGraph, list[EntityLink]]:
        graph = HeterogeneousGraph()
        for code, node in self.ontology.nodes.items():
            graph.add_node(GraphNode(
                node_id=code, node_type='ICD', name=node.title,
                description=' | '.join(node.synonyms), confidence=1.0,
                metadata={'code': code, 'level': node.level, 'is_billable': node.is_billable, 'icd_version': node.icd_version},
            ))
        for entity in entities:
            graph.add_node(GraphNode(
                node_id=entity.entity_id, node_type=entity.entity_type,
                name=entity.normalized_name, description=entity.description,
                confidence=entity.confidence, semantic_type=entity.semantic_type,
                metadata={'evidence': entity.evidence},
            ))
        for node in self.ontology.nodes.values():
            if node.parent_code:
                graph.add_edge(GraphEdge(node.parent_code, node.code, 'PART_OF', 1.0, 'official-ontology'))
        for relation in relations:
            graph.add_edge(GraphEdge(
                relation.source_id, relation.target_id, relation.relation_type,
                relation.confidence, relation.source, relation.evidence,
            ))
        if use_external_kg and external_edges_path and Path(external_edges_path).exists():
            self._add_external_edges(graph, external_edges_path)
        links = self.link_entities(entities)
        for link in links:
            relation = 'ANCHORED_TO' if topology_aware else 'SIMILAR_TO'
            graph.add_edge(GraphEdge(link.entity_id, link.code, relation, link.similarity, 'dense-link'))
        return graph, links

    @staticmethod
    def _add_external_edges(graph: HeterogeneousGraph, path: str | Path) -> None:
        with Path(path).open('r', encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                source = (row.get('source') or '').strip()
                target = (row.get('target') or '').strip()
                relation = (row.get('relation') or 'ASSOCIATED_WITH').strip().upper()
                if relation == 'PART_OF':
                    continue
                if source in graph.nodes and target in graph.nodes:
                    graph.add_edge(GraphEdge(
                        source=source,
                        target=target,
                        relation=relation,
                        confidence=float(row.get('confidence') or 1.0),
                        source_type=(row.get('source_type') or 'external-kg').strip(),
                        evidence=(row.get('evidence') or '').strip(),
                    ))
