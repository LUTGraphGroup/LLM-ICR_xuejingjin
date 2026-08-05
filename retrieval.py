from __future__ import annotations

import json
from dataclasses import replace
from typing import Iterable

import numpy as np

from .embeddings import Embedder
from .graph import HeterogeneousGraph
from .llm_client import BaseLLMClient
from .prompts import PromptRepository
from .schemas import RetrievedPath


class DenseIndex:
    def __init__(self, node_ids: list[str], texts: list[str], embedder: Embedder):
        self.node_ids = node_ids
        self.embedder = embedder
        self.matrix = embedder.encode(texts)
        self.faiss = None
        try:
            import faiss
            self.faiss = faiss.IndexFlatIP(self.matrix.shape[1])
            self.faiss.add(self.matrix.astype(np.float32))
        except ImportError:
            self.faiss = None

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        vector = self.embedder.encode([query]).astype(np.float32)
        top_k = min(max(1, top_k), len(self.node_ids))
        if self.faiss is not None:
            scores, indices = self.faiss.search(vector, top_k)
            return [(self.node_ids[int(index)], float(score)) for index, score in zip(indices[0], scores[0]) if index >= 0]
        scores = vector @ self.matrix.T
        order = np.argsort(-scores[0])[:top_k]
        return [(self.node_ids[int(index)], float(scores[0, index])) for index in order]


class MedGraphRetriever:
    def __init__(
        self,
        graph: HeterogeneousGraph,
        embedder: Embedder,
        llm: BaseLLMClient,
        prompts: PromptRepository,
        config: dict,
    ):
        self.graph = graph
        self.embedder = embedder
        self.llm = llm
        self.prompts = prompts
        self.config = config
        node_ids = sorted(graph.nodes)
        texts = [graph.nodes[node_id].text() for node_id in node_ids]
        self.index = DenseIndex(node_ids, texts, embedder)

    def retrieve(self, query: str) -> list[RetrievedPath]:
        entries = self.index.search(query, int(self.config.get('entry_top_k', 10)))
        dense_by_node = dict(entries)
        raw_paths: list[RetrievedPath] = []
        max_hops = int(self.config.get('max_hops', 3))
        for entry_id, dense_score in entries:
            raw_paths.extend(self._expand(entry_id, dense_score, max_hops))
        if not raw_paths:
            return []
        dedup: dict[tuple[str, ...], RetrievedPath] = {}
        for path in raw_paths:
            key = tuple(path.node_ids)
            if key not in dedup or path.relation_score > dedup[key].relation_score:
                dedup[key] = path
        paths = list(dedup.values())
        if self.config.get('use_llm_path_reranker', True):
            self._apply_llm_scores(query, paths)
        dense_weight = float(self.config.get('dense_weight', 0.45))
        relation_weight = float(self.config.get('relation_weight', 0.25))
        llm_weight = float(self.config.get('llm_weight', 0.30))
        for path in paths:
            path.combined_score = (
                dense_weight * path.dense_score
                + relation_weight * path.relation_score
                + llm_weight * path.llm_score
            )
        paths.sort(key=lambda item: (-item.combined_score, item.path_id))
        return paths[:int(self.config.get('top_paths', 5))]

    def _expand(self, start: str, dense_score: float, max_hops: int) -> list[RetrievedPath]:
        results: list[RetrievedPath] = []

        def dfs(current: str, node_ids: list[str], relations: list[str], confidences: list[float]):
            if relations:
                path_id = 'P_' + '_'.join(node_ids)
                text_parts: list[str] = []
                for idx, node_id in enumerate(node_ids):
                    text_parts.append(f'{node_id}:{self.graph.nodes[node_id].name}')
                    if idx < len(relations):
                        text_parts.append(relations[idx])
                results.append(RetrievedPath(
                    path_id=path_id,
                    node_ids=list(node_ids),
                    relations=list(relations),
                    path_text=' -> '.join(text_parts),
                    dense_score=max(0.0, min(1.0, (dense_score + 1.0) / 2.0)),
                    relation_score=float(np.mean(confidences)) if confidences else 0.0,
                ))
            if len(relations) >= max_hops:
                return
            for neighbor, edge in self.graph.neighbors(current, include_incoming=True):
                if neighbor in node_ids:
                    continue
                dfs(neighbor, node_ids + [neighbor], relations + [edge.relation], confidences + [edge.confidence])

        dfs(start, [start], [], [])
        return results

    def _apply_llm_scores(self, query: str, paths: list[RetrievedPath]) -> None:
        payload = [{'path_id': path.path_id, 'path': path.path_text} for path in paths]
        response = self.llm.complete_json(
            'path', self.prompts.system(),
            self.prompts.path().format(record_text=query, paths_json=json.dumps(payload, ensure_ascii=False)),
        )
        score_by_id = {
            str(item.get('path_id')): min(1.0, max(0.0, float(item.get('score', 0.0))))
            for item in response.get('path_scores', [])
        }
        for path in paths:
            path.llm_score = score_by_id.get(path.path_id, 0.0)
