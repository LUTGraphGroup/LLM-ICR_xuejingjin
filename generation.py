from __future__ import annotations

import json

from .llm_client import BaseLLMClient
from .ontology import ICDOntology
from .prompts import PromptRepository
from .schemas import ICDCandidate, RetrievedPath


class CandidateGenerator:
    def __init__(self, llm: BaseLLMClient, prompts: PromptRepository, ontology: ICDOntology, threshold: float = 0.8):
        self.llm = llm
        self.prompts = prompts
        self.ontology = ontology
        self.threshold = threshold

    def generate(self, text: str, language: str, paths: list[RetrievedPath]) -> list[ICDCandidate]:
        allowed_codes = self._allowed_codes(paths)
        path_payload = [path.to_dict() for path in paths]
        code_payload = [self.ontology.node(code).to_dict() for code in allowed_codes]
        response = self.llm.complete_json(
            'code', self.prompts.system(),
            self.prompts.code(language).format(
                record_text=text,
                paths_json=json.dumps(path_payload, ensure_ascii=False),
                allowed_codes_json=json.dumps(code_payload, ensure_ascii=False),
            ),
        )
        candidates: list[ICDCandidate] = []
        seen: set[str] = set()
        for row in response.get('candidates', []):
            code = str(row.get('code') or '').strip()
            if code in seen or code not in allowed_codes or not self.ontology.contains(code):
                continue
            probability = min(1.0, max(0.0, float(row.get('probability', row.get('confidence', 0.0)))))
            if probability < self.threshold:
                continue
            seen.add(code)
            candidates.append(ICDCandidate(
                code=code,
                description=str(row.get('description') or self.ontology.node(code).title),
                probability=probability,
                evidence=str(row.get('evidence') or ''),
                reasoning_path=str(row.get('reasoning_path') or ''),
            ))
        candidates.sort(key=lambda item: (-item.probability, item.code))
        return candidates

    def _allowed_codes(self, paths: list[RetrievedPath]) -> list[str]:
        codes: set[str] = set()
        for path in paths:
            for node_id in path.node_ids:
                if self.ontology.contains(node_id):
                    codes.add(node_id)
                    codes.update(self.ontology.children.get(node_id, []))
                    parent = self.ontology.parent(node_id)
                    if parent:
                        codes.add(parent)
        if not codes:
            codes = set(self.ontology.nodes)
        return sorted(codes)
