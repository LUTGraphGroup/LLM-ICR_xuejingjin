from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path

from .schemas import ICDNode


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


class ICDOntology:
    def __init__(self, nodes: list[ICDNode]):
        if not nodes:
            raise ValueError('Ontology must contain at least one node.')
        self.nodes = {node.code: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError('Ontology contains duplicate codes.')
        self.children: dict[str, list[str]] = defaultdict(list)
        for node in nodes:
            if node.parent_code:
                if node.parent_code not in self.nodes:
                    raise ValueError(f'Unknown parent {node.parent_code!r} for code {node.code!r}.')
                self.children[node.parent_code].append(node.code)
        for key in self.children:
            self.children[key].sort()
        self.roots = sorted(node.code for node in nodes if not node.parent_code)

    @classmethod
    def from_csv(cls, path: str | Path) -> 'ICDOntology':
        nodes: list[ICDNode] = []
        with Path(path).open('r', encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                synonyms = [x.strip() for x in (row.get('synonyms') or '').split('|') if x.strip()]
                nodes.append(ICDNode(
                    code=(row.get('code') or '').strip(),
                    title=(row.get('title') or '').strip(),
                    parent_code=(row.get('parent_code') or '').strip() or None,
                    level=int(row.get('level') or 0),
                    synonyms=synonyms,
                    icd_version=(row.get('icd_version') or '').strip(),
                    language=(row.get('language') or 'en').strip(),
                    is_billable=_as_bool(row.get('is_billable', True)),
                ))
        return cls(nodes)

    def contains(self, code: str) -> bool:
        return code in self.nodes

    def node(self, code: str) -> ICDNode:
        return self.nodes[code]

    def parent(self, code: str) -> str | None:
        return self.nodes[code].parent_code

    def ancestors(self, code: str) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        current = self.parent(code)
        while current:
            if current in seen:
                raise ValueError(f'Cycle detected at {current}.')
            seen.add(current)
            result.append(current)
            current = self.parent(current)
        return result

    def descendants(self, code: str) -> list[str]:
        result: list[str] = []
        queue = deque(self.children.get(code, []))
        while queue:
            child = queue.popleft()
            result.append(child)
            queue.extend(self.children.get(child, []))
        return result

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return ancestor in self.ancestors(descendant)

    def is_leaf(self, code: str) -> bool:
        return not self.children.get(code)

    def valid_version(self, code: str, version: str) -> bool:
        node_version = self.nodes[code].icd_version
        return not version or not node_version or node_version == version

    def hierarchy_path(self, code: str) -> list[str]:
        return list(reversed(self.ancestors(code))) + [code]

    def search_texts(self) -> tuple[list[str], list[str]]:
        codes = sorted(self.nodes)
        return codes, [self.nodes[code].searchable_text() for code in codes]
