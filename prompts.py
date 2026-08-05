from __future__ import annotations

from pathlib import Path


class PromptRepository:
    def __init__(self, project_root: str | Path):
        self.root = Path(project_root) / 'prompts'

    def read(self, name: str) -> str:
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(path)
        return path.read_text(encoding='utf-8')

    def system(self) -> str:
        return self.read('system_safety.txt')

    def entity(self, language: str) -> str:
        return self.read('entity_extraction_zh.txt' if language.startswith('zh') else 'entity_extraction_en.txt')

    def relation(self, language: str) -> str:
        return self.read('relation_extraction_zh.txt' if language.startswith('zh') else 'relation_extraction_en.txt')

    def path(self) -> str:
        return self.read('path_scoring.txt')

    def code(self, language: str) -> str:
        return self.read('code_generation_zh.txt' if language.startswith('zh') else 'code_generation_en.txt')
