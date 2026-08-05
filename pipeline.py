from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .calibration import CAIRCalibrator
from .config import resolve_path
from .data import ClinicalRecord
from .embeddings import build_embedder
from .extraction import EntityExtractor, RelationExtractor
from .generation import CandidateGenerator
from .graph import DualSourceGraphBuilder
from .llm_client import build_llm_client
from .ontology import ICDOntology
from .policy import enforce_external_api_policy
from .prompts import PromptRepository
from .quality import RelationConstraintSet
from .retrieval import MedGraphRetriever
from .schemas import PipelineResult
from .text import load_term_dictionary, normalize_text, sliding_windows


class LLMICRPipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.project_root = Path(config['_project_root'])
        self.prompts = PromptRepository(self.project_root)
        self.llm = build_llm_client(config['llm'])
        self.ontology = ICDOntology.from_csv(resolve_path(config, config['data']['ontology_nodes']))
        emb_cfg = config['embeddings']
        self.entity_embedder = build_embedder(
            emb_cfg.get('entity_backend', 'hashing'),
            emb_cfg.get('entity_model', ''),
            int(emb_cfg.get('entity_dimension', 768)),
            emb_cfg.get('device', 'cpu'),
        )
        self.calibration_embedder = build_embedder(
            emb_cfg.get('calibration_backend', 'hashing'),
            emb_cfg.get('calibration_model', ''),
            int(emb_cfg.get('calibration_dimension', 384)),
            emb_cfg.get('device', 'cpu'),
        )
        self.term_dictionary = load_term_dictionary(resolve_path(config, config['data'].get('term_dictionary', '')))
        self.relation_constraints = RelationConstraintSet.from_csv(
            resolve_path(config, config['data'].get('relation_constraints', ''))
        )
        self.entity_extractor = EntityExtractor(self.llm, self.prompts, float(config['thresholds']['entity']))
        self.relation_extractor = RelationExtractor(self.llm, self.prompts, float(config['thresholds']['relation']))
        self.graph_builder = DualSourceGraphBuilder(
            self.ontology, self.entity_embedder, float(config['thresholds'].get('link', 0.15))
        )

    def run_record(self, record: ClinicalRecord) -> PipelineResult:
        enforce_external_api_policy(record, self.config['llm'])
        normalized = normalize_text(record.text, self.term_dictionary)
        windows = sliding_windows(
            normalized,
            int(self.config['text'].get('max_window_tokens', 512)),
            float(self.config['text'].get('overlap_ratio', 0.2)),
            record.language,
        )
        # The paper limits records to four API calls. We concatenate normalized windows
        # for one entity call and one relation call rather than calling per window.
        llm_text = '\n\n[WINDOW]\n'.join(windows)
        max_chars = int(self.config['llm'].get('max_input_chars', 16000))
        llm_text = llm_text[:max_chars]
        entities = self.entity_extractor.extract(llm_text, record.language)
        relations = self.relation_extractor.extract(llm_text, record.language, entities)
        relations = self.relation_constraints.apply(entities, relations)

        external_edges = resolve_path(self.config, self.config['data']['ontology_edges'])
        graph, links = self.graph_builder.build(
            entities,
            relations,
            external_edges_path=external_edges,
            use_external_kg=bool(self.config.get('ablation', {}).get('use_external_kg', True)),
            topology_aware=bool(self.config.get('ablation', {}).get('dual_source_topology', True)),
        )
        retrieval_cfg = dict(self.config['retrieval'])
        if not self.config.get('ablation', {}).get('use_medgraphrag', True):
            retrieval_cfg.update({'max_hops': 1, 'use_llm_path_reranker': False, 'top_paths': 1})
        retriever = MedGraphRetriever(graph, self.entity_embedder, self.llm, self.prompts, retrieval_cfg)
        paths = retriever.retrieve(llm_text)
        generator = CandidateGenerator(
            self.llm, self.prompts, self.ontology, float(self.config['thresholds']['code'])
        )
        candidates = generator.generate(llm_text, record.language, paths)
        cair_cfg = dict(self.config['cair'])
        if not self.config.get('ablation', {}).get('use_cair', True):
            cair_cfg['enabled'] = False
        calibrator = CAIRCalibrator(self.ontology, self.calibration_embedder, cair_cfg, self.config['quality'])
        predictions = calibrator.calibrate(
            candidates, paths, float(self.config['thresholds']['code']), record.icd_version
        )
        predicted_labels = [prediction.code for prediction in predictions if prediction.accepted]
        manual_review_required = any(prediction.status == 'deferred' for prediction in predictions)
        return PipelineResult(
            record_id=record.record_id,
            gold_labels=record.labels,
            normalized_text=normalized,
            entities=entities,
            relations=relations,
            retrieved_paths=paths,
            predictions=predictions,
            predicted_labels=predicted_labels,
            manual_review_required=manual_review_required,
            metadata={
                'window_count': len(windows),
                'entity_links': [link.__dict__ if hasattr(link, '__dict__') else {
                    'entity_id': link.entity_id, 'code': link.code, 'similarity': link.similarity
                } for link in links],
                'config_hash': self.config_hash(),
            },
        )

    def config_hash(self) -> str:
        public_config = {k: v for k, v in self.config.items() if not k.startswith('_')}
        payload = json.dumps(public_config, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()
