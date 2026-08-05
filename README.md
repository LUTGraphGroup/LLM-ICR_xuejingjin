# LLM-ICR: Ontology-Anchored Inference Chain Reasoning

This package is an independent, executable reference implementation reconstructed
from the paper *Ontology-anchored Inference Chain Reasoning: A Neuro-symbolic
Approach to Cross-lingual Disease Diagnosis Coding*.

The paper describes four sequential stages:

1. multilingual semantic normalization and sliding-window chunking;
2. dual-source topological synthesis of patient-specific entities and the ICD ontology;
3. MedGraphRAG retrieval of multi-hop inference chains;
4. LLM candidate generation followed by Confidence-Aware Iterative Calibration (CAIR)
   and quality verification.
