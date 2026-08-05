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

The implementation includes those stages, deterministic offline tests, restricted-data
schemas, preprocessing scripts, ablation utilities, parameter studies, transfer-study
helpers, and table/figure generation scripts.

## Critical reproducibility statement

This archive **does not contain** MIMIC-III, MIMIC-IV, MIMIC-IV-Note, UMLS content,
complete official ICD terminology, private TRD records, or paper-generated LLM caches.
It therefore cannot by itself regenerate the numerical values reported in the paper.
It can:

- run an end-to-end synthetic example without network access;
- preprocess legally obtained data locally;
- build an ontology and graph from user-supplied licensed resources;
- run the four-stage LLM-ICR pipeline with a mock, local OpenAI-compatible, or hosted
  backend;
- evaluate predictions and aggregate repeated runs;
- reproduce the *format* of reported tables and figures from result files.

See `REPRODUCIBILITY_NOTES.md` and `docs/KNOWN_PAPER_INCONSISTENCIES.md` before
claiming exact reproduction.

## Security and data-use warning

Do not send MIMIC or private TRD text to any third-party API unless your data-use
agreement, institutional approvals, and the service's retention and review settings
explicitly permit it. The default dummy configuration uses a deterministic local mock.
For records marked `restricted`, hosted API use is blocked unless both the configuration
and an explicit environment acknowledgement are set. The acknowledgement is only a
software guard; it is not legal or ethical authorization.

## Quick start: completely offline

```bash
python -m pip install -e . --no-build-isolation
pytest -q
python -m llm_icr.cli.run_pipeline --config configs/dummy.yaml
python -m llm_icr.cli.evaluate --config configs/dummy.yaml
```

Outputs are written under `results/runs/dummy/`.

## LLM backends

- `mock`: deterministic and offline; intended only for tests.
- `openai_compatible`: uses the OpenAI Python SDK against either an approved hosted
  service or a locally deployed OpenAI-compatible endpoint.

The paper reports a fixed GPT-4 Turbo snapshot for all LLM tasks in Section 3.5, but
also states in Section 4.4 that GPT-3.5-turbo was used for NER and relation extraction.
This package exposes task-specific model names so either interpretation can be tested.
Historical model snapshots may no longer be available.
