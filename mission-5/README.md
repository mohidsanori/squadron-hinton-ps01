# Mission 5 — Dual-Path GraphRAG, Hybrid Search & NotebookLM Grounding Lab

**Course:** Conversational AI: History, Applications, Future · **Squadron:** Hinton

This folder includes all six deliverables requested by the instructor, plus a quick-start README and dependencies. The HNSW and hybrid search are educational from-scratch implementations; the safety scores are transparent local proxies, not a verified NLI judgment. The NotebookLM manual experiment is **pending** and must be performed in a Google account before claiming empirical results.

## Files

- `01_HNSW_Vector_Index.py`: cosine normalization, sampled layers, greedy navigation, HNSW search and ANN-vs-flat benchmark
- `02_BM25_and_RRF_Combiner.py`: exact hyphenated-code tokenization, BM25, reciprocal rank fusion (`k=60`)
- `03_NotebookLM_RAG_Audit.py`: templates for five cross-document probes, three refusal probes, citation scoring, and audio audit
- `04_RAG_Provenance_and_Safety.py`: adversarial chunk quarantine, RAG triad proxies, JSONL provenance logs
- `ps05_solution.ipynb`: integrated executable demonstration and optional large benchmark
- `notebooklm_audit_report.md`: evidence checklist and candid evaluation report, to fill with observations

## Getting started

```bash
cd mission-5
python -m pip install -r requirements.txt
jupyter notebook ps05_solution.ipynb
```

Run the notebook cells top-to-bottom. The example corpus and vectors are **synthetic teaching examples**, not claims about an actual Noosphere archive. The notebook runs a small local benchmark; for the assignment-scale test run:

```bash
python 01_HNSW_Vector_Index.py --n 25000 --dim 768 --queries 20
```

This pure-Python index can take substantial time and RAM to construct at that scale; report **measured** latency/recall, rather than assuming O(log N) or sub-ms behavior. For smaller smoke tests: `python 01_HNSW_Vector_Index.py --n 300 --dim 64 --queries 3`.

## NotebookLM human-in-the-loop requirement

1. Download the primary Lewis et al. (2020) and Edge et al. (2024) papers and locate the real USSE Noosphere Technical Archive Extract from the instructor's repository.
2. Create a notebook at https://notebooklm.google.com and upload exactly these three sources.
3. Run `python 03_NotebookLM_RAG_Audit.py --create notebooklm_audit_evidence.json` and fill the JSON from **actual** NotebookLM output and badge clicks.
4. Execute all five cross-document and three negative probes. Save highlighted citation spans and flag unsupported claims.
5. Generate Audio Overview; transcribe a two-minute excerpt and rate factual fidelity, voice distinction and teaching quality.
6. Run `python 03_NotebookLM_RAG_Audit.py --score notebooklm_audit_evidence.json` and transfer real results into the report. Do not submit an unperformed audit as if it were completed.

**Repository placement:** You requested `mission-5/`; the instructor's PDF instead specifies `weeks/week-05/`. Check your team conventions and instructor expectations before final submission. **Signing-off:** submitters must supply their own four role sign-offs after verifying each station.
