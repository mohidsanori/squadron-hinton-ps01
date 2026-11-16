# Squadron Hinton — Mission 09: Adaptive RAG Router, CRAG / Self-RAG, TorchTPU Indexer & Route-Hijack Sentinel

**Course:** 14013 Conversational Artificial Intelligence: History, Application, Future — WiSe 26/27
**Mission:** ps09 — Large Magellanic Cloud, Tarantula Nebula (SEC-09-LMC), AURA Subsystem 09
**Instructor repo:** [Unisvet/couse-c-ai](https://github.com/Unisvet/couse-c-ai) (`mission-09/`)

Static "retrieve-then-read" RAG pays for retrieval on every query, trusts stale or poisoned documents, cannot handle multi-hop questions and can be driven into endless re-querying. This mission builds the four pieces that address those problems:

- **Adaptive router** — scores query complexity and picks no retrieval, one pass, or a decomposed multi-hop loop; probes `torch_tpu` → `cuda` → `cpu`.
- **CRAG and Self-RAG** — tri-band retrieval gating (0.72 / 0.38), sentence-strip filtering, relay reformulation, and beam ranking by `[ISREL]`, `[ISSUP]`, `[ISUSE]`.
- **bfloat16 ColBERT indexer** — masked, vectorised MaxSim with `torch.einsum("id,mjd->mij", ...)` over device-pinned token embeddings, plus an authority-and-recency contradiction resolver.
- **Route-hijack sentinel** — strips directive and critique-spoofing text from retrieved chunks and stops sponge loops with a hop ceiling, Jaccard similarity and a repeated-content check.

## Squad roster

| Station | Role | Officer | Deliverable |
|---|---|---|---|
| architect | Systems Architect | Mohid | [`adaptive_rag_tpu_router.py`](adaptive_rag_tpu_router.py) |
| protocol | Communications & Protocol Officer | Sharjeel | [`crag_selfrag_protocol.py`](crag_selfrag_protocol.py) |
| data | Data & Payload Specialist | Shayan | [`torchtpu_hbm_indexer.py`](torchtpu_hbm_indexer.py) |
| security | Security & Ethics Sentinel | Ahmed | [`route_hijack_sentinel.py`](route_hijack_sentinel.py) |
| all | Squadron documentation | all four | [`mission-09-doc.md`](mission-09-doc.md), [`mission-09-doc.pdf`](mission-09-doc.pdf) |

## Folder layout

```
mission-09/
├── README.md                     roster, results, sign-offs, how to run
├── adaptive_rag_tpu_router.py    Systems Architect
├── crag_selfrag_protocol.py      Protocol Officer
├── torchtpu_hbm_indexer.py       Data Specialist
├── route_hijack_sentinel.py      Security Sentinel
├── integration_demo.py           the four modules as one agentic loop; produces the report's measurements
├── mission-09-doc.md             squadron documentation (Part 5)
└── mission-09-doc.pdf            the same, for the Moodle drop-box
```

## How to run

Needs PyTorch (CPU build is enough) and, for nothing else, the standard library. Python 3.11+.

```bash
# from the repository root
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
cd mission-09
python adaptive_rag_tpu_router.py
python crag_selfrag_protocol.py
python torchtpu_hbm_indexer.py
python route_hijack_sentinel.py
python integration_demo.py        # about 20 s
```

Each script ends with `[PASS]` or `[OK]`; any failed check stops it with an `AssertionError`. On a Colab TPU v5e-1 with `torch_tpu` installed, `get_compute_device()` selects `torch.device("tpu")` automatically.

> **Windows with Anaconda:** importing torch next to Anaconda's MKL can abort with `OMP: Error #15`. Use a plain `venv` as above rather than setting `KMP_DUPLICATE_LIB_OK`.

## What each script checks

| Script | Checks (all pass) |
|---|---|
| `adaptive_rag_tpu_router.py` | the three instructor queries route A / B / C; the route is a pure function of the score and the two thresholds; ordered sub-queries; word-boundary operators; invalid configuration rejected; cost ordering A < B < C |
| `crag_selfrag_protocol.py` | exact band boundaries at 0.72 and 0.38; strip filtering keeps decimals and abbreviations; the higher-log-prob hallucination loses on `[ISSUP]` but wins when the reflection weights are zero; a number absent from the evidence blocks full support |
| `torchtpu_hbm_indexer.py` | einsum MaxSim equals an explicit loop; padding never changes or wins a score; query mask; bfloat16 within 0.0034 of float32 with the same top-1 in 40 of 40 corpora; bfloat16 uses half the memory; resolver weights (0.989 against 0.396), authority beating recency, low-confidence flag |
| `route_hijack_sentinel.py` | the poisoned chunk is cleaned and the legitimate sentences survive; 12 of 12 attack variants caught (including zero-width, full-width and line-break evasions) with 0 of 5 benign blocked; hop ceiling allows three hops and stops the fourth; Jaccard behaviour and its measured limit |
| `integration_demo.py` | routing and modelled savings; bfloat16 accuracy, memory and latency; CRAG on three real retrieval cases; stale catalog against resolver; route hijack, sponge and critique-spoofing attacks with and without the sentinel |

## Key results

Full tables and analysis are in [`mission-09-doc.pdf`](mission-09-doc.pdf). **All runs were on a CPU; the TPU path is implemented but was not exercised.**

| Experiment | Result |
|---|---|
| Routing on a 12-query mix (5 / 5 / 2) | modelled tokens 7,260; 75.4 % fewer than an always-agentic pipeline, 4.3 % more than naive RAG (which cannot answer the 2 multi-hop queries) |
| bfloat16 against float32 MaxSim | max error 0.0046 on scores up to 8; same top-1 in 59 of 60 corpora; index 16.4 MB against 32.8 MB |
| bfloat16 speed on this CPU | 4.5x to 8.6x **slower** than float32 (no bfloat16 matrix unit); TPU speed-up not tested |
| CRAG on real retrieval | good match CORRECT (1.0), partial match AMBIGUOUS (0.625), no local answer INCORRECT (0.0) |
| Stale catalog | similarity ranking returned the 1978 catalog first; the resolver chose the 2026 telemetry (0.989 against 0.396) |
| Route hijack | undefended: false answer from a poisoned chunk plus lying relay; with the sentinel: 0 relay hops and the correct answer |
| Sponge loop with padded queries | undefended 8 hops; ceiling and Jaccard only 3 hops (padding defeats Jaccard); full sentinel 2 hops via repeated content |

## Notes and limitations

- The starter files used different names and values from the briefing. We follow the briefing (`AdaptiveRAGRouter`, `decompose_multihop_query`, `evaluate_and_action`, `score_candidate_beam`, `alpha` 0.72 / 0.38, the `log P(y)` term in the Self-RAG score) and keep the starter names as aliases, so either set works.
- Starter bugs fixed: a mutable default argument for the query history in the sentinel; substring matching of operators such as `vs`; no padding mask or float32 normalisation in the indexer; the hop-ceiling meaning was ambiguous, so `current_hop` now counts completed hops (K_max = 3 allows three).
- The corpus, relay, vulnerable evaluator and candidate answers are scripted. Token embeddings are hashed vectors, so MaxSim behaves like a lexical matcher. The router's TTFT and token figures come from a stated cost model, not measurements.
- The CRAG evaluator and Self-RAG critic are lexical stand-ins for the trained models of the papers. The sentinel is signature based: a paraphrased instruction gets through, and an injection without closing punctuation removes its neighbouring sentence too.
- The starter's doc template quoted a MaxSim score of 7.9883 and the resolver weight 0.989; our run gives 7.9922 for the MaxSim and the same 0.989 for the weight.

## Station sign-offs

Each officer ticks their own box once they have read every deliverable and agree the squad's submission is ready.

- [x] **architect** — Systems Architect
- [x] **protocol** — Communications & Protocol Officer
- [x] **data** — Data & Payload Specialist
- [x] **security** — Security & Ethics Sentinel
