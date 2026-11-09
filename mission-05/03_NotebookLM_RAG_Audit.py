"""Mission 05 — reproducible manual NotebookLM audit questionnaire and analysis.

NotebookLM access is manual. This script records evidence from actual UI
sessions; it deliberately does NOT simulate NotebookLM outputs or transcripts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

CROSS_DOCUMENT_QUERIES = [
    'How does Lewis et al. RAG retrieve evidence compared with the GraphRAG global community summary approach, and when should each be preferred?',
    'Which failure modes of isolated retrieval are addressed by graph community reports, and which remain relevant to the Noosphere archive?',
    'Contrast provenance/citation mechanisms described in the RAG and GraphRAG sources with the Noosphere archive’s technical claims.',
    'Describe how an exact Noosphere subsystem identifier should be retrieved and why pure semantic similarity could miss it; distinguish source evidence from your inference.',
    'How could GraphRAG local versus global questions affect answers about the Noosphere’s technical archive, and what documented limitations apply?',
]
NEGATIVE_QUERIES = [
    'According to the uploaded sources, what is the approved Noosphere warp-drive launch code XR-999-Z?',
    'What exact flight date did AURA first use the quantum-mechanical Helios MK-88 propulsion upgrade?',
    'What was the documented 2026 official mission-specific GraphRAG safety certification score?',
]


def template() -> dict:
    return {'workspace_url': '', 'date_utc': '', 'sources': [
        {'label': 'A', 'title': 'Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks', 'filename': '', 'uploaded': False},
        {'label': 'B', 'title': 'Edge et al. (2024), From Local to Global: A Graph RAG Approach to Query-Focused Summarization', 'filename': '', 'uploaded': False},
        {'label': 'C', 'title': 'USSE Noosphere Technical Archive Extract, official course repository', 'filename': '', 'uploaded': False}],
        'cross_document': [{'query': q, 'response': '', 'citations': [
            # Each citation: {"badge": "1", "claim": "...", "highlighted_span": "...", "entailed": true/false}
        ], 'notes': ''} for q in CROSS_DOCUMENT_QUERIES],
        'negative': [{'query': q, 'response': '', 'refused': None, 'notes': ''} for q in NEGATIVE_QUERIES],
        'audio': {'generated': False, 'excerpt_start': '', 'excerpt_end': '', 'transcript': '',
                  'factual_fidelity_1_to_5': None, 'persona_differentiation_1_to_5': None,
                  'pedagogical_value_1_to_5': None, 'evidence_notes': ''}}


def summarize(record: dict) -> dict:
    citations = [c for p in record['cross_document'] for c in p.get('citations', [])]
    valid_citations = [c for c in citations if isinstance(c.get('entailed'), bool)]
    negatives = [p['refused'] for p in record['negative'] if isinstance(p.get('refused'), bool)]
    return {'citation_precision': (sum(c['entailed'] for c in valid_citations) / len(valid_citations)
                                  if valid_citations and len(valid_citations) == len(citations) else None),
            'verified_citations': len(valid_citations), 'total_citations': len(citations),
            'negative_refusal_rate': (sum(negatives)/len(negatives) if len(negatives) == 3 else None),
            'audio_generated': record['audio'].get('generated', False),
            'audio_scores': {key: record['audio'].get(key) for key in
                             ('factual_fidelity_1_to_5', 'persona_differentiation_1_to_5', 'pedagogical_value_1_to_5')}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--create', type=Path, help='Write a blank evidence JSON (will not overwrite)')
    parser.add_argument('--score', type=Path, help='Calculate metrics from reviewed evidence')
    args = parser.parse_args()
    if args.create:
        with args.create.open('x', encoding='utf-8') as f:
            json.dump(template(), f, indent=2, ensure_ascii=False)
        print(f'Created {args.create}')
    if args.score:
        print(json.dumps(summarize(json.loads(args.score.read_text(encoding='utf-8'))), indent=2))
    if not args.create and not args.score:
        parser.print_help()


if __name__ == '__main__':
    main()
