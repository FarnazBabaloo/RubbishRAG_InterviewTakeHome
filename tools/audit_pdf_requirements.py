"""Recompute PDF-required metrics from existing traces, with zero API calls."""
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rubbish_rag.pipeline import corpus, fuse_hits


def main():
    quota = (ROOT / 'traces/quota.json').read_bytes()
    records = [json.loads(line) for line in (ROOT / 'traces/trace.jsonl').read_text().splitlines() if line.strip()]
    visible = json.loads((ROOT / 'server/visible_bench.json').read_text())
    latest = {r['q']: r for r in records if r.get('event') == 'pipeline'}
    baseline, final, raw_hit3, rrf_hit3, remote_hits = [], [], [], [], []
    for item in visible:
        line, first = next((i, r) for i, r in enumerate(records, 1) if r.get('q') == item['query'])
        current = latest[item['query']]
        gold = item['gold_doc_id']
        scores = sorted((h['fused'] for h in first['hits']), reverse=True)[:5]
        baseline.append({'query': item['query'], 'trace_line': line, 'timestamp': first['ts'],
                         'hit3': gold in [h['doc_id'] for h in first['hits'][:3]],
                         'spread': scores[0] - scores[-1]})
        cites = current['prediction']['cites'][:3]
        final.append({'id': item['id'], 'trace_id': current['trace_id'],
                      'hit3': gold in cites, 'hit1': bool(cites) and cites[0] == gold, 'cites': cites})
        remote = current['remote_hits']
        remote_hits.extend(remote)
        raw_hit3.append(gold in [h['doc_id'] for h in sorted(remote, key=lambda h: -h['fused'])[:3]])
        rrf_hit3.append(gold in [h['doc_id'] for h in fuse_hits(remote)[:3]])
    cites = [i for r in final for i in r['cites']]
    report = {'mode': 'cached_requirements_audit', 'real_retrieval_calls': 0,
              'baseline_raw_hit3': mean(b['hit3'] for b in baseline),
              'final_citation_hit3': mean(r['hit3'] for r in final),
              'final_citation_hit1': mean(r['hit1'] for r in final),
              'current_remote_raw_hit3': mean(raw_hit3), 'current_remote_rrf_hit3': mean(rrf_hit3),
              'poison_citations': sum(i not in corpus() for i in cites), 'total_top3_citations': len(cites),
              'distinct_client_queries': len(latest), 'baseline_traces': baseline, 'final_traces': final}
    (ROOT / 'traces/requirements_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter([len(h['text']) for h in remote_hits], [h['bm25'] for h in remote_hits], s=12, alpha=.5)
    axes[0].set(xlabel='Returned text length (characters)', ylabel='Raw remote BM25', title='Latest fixed remote pools (descriptive)')
    values = [mean(raw_hit3), mean(rrf_hit3)]
    axes[1].bar(['Raw fused', 'RRF only'], values)
    for i, value in enumerate(values):
        axes[1].text(i, value + .025, f'{value:.0%}', ha='center')
    axes[1].set(ylim=(0, 1), ylabel='Candidate Hit@3 (20 visible queries)', title='Same pools; no local search or reranking')
    fig.suptitle('Task B: observed lengths and fusion ablation (no causal length claim)')
    fig.tight_layout()
    fig.savefig(ROOT / 'plots/length_vs_bm25.png', dpi=140)
    plt.close(fig)
    assert (ROOT / 'traces/quota.json').read_bytes() == quota
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, indent=2))


if __name__ == '__main__':
    main()
