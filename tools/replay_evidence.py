"""Evaluate cached candidates and ablations without remote calls.

Local candidates can be recomputed against the public CSV. Remote pools remain
fixed; replay does not demonstrate fresh retrieval or hidden-test generalization.
"""
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rubbish_rag.pipeline import rerank, resolve, corpus
from rubbish_rag.local_search import local_candidates
from server.evaluator import score_one, aggregate


def main():
    entries = [json.loads(line) for line in (ROOT/'traces/trace.jsonl').read_text().splitlines() if line.strip()]
    captured = {e['q']:e for e in entries if e.get('event')=='pipeline' and e['hits']}
    visible = json.loads((ROOT/'server/visible_bench.json').read_text())
    results, no_rrf, no_local = [], [], []
    merged_recall, remote_recall, references, remote_pools, ranked_pools = [], [], [], [], []
    for item in visible:
        entry = captured.get(item['query'])
        if entry is None:
            continue
        remote = [h for h in entry['hits'] if h.get('source') != 'local_corpus']
        merged = remote + local_candidates(entry['q'])
        ranked = rerank(entry['q'],merged)
        pred = resolve(entry['q'],ranked[:5])
        results.append({**score_one(pred,item),'id':item['id'],'trace_id':entry['trace_id'],'prediction':pred})
        plain = [{k:v for k,v in h.items() if k not in ('bm25','dense','rrf_score','local_bm25')} for h in merged]
        no_rrf.append(score_one(resolve(entry['q'],rerank(entry['q'],plain)[:5]),item))
        no_local.append(score_one(resolve(entry['q'],rerank(entry['q'],remote)[:5]),item))
        merged_recall.append(item['gold_doc_id'] in [h['doc_id'] for h in merged])
        remote_recall.append(item['gold_doc_id'] in [h['doc_id'] for h in remote])
        references.append(entry['trace_id']); remote_pools.append(remote); ranked_pools.append(ranked[:5])
    unknown = lambda pools: sum(h['doc_id'] not in corpus() for pool in pools for h in pool)/max(1,sum(map(len,pools)))
    redundancy = lambda hits: 1-len({h['doc_id'] for h in hits})/max(1,len(hits))
    report = {'mode':'offline_replay','limitation':'Public-corpus supplement is recomputed, remote pools are fixed; not hidden evaluation.',
              'agg':aggregate(results),'without_rrf':aggregate(no_rrf),'without_local_search':aggregate(no_local),
              'merged_pool_recall':mean(merged_recall),'remote_pool_recall':mean(remote_recall),
              'unknown_remote_hit_rate':unknown(remote_pools),'unknown_ranked_hit_rate':unknown(ranked_pools),
              'mean_remote_top5_redundancy':mean(redundancy(sorted(p,key=lambda h:h.get('fused',0),reverse=True)[:5]) for p in remote_pools),
              'mean_ranked_top5_redundancy':mean(redundancy(p) for p in ranked_pools),
              'trace_ids':references,'results':results}
    (ROOT/'traces/replay_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','trace_ids')},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    ax.bar(['Full replay','No RRF','No local search'],[report['agg']['accuracy'],report['without_rrf']['accuracy'],report['without_local_search']['accuracy']])
    ax.set_ylim(0,1.1); ax.set_ylabel('Visible Hit@3 accuracy, cached remote pools')
    fig.tight_layout(); fig.savefig(ROOT/'plots/rrf_ablation.png'); plt.close(fig)


if __name__ == '__main__':
    main()
