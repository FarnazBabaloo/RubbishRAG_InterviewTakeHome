"""Paired cached-pool latency benchmark; includes trace writes, excludes imports.

The first pass includes lazy corpus/index initialization. Five following passes
measure warm execution. Predictions, decisions and selected evidence are saved
for equivalence comparison. Remote transport is mocked: zero quota is consumed.
"""
import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rubbish_rag import execution
from rubbish_rag.normalize_fa import normalize_fa
import server._hidden_retriever as remote


def equivalent(left, right):
    """Allow only floating-point summation noise across Python hash seeds."""
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
    if isinstance(left, dict):
        return left.keys()==right.keys() and all(equivalent(left[k],right[k]) for k in left)
    if isinstance(left, list):
        return len(left)==len(right) and all(equivalent(a,b) for a,b in zip(left,right))
    return left==right


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--label',required=True,choices=['before','after'])
    args=ap.parse_args()
    target=ROOT/f'traces/latency_{args.label}.json'
    if target.exists():ap.error('Preserve existing results; choose a new experiment before rerunning')
    quota=(ROOT/'traces/quota.json').read_bytes()
    data=json.loads((ROOT/'server/visible_bench.json').read_text())
    records=[json.loads(l) for l in (ROOT/'traces/trace.jsonl').read_text().splitlines() if l.strip()]
    pools={r['q']:r['remote_hits'] for r in records if r.get('event')=='pipeline' and r.get('remote_hits')}
    replies={(q if normalize_fa(q)==q else q+' '+normalize_fa(q)):pools[q] for q in [r['query'] for r in data]}
    def transport(query,**kwargs):return copy.deepcopy(replies[query])
    passes=[];behaviors=[]
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'trace.jsonl'
        with patch.object(remote,'remote_retrieve',transport),patch.object(execution,'TRACE',path):
            for repeat in range(6):
                durations=[]
                for row in data:
                    start=time.perf_counter()
                    execution.execute(row['query'])
                    durations.append((time.perf_counter()-start)*1000)
                passes.append(durations)
        for line in path.read_text().splitlines():
            r=json.loads(line)
            behaviors.append({k:r[k] for k in ('q','prediction','pre_decision','selected_hits','rounds','retrieval_calls')})
    warm=sorted(x for p in passes[1:] for x in p)
    report={'mode':'cached identical pools, CPU wall time including trace writes; not live API latency',
            'label':args.label,'real_retrieval_calls':0,'cold_pass_mean_ms':statistics.mean(passes[0]),
            'warm_mean_ms':statistics.mean(warm),'warm_p50_ms':statistics.median(warm),'warm_p95_ms':warm[94],
            'passes_ms':passes,'behaviors':behaviors,
            'pools_sha256':hashlib.sha256(json.dumps(replies,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    if args.label=='after':
        before=json.loads((ROOT/'traces/latency_before.json').read_text())
        assert before['pools_sha256']==report['pools_sha256']
        assert equivalent(before['behaviors'],behaviors), 'Behavior changed'
        report['behavior_equivalent']=True
        report['numeric_tolerance']=1e-12
        report['warm_reduction_pct']=100*(1-report['warm_mean_ms']/before['warm_mean_ms'])
    assert quota==(ROOT/'traces/quota.json').read_bytes()
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('passes_ms','behaviors')},indent=2))
if __name__=='__main__':main()
