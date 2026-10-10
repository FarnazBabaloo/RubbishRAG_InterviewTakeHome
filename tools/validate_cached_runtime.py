"""Exercise current production stages using captured API replies, never network.

This is a deterministic integration replay. Timing here is not production timing.
"""
import copy
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rubbish_rag import execution
from rubbish_rag.normalize_fa import normalize_fa
from server.evaluator import score_one,aggregate
import server._hidden_retriever as remote


def main():
    records=[json.loads(l) for l in (ROOT/'traces/trace.jsonl').read_text().splitlines() if l.strip()]
    cached={r['q']:r for r in records if r.get('event')=='pipeline' and r['hits']}
    replies={}
    for query,record in cached.items():
        norm=normalize_fa(query)
        sent=query if norm==query else query+' '+norm
        replies[sent]=[h for h in record['hits'] if h.get('source')!='local_corpus']
    counter=0
    def transport(query,topk,api_key):
        nonlocal counter
        counter += 1
        # Fail closed rather than retrieving an uncached query.
        if query not in replies:
            raise ValueError('No captured API response for requested query')
        return copy.deepcopy(replies[query])
    quota_before=(ROOT/'traces/quota.json').read_bytes()
    visible=json.loads((ROOT/'server/visible_bench.json').read_text())
    results=[]
    with tempfile.TemporaryDirectory(prefix='ics-runtime-') as tmp:
        with patch.object(remote,'remote_retrieve',transport),patch.object(execution,'TRACE',Path(tmp)/'trace.jsonl'):
            for item in visible:
                _,pred=execution.execute(item['query'])
                results.append({**score_one(pred,item),'id':item['id'],'prediction':pred})
            report={'mode':'cached_runtime_integration','agg':aggregate(results),
                    'mocked_transport_calls':counter,'real_retrieval_calls':0,'results':results}
            (ROOT/'traces/runtime_replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
            replay_entries=[]
            for line in (Path(tmp)/'trace.jsonl').read_text().splitlines():
                entry=json.loads(line)
                entry.update(event='offline_replay',mocked_transport_calls=entry['retrieval_calls'],retrieval_calls=0,
                             source_trace_id=cached[entry['q']]['trace_id'])
                replay_entries.append(entry)
            (ROOT/'traces/replay_trace.jsonl').write_text('\n'.join(json.dumps(e,ensure_ascii=False) for e in replay_entries)+'\n')
    assert (ROOT/'traces/quota.json').read_bytes()==quota_before
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


if __name__=='__main__':
    main()
