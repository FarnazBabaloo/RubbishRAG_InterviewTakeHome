"""Replay the exposed 40-case development set without consuming quota.

Use original trace IDs, not latest-by-query selection. Missing recorded transport
replies fail closed, so replay cannot accidentally retrieve new evidence.
"""
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rubbish_rag import execution
from rubbish_rag.normalize_fa import normalize_fa
import server._hidden_retriever as remote


def main():
    before = (ROOT/'traces/quota.json').read_bytes()
    original = json.loads((ROOT/'traces/clarification_evaluation.json').read_text())
    records = {r['trace_id']: r for r in map(json.loads, (ROOT/'traces/trace.jsonl').read_text().splitlines()) if r.get('trace_id')}
    replies = {}
    for case in original['results']:
        entry = records[case['prediction']['trace_id']]
        if entry['retrieval_calls']:
            query = entry['q']; norm = normalize_fa(query)
            replies[query if query==norm else query+' '+norm] = entry['remote_hits']
    def transport(query, **kwargs):
        if query not in replies:
            raise AssertionError('No original API reply available for this query')
        return copy.deepcopy(replies[query])
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        trace = Path(tmp)/'replay.jsonl'
        with patch.object(remote, 'remote_retrieve', transport), patch.object(execution, 'TRACE', trace):
            for case in original['results']:
                _, pred = execution.execute(case['query'])
                results.append({'id':case['id'], 'query':case['query'], 'expected':case['expected'],
                                'prediction':pred, 'routing_correct':pred['status']==case['expected']})
        entries = []
        for line in trace.read_text().splitlines():
            entry = json.loads(line)
            entry.update(event='offline_clarification_regression', mocked_transport_calls=entry['retrieval_calls'], retrieval_calls=0)
            entries.append(entry)
        (ROOT/'traces/clarification_regression_trace.jsonl').write_text('\n'.join(json.dumps(e, ensure_ascii=False) for e in entries)+'\n')
    report = {'mode':'exposed_development_set_cached_regression', 'real_retrieval_calls':0,
              'dataset_sha256':original['dataset_sha256'],
              'runtime_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'rubbish_rag').glob('*.py'))},
              'correct':sum(r['routing_correct'] for r in results), 'total':len(results),
              'ambiguous_clarified':sum(r['expected']=='clarify' and r['routing_correct'] for r in results),
              'specific_answered':sum(r['expected']=='answer' and r['routing_correct'] for r in results),
              'results':results}
    (ROOT/'traces/clarification_regression.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    assert before==(ROOT/'traces/quota.json').read_bytes()
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','runtime_sha256')}, indent=2))


if __name__=='__main__':
    main()
