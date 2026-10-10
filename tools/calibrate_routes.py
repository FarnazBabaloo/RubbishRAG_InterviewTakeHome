"""Fit on the frozen calibration split; audit once on separate exact queries.

Labels were authored by the same assistant: this is not independent validation.
Dirichlet(1,1,1) smoothing avoids certainty from tiny groups. No retrieval calls.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rubbish_rag.judge import (ACTIONS, HeuristicFeatureJudge, get_judge, confidence_group,
                              route_probabilities, compute_brier_score, compute_ece)

def measures(rows, judge):
    results=[]
    for row in rows:
        p=judge.pre_retrieval(row['query'])
        results.append({'id':row['id'],'expected':row['expected_action'],'prediction':p.to_dict(),
                        'correct':p.action==row['expected_action']})
    conf=[r['prediction']['confidence'] for r in results];correct=[int(r['correct']) for r in results]
    retrieve=[r['prediction']['retrieve_probability'] for r in results];truth=[int(r['expected']=='retrieve') for r in results]
    risk=[]
    for threshold in (0,.5,.75,.85,.9,.95):
        accepted=[r for r in results if r['prediction']['confidence']>=threshold]
        risk.append({'threshold':threshold,'coverage':len(accepted)/len(results),
                     'risk':1-sum(r['correct'] for r in accepted)/len(accepted) if accepted else None})
    return {'accuracy':sum(correct)/len(correct),'total':len(rows),'decision_brier':compute_brier_score(conf,correct),
            'decision_ece':compute_ece(conf,correct),'retrieval_brier':compute_brier_score(retrieve,truth),
            'retrieval_ece':compute_ece(retrieve,truth),'risk_coverage':risk,'results':results}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--fit',action='store_true');ap.add_argument('--audit',action='store_true');args=ap.parse_args()
    if args.fit==args.audit:ap.error('Choose --fit or --audit')
    path=ROOT/'evaluation/staff_calibration_v2.json';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==(path.with_suffix('.sha256').read_text().split()[0])
    quota=(ROOT/'traces/quota.json').read_bytes()
    data=json.loads(path.read_text());rows=[r for r in data['cases'] if r['split']==('calibration' if args.fit else 'audit')]
    target=ROOT/('traces/route_calibration_fit.json' if args.fit else 'traces/route_calibration_audit.json')
    if target.exists() or (args.fit and (ROOT/'traces/route_calibration_audit.json').exists()):ap.error('Preserve completed experiment; no overwrite/refit')
    raw=HeuristicFeatureJudge(temperature=.95)
    if args.fit:
        groups={}
        for row in rows:
            decision=raw.pre_retrieval(row['query'])
            key=confidence_group(row['query'],decision.action)
            counts=groups.setdefault(key,[0,0,0]);counts[ACTIONS.index(row['expected_action'])]+=1
        config={'method':'dirichlet_route_groups','prior_per_class':1,'fit_split':'calibration','fit_samples':len(rows),
                'dataset_sha256':digest,'counts':groups,'groups':{k:[(c+1)/(sum(v)+3) for c in v] for k,v in groups.items()},
                'fallback':[1/3,1/3,1/3]}
        (ROOT/'rubbish_rag/judge_calibration.json').write_text(json.dumps(config,indent=2))
    report={'mode':'calibration fit diagnostics' if args.fit else 'one-pass separate exact-query audit; same-author labels',
            'dataset_sha256':digest,'methodology':data['methodology'],'real_retrieval_calls':0,
            'raw':measures(rows,raw),'calibrated':measures(rows,get_judge()),
            'runtime_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'rubbish_rag').glob('*')) if p.suffix in ('.py','.json')}}
    assert quota==(ROOT/'traces/quota.json').read_bytes()
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:{a:b for a,b in report[k].items() if a not in ('results','risk_coverage')} for k in ('raw','calibrated')},indent=2))
if __name__=='__main__':main()
