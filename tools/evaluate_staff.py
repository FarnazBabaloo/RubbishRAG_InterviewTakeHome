"""Fit routing temperature on train; report dev, then frozen held-out test.

No retrieval calls. The public test split is read only for --test, after the
calibration artifact is fixed. Decision/evidence probabilities are distinguished.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rubbish_rag.judge import (HeuristicFeatureJudge, get_judge, evaluate_judge, expected_route,
                              route_probabilities, ACTIONS, compute_brier_score, compute_ece)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',action='store_true')
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--regression',action='store_true',help='Current runtime on exposed public splits; not a new held-out test')
    args=parser.parse_args()
    if sum((args.fit,args.test,args.regression))!=1: parser.error('Choose exactly one of --fit, --test or --regression')
    data_path=ROOT/'data/judge_dataset.json'
    data=json.loads(data_path.read_text())
    config=ROOT/'rubbish_rag/judge_calibration.json'
    if args.regression:
        report={'mode':'post-error-analysis regression; public test is exposed, not held out',
                'calibration_note':'Current delivered calibration; see route_calibration_audit.json for the separate-query experiment. These public splits are exposed regression data.',
                'real_retrieval_calls':0,'splits':{s:evaluate_judge(split=s) for s in ('train','dev','test')},
                'uncalibrated_test':evaluate_judge(judge=HeuristicFeatureJudge(),split='test')['calibration'],
                'runtime_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'rubbish_rag').glob('*')) if p.suffix in ('.py','.json')}}
        (ROOT/'traces/staff_regression.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps({s:{k:v for k,v in r.items() if k!='results'} for s,r in report['splits'].items()},indent=2))
        return
    if args.fit:
        if (ROOT/'traces/staff_test.json').exists(): parser.error('Held-out report exists: no refitting on this experiment')
        train=[r for r in data if r['split']=='train']
        candidates=[]
        for temperature in [0.5+i*.05 for i in range(51)]:
            judge=HeuristicFeatureJudge(temperature=temperature)
            loss=sum(-math.log(max(1e-12,route_probabilities(judge.pre_retrieval(r['query']))[ACTIONS.index(expected_route(r))])) for r in train)/len(train)
            candidates.append((loss,temperature))
        loss,temperature=min(candidates)
        config.write_text(json.dumps({'method':'scalar temperature; train multiclass negative log likelihood',
                                      'temperature':temperature,'fit_split':'train','fit_samples':len(train),
                                      'fit_nll':loss,'dataset_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest()},indent=2))
        report={'config':json.loads(config.read_text()),'train':evaluate_judge(split='train'),'dev':evaluate_judge(split='dev')}
        (ROOT/'traces/staff_development.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps({'config':report['config'],'train_accuracy':report['train']['routing_accuracy'],'dev_accuracy':report['dev']['routing_accuracy']},indent=2))
        return
    target=ROOT/'traces/staff_test.json'
    if target.exists():parser.error('Test report already exists; preserve it rather than re-evaluate')
    assert config.exists()
    judge=get_judge()
    evaluated=evaluate_judge(judge=judge,split='test')
    uncalibrated=evaluate_judge(judge=HeuristicFeatureJudge(),split='test')
    # Load only the supplied editable baseline, never sealed server files.
    spec=importlib.util.spec_from_file_location('rubbish_rag.staff_baseline',ROOT/'tools/staff_baseline.py')
    baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
    original=baseline.HeuristicFeatureJudge()
    heldout=[r for r in data if r['split']=='test']
    baseline_actions=[original.pre_retrieval(r['query']).action for r in heldout]
    original_accuracy=sum(a==expected_route(r) for a,r in zip(baseline_actions,heldout))/len(heldout)
    elapsed=[]
    for row in heldout:
        start=time.perf_counter();judge.pre_retrieval(row['query']);elapsed.append((time.perf_counter()-start)*1000)
    confidences=[r['prediction']['confidence'] for r in evaluated['results']]
    correct=[r['correct'] for r in evaluated['results']]
    risk=[]
    for threshold in (0,.5,.75,.9,.95,.99):
        accepted=[c for p,c in zip(confidences,correct) if p>=threshold]
        risk.append({'threshold':threshold,'coverage':len(accepted)/len(correct),'risk':1-sum(accepted)/len(accepted) if accepted else None})
    report={'mode':'heldout_public_judge_test','test_previously_used_for_tuning':False,
            'label_semantics':'clarify overrides expected_retrieval=RETRIEVE; retrieval probability means call now, not eventual knowledge need',
            'metrics':evaluated,'uncalibrated_calibration':uncalibrated['calibration'],
            'provided_baseline_three_way_accuracy':original_accuracy,
            'mean_pre_latency_ms':sum(elapsed)/len(elapsed),'risk_coverage':risk,
            'runtime_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'rubbish_rag').glob('*')) if p.suffix in ('.py','.json')},
            'real_retrieval_calls':0}
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','runtime_sha256')},indent=2))
    print(json.dumps({k:v for k,v in evaluated.items() if k!='results'},indent=2))


if __name__=='__main__':main()
