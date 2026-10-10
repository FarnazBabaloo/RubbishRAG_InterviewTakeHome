"""Evaluate a frozen, manually labelled routing set through the real pipeline.

Requires --live because non-gated queries consume retrieval quota. This script
never changes labels, runtime code, or benchmark metrics.json. Results are saved
per case before continuing; an existing report prevents accidental paid reruns.
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rubbish_rag.execution import execute


def wilson(successes, total):
    if not total:
        return None
    z = 1.96
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [center-half, center+half]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Run real retrieval, at most one call per dataset case')
    parser.add_argument('--dataset', default='evaluation/clarification_holdout.json')
    parser.add_argument('--output-prefix', default='clarification')
    args = parser.parse_args()
    if not args.live:
        parser.error('Pass --live to run real retrieval; read the saved report for existing results.')
    if not args.output_prefix.replace('_', '').isalnum():
        parser.error('Output prefix must contain only letters, digits or underscores')
    dataset = ROOT/args.dataset
    report_path = ROOT/'traces'/f'{args.output_prefix}_evaluation.json'
    cases_path = ROOT/'traces'/f'{args.output_prefix}_cases.jsonl'
    if report_path.exists() or cases_path.exists():
        parser.error('Evaluation results already exist; inspect them instead of rerunning.')
    frozen = dataset.with_suffix('.sha256').read_text().split()[0]
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == frozen
    cases = json.loads(dataset.read_text())['cases']
    assert len({c['query'] for c in cases}) == len(cases)
    quota_path = ROOT/'traces/quota.json'
    before = json.loads(quota_path.read_text()).get('demo-key', 0)
    assert before + len(cases) <= 1000, 'Insufficient remaining quota for the worst case'
    code_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((ROOT/'rubbish_rag').glob('*.py'))}
    results = []
    for case in cases:
        try:
            _, prediction = execute(case['query'], api_key='demo-key')
        except Exception as exc:
            prediction = {'status': 'error', 'error_type': type(exc).__name__, 'message': str(exc)}
        valid = (prediction.get('status') == 'clarify'
                 and isinstance(prediction.get('question'), str) and bool(prediction['question'].strip())
                 and isinstance(prediction.get('options'), list) and len(prediction['options']) >= 2
                 and all(isinstance(o, str) and o.strip() for o in prediction['options']))
        row = {**case, 'prediction': prediction, 'valid_clarification_contract': valid,
               'routing_correct': prediction.get('status') == case['expected']}
        with cases_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        results.append(row)
    tp = sum(r['expected']=='clarify' and r['prediction']['status']=='clarify' for r in results)
    fp = sum(r['expected']=='answer' and r['prediction']['status']=='clarify' for r in results)
    tn = sum(r['expected']=='answer' and r['prediction']['status']=='answer' for r in results)
    fn = sum(r['expected']=='clarify' and r['prediction']['status']=='answer' for r in results)
    errors = sum(r['prediction']['status']=='error' for r in results)
    ambiguous = sum(r['expected']=='clarify' for r in results)
    specific = len(results)-ambiguous
    valid_tp = sum(r['expected']=='clarify' and r['valid_clarification_contract'] for r in results)
    families = {}
    for family in sorted({r['family'] for r in results}):
        rows = [r for r in results if r['family']==family]
        families[family] = {'correct': sum(r['routing_correct'] for r in rows), 'total': len(rows),
                           'ambiguous_clarified': sum(r['expected']=='clarify' and r['valid_clarification_contract'] for r in rows)}
    after = json.loads(quota_path.read_text()).get('demo-key', 0)
    report = {'mode':'live_frozen_manual_routing_evaluation', 'created_at':datetime.now(timezone.utc).isoformat(),
              'dataset_sha256':frozen, 'runtime_sha256':code_hashes,
              'limitations':'Same assistant authored labels after seeing implementation; new exact queries, but familiar domain scenarios. Not hidden evaluation or independent annotation. Routing/contract correctness is not semantic correctness of questions or answers.',
              'total':len(results), 'ambiguous':ambiguous, 'specific':specific,
              'confusion':{'true_clarify':tp,'false_clarify':fp,'true_answer':tn,'missed_clarify':fn,'errors':errors},
              'routing_accuracy':(tp+tn)/len(results), 'clarification_recall':tp/ambiguous,
              'valid_contract_recall':valid_tp/ambiguous,
              'clarification_recall_wilson95':wilson(tp,ambiguous),
              'clarification_precision':tp/(tp+fp) if tp+fp else None,
              'clarification_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
              'false_clarification_rate':fp/specific,'specific_answer_rate':tn/specific,
              'by_family':families,'quota_before':before,'quota_after':after,'real_retrieval_calls':after-before,
              'results':results}
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == frozen
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in code_hashes.items())
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','runtime_sha256')}, indent=2))
    for r in results:
        if not r['routing_correct']:
            print(r['id'],r['query'],'expected',r['expected'],'got',r['prediction']['status'])


if __name__ == '__main__':
    main()
