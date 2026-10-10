"""Compare context policies on fixed captured pools and typed evidence fixtures.

Token counts use normalize_fa.tokens, not a model tokenizer; no generation model
is called. Fixed-k=10 does not claim unseen remote top-20 results were captured.
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
from rubbish_rag.pipeline import corpus
from rubbish_rag.judge import get_judge,compute_brier_score,compute_ece
import server._hidden_retriever as remote
from server.evaluator import score_one,aggregate


def main():
    before=(ROOT/'traces/quota.json').read_bytes()
    records=[json.loads(l) for l in (ROOT/'traces/trace.jsonl').read_text().splitlines() if l.strip()]
    captured={r['q']:r for r in records if r.get('event')=='pipeline' and r['hits'] and r.get('pipeline_version')=='public-corpus-supplement'}
    visible=json.loads((ROOT/'server/visible_bench.json').read_text())
    replies={}
    for item in visible:
        query=item['query'];norm=normalize_fa(query)
        replies[query if norm==query else query+' '+norm]=captured[query]['remote_hits']
    def transport(query,**kwargs):
        if query not in replies:raise AssertionError('Missing cached pool')
        return copy.deepcopy(replies[query])
    policies={};traces=[]
    with tempfile.TemporaryDirectory() as tmp:
        for name,k,adaptive in [('fixed5',5,False),('fixed10',10,False),('adaptive',8,True)]:
            path=Path(tmp)/(name+'.jsonl');results=[]
            with patch.object(remote,'remote_retrieve',transport),patch.object(execution,'TRACE',path):
                for item in visible:
                    _,pred=execution.execute(item['query'],topk=k,adaptive=adaptive)
                    results.append({**score_one(pred,item),'id':item['id'],'trace_id':pred['trace_id']})
            entries=[json.loads(l) for l in path.read_text().splitlines()]
            for entry in entries:
                entry.update(event='offline_staff_policy',policy=name,source_trace_id=captured[entry['q']]['trace_id'],mocked_transport_calls=entry['retrieval_calls'],retrieval_calls=0)
            traces.extend(entries)
            policies[name]={'agg':aggregate(results),'context_tokens':sum(e['context_token_estimate'] for e in entries),
                            'mean_selected_documents':sum(len(e['selected_hits']) for e in entries)/len(entries),
                            'results':results}
    savings=100*(1-policies['adaptive']['context_tokens']/policies['fixed5']['context_tokens'])
    (ROOT/'traces/staff_context.json').write_text(json.dumps({'mode':'fixed-captured-pool comparison','token_unit':'normalize_fa.tokens on selected excerpts; no LLM tokenizer',
                                                            'real_retrieval_calls':0,'policies':policies,'token_reduction_vs_fixed5_pct':savings},ensure_ascii=False,indent=2))
    (ROOT/'traces/staff_replay_trace.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in traces)+'\n')
    def hit(i):return {'doc_id':i,'text':corpus()[i]['BriefAnswer'],'fused':.99}
    fixtures=[
        ('valid','رتبه C1 چیست؟',[hit(106)],{},'sufficient','answer'),
        ('empty','رتبه C1 چیست؟',[],{},'insufficient','retrieve_more'),
        ('forged_id','رتبه C1 چیست؟',[{'doc_id':106,'text':'وام تضمینی برای همه','fused':999}],{},'insufficient','retrieve_more'),
        ('unknown','رتبه C1 چیست؟',[{'doc_id':99999,'text':'رتبه C1 '*30,'fused':999}],{},'insufficient','retrieve_more'),
        ('conflicting_rank','رتبه C1 چیست؟',[hit(108)],{},'conflicting','abstain'),
        ('missing_facet','تفاوت C1 و C3 چیست؟',[hit(106)],{},'insufficient','retrieve_more'),
        ('both_facets','تفاوت C1 و C3 چیست؟',[hit(106),hit(108)],{},'sufficient','answer'),
        ('round_limit','رتبه C1 چیست؟',[],{'round':2,'max_rounds':2},'insufficient','abstain'),
        ('scam','وام فوری بدون ضامن و بدون اعتبارسنجی',[hit(106)],{},'irrelevant','abstain'),
    ]
    cases=[]
    for name,q,h,state,suff,action in fixtures:
        decision=get_judge().post_retrieval(q,h,state)
        cases.append({'fixture_id':'staff_evidence_'+name,'query':q,'hits':h,'state':state,'expected_sufficiency':suff,
                      'expected_action':action,'decision':decision.to_dict(),
                      'correct':decision.sufficiency==suff and decision.next_action==action})
    probs=[c['decision']['sufficiency_probability'] for c in cases];ys=[int(c['expected_sufficiency']=='sufficient') for c in cases]
    evidence={'mode':'controlled public-corpus and synthetic evidence fixtures, not semantic held-out labels',
              'total':len(cases),'correct':sum(c['correct'] for c in cases),'sufficiency_brier':compute_brier_score(probs,ys),
              'sufficiency_ece':compute_ece(probs,ys),'cases':cases,'real_retrieval_calls':0}
    (ROOT/'traces/staff_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    assert (ROOT/'traces/quota.json').read_bytes()==before
    print(json.dumps({'policies':{k:{a:b for a,b in v.items() if a!='results'} for k,v in policies.items()},'token_reduction_pct':savings,'evidence_correct':evidence['correct'],'evidence_total':len(cases)},indent=2))


if __name__=='__main__':main()
