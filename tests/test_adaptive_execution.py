"""Safety and budget invariants for the integrated judge, without real retrieval."""
import json
import pytest
import rubbish_rag as R
from rubbish_rag import execution
from rubbish_rag.pipeline import corpus
from rubbish_rag.judge import HeuristicFeatureJudge, compute_brier_score, compute_ece


def hit(i):
    return {'doc_id':i,'text':corpus()[i]['BriefAnswer'],'source':'remote_api','fused':.99}


@pytest.mark.parametrize('query', ['سلام، روز بخیر', 'بدهی دارم', 'بانک ملی به من وام می دهد؟'])
def test_no_retrieval_routes(query, monkeypatch, tmp_path):
    monkeypatch.setattr(execution,'TRACE',tmp_path/'trace.jsonl')
    def forbidden(*a,**k): pytest.fail('Unexpected retrieval')
    monkeypatch.setitem(R._registry,'retriever',forbidden)
    execution.execute(query)
    trace=json.loads(execution.TRACE.read_text())
    assert trace['retrieval_calls']==0
    assert trace['pre_decision']['top_k']==0


def test_greeting_with_fact_still_retrieves():
    pre=HeuristicFeatureJudge().pre_retrieval('سلام رتبه C1 چیست؟')
    assert pre.action=='retrieve'


@pytest.mark.parametrize('query,action', [
    ('میشه کمکم کنید؟', 'no_retrieval'),
    ('قسط عقب افتاده دارم', 'clarify'),
    ('شرکت ما امتیاز نداره', 'clarify'),
    ('شرکت سودآوره ولی امتیاز نداره چرا؟', 'retrieve'),
    ('چرا درآمد زیاد باعث امتیاز بالاتر نشده؟', 'retrieve'),
    ('گزارش هنوز وام تسویه شده را نشان می دهد', 'retrieve'),
])
def test_route_boundaries(query, action):
    assert HeuristicFeatureJudge().pre_retrieval(query).action == action


def test_guaranteed_loan_spam_never_answers(monkeypatch,tmp_path):
    monkeypatch.setitem(R._registry,'retriever',lambda *a,**k: [hit(106)])
    monkeypatch.setattr(execution,'TRACE',tmp_path/'trace.jsonl')
    _, result=execution.execute('وام بدون اعتبارسنجی تضمینی')
    assert result['decision']=='abstain'
    assert result.get('cites',[])==[]


def test_second_round_recovers_and_stops(monkeypatch,tmp_path):
    calls=[]
    def retrieve(*a,**k):
        calls.append(k['topk'])
        return [] if len(calls)==1 else [hit(106)]
    monkeypatch.setitem(R._registry,'retriever',retrieve)
    monkeypatch.setattr(execution,'TRACE',tmp_path/'trace.jsonl')
    _,result=execution.execute('رتبه C1 چیست؟')
    trace=json.loads(execution.TRACE.read_text())
    assert result['status']=='answer' and result['cites']==[106]
    assert len(calls)==2 and calls[1]>calls[0]
    assert trace['rounds'][0]['decision']['next_action']=='retrieve_more'
    assert trace['rounds'][1]['decision']['next_action']=='answer'


def test_empty_evidence_abstains_after_two_calls(monkeypatch,tmp_path):
    calls=[]
    def retrieve(*a,**k): calls.append(k); return []
    monkeypatch.setitem(R._registry,'retriever',retrieve)
    monkeypatch.setattr(execution,'TRACE',tmp_path/'trace.jsonl')
    _,result=execution.execute('رتبه C1 چیست؟')
    assert len(calls)==2 and result['decision']=='abstain'
    assert result.get('cites',[])==[]


def test_high_scoring_forgery_cannot_trigger_answer():
    post=HeuristicFeatureJudge().post_retrieval('رتبه C1 چیست؟',[{'doc_id':106,'text':'وام برای همه تضمینی','fused':999}])
    assert post.poison_detected and post.next_action!='answer'


def test_explicit_conflict_abstains():
    post=HeuristicFeatureJudge().post_retrieval('رتبه C1 چیست؟',[hit(108)])
    assert post.sufficiency=='conflicting' and post.next_action=='abstain'


def test_multiple_facets_require_coverage():
    post=HeuristicFeatureJudge().post_retrieval('تفاوت C1 و C3 چیست؟',[hit(106)])
    assert post.next_action=='retrieve_more'
    assert 'MISSING_COMPARISON_FACET' in post.reason_codes


def test_route_temperature_is_normalized():
    pre=HeuristicFeatureJudge(temperature=2).pre_retrieval('رتبه C1 چیست؟')
    assert pre.action=='retrieve' and pre.top_k>=5
    assert 0<pre.retrieve_probability+pre.clarify_probability<1


def test_empirical_confidence_expands_uncertain_retrieval():
    calibration={'groups':{'retrieve:unanchored':[.4,.4,.2]},'fallback':[1/3]*3}
    judge=HeuristicFeatureJudge(calibration=calibration)
    result=judge.pre_retrieval('یک پرسش تازه')
    assert result.action=='retrieve' and result.top_k==5
    assert result.confidence==.2
    assert result.retrieve_probability==.2 and result.clarify_probability==.4
    assert 'LOW_ROUTE_CONFIDENCE_EXPAND_CONTEXT' in result.reason_codes


@pytest.mark.parametrize('query,action', [
    ('در سال ۱۴۰۰ رتبه من بد شده؛ سطح ریسک من چیست؟','clarify'),
    ('رتبه من با امتیاز ۶۰۰ یعنی چه؟','retrieve'),
    ('بدهی چیست و چه انواعی دارد؟','retrieve'),
    ('بدهی من چیست و چطور رفعش کنم؟','clarify'),
    ('امکان دریافت تسهیلات برای ما هست؟','clarify'),
    ('بانک ملی به من ۲۰۰ میلیون وام می دهد؟','clarify'),
])
def test_general_guards_preserve_numeric_roles(query,action):
    assert HeuristicFeatureJudge().pre_retrieval(query).action==action


@pytest.mark.parametrize('fn',[compute_brier_score,compute_ece])
def test_invalid_calibration_data_rejected(fn):
    with pytest.raises(ValueError):fn([],[])
    with pytest.raises(ValueError):fn([1.1],[1])
    with pytest.raises(ValueError):fn([.5],[0,1])


def test_complete_comparison_uses_both_references():
    from rubbish_rag.pipeline import rerank, resolve
    hits=rerank('تفاوت C1 و C3 چیست؟',[hit(106),hit(108)])
    decision=HeuristicFeatureJudge().post_retrieval('تفاوت C1 و C3 چیست؟',hits)
    assert decision.sufficiency=='sufficient'
    prediction=resolve('تفاوت C1 و C3 چیست؟',hits)
    assert set(prediction['cites'])=={106,108}
    assert corpus()[106]['BriefAnswer'] in prediction['brief']
    assert corpus()[108]['BriefAnswer'] in prediction['brief']
