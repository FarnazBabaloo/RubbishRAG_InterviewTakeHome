"""Offline regressions for real failure mechanisms; no server calls."""
import copy
import pytest
from rubbish_rag.normalize_fa import normalize_fa
from rubbish_rag.slots import extract_slots
from rubbish_rag.pipeline import chunk, fuse_hits, rerank, resolve, retrieve, corpus


@pytest.mark.parametrize('text', ['میشه', 'نمیشه', 'رتبه‌م', 'بانك ملي', '۲۵۰٫۵', 'سابقه‌م'])
def test_normalization_idempotence(text):
    once = normalize_fa(text)
    assert normalize_fa(once) == once


def test_numbers_and_slots_do_not_change_meaning():
    assert extract_slots('رتبه A ۲ و امتیاز ۲۵۰٫۵')['NUMS'] == ['250.5']
    assert extract_slots('بانک توسعه صادرات')['BANK'] == 'توسعه صادرات'
    assert extract_slots('صادرات/واردات شرکت')['BANK'] is None
    assert extract_slots('شخص حقوقی')['PERSONA'] == 'حقوقی'
    assert normalize_fa('رتبه‌م') == 'رتبه من'


def test_chunk_coverage_and_metadata():
    answer = 'عدد ۲۴ ساعت باید همراه واحد بماند. ' * 40
    pieces = chunk([{'doc_id': 7, 'Answer': answer, 'Category': 'چک', 'Keyword': 'رفع سوءاثر'}])
    assert ' '.join(p['body'] for p in pieces) == ' '.join(answer.split())
    assert all(p['text'].startswith('[چک | رفع سوءاثر]') for p in pieces)
    assert all(len(p['body']) <= 400 for p in pieces)
    assert all('۲۴ ساعت' in p['body'] for p in pieces)


def test_rrf_is_invariant_to_score_scale():
    hits = [{'doc_id': i, 'bm25': b, 'dense': d, 'fused': 0.2} for i,b,d in [(1,1000,0.1),(2,5,0.8),(3,4,0.7)]]
    scaled = [{**h, 'bm25': h['bm25'] * 100} for h in hits]
    assert [(h['doc_id'],h['rrf_score']) for h in fuse_hits(hits)] == [(h['doc_id'],h['rrf_score']) for h in fuse_hits(scaled)]


def test_poison_forgery_dedup_and_rank_conflict():
    good = {'doc_id': 106, 'text': corpus()[106]['BriefAnswer'], 'fused': 0.5}
    wrong = {'doc_id': 108, 'text': corpus()[108]['BriefAnswer'], 'fused': 1.0}
    poison = {'doc_id': 9004, 'text': 'رتبه C1 ' * 20, 'fused': 1.0}
    forged = {'doc_id': 106, 'text': 'وام برای همه تضمین شده', 'fused': 1.0}
    hits = [poison,wrong,forged,good,copy.deepcopy(good)]
    saved = copy.deepcopy(hits)
    ranked = rerank('رتبه C1 چیست؟', hits)
    assert [h['doc_id'] for h in ranked] == [106]
    assert hits == saved
    assert resolve('رتبه C1 چیست؟',ranked)['brief'] == corpus()[106]['BriefAnswer']


def test_ambiguity_does_not_need_retrieval(monkeypatch, tmp_path):
    from rubbish_rag import execution
    import rubbish_rag as R
    monkeypatch.setattr(execution, 'TRACE', tmp_path/'trace.jsonl')
    def forbidden(*args, **kwargs):
        pytest.fail('Ambiguous input consumed retrieval quota')
    monkeypatch.setitem(R._registry,'retriever',forbidden)
    hits,result = execution.execute('بدهی دارم')
    assert hits == [] and result['status'] == 'clarify'
    assert 'مالیاتی' in ' '.join(result['options'])


@pytest.mark.parametrize('topk', [0, 21])
def test_invalid_topk_rejected_before_clarification(topk):
    from rubbish_rag.execution import execute
    with pytest.raises(ValueError, match='topk'):
        execute('بدهی دارم', topk=topk)


def test_registered_retriever_one_call_and_normalized_query(monkeypatch):
    import server._hidden_retriever as remote
    calls = []
    def fake(query, topk, api_key):
        calls.append((query,topk,api_key))
        return []
    monkeypatch.setattr(remote,'remote_retrieve',fake)
    hits = retrieve('بانك ملي ۲۵۰',api_key='demo-key')
    assert all(h['source'] == 'local_corpus' for h in hits)
    assert calls == [('بانك ملي ۲۵۰ بانک ملی 250',10,'demo-key')]
