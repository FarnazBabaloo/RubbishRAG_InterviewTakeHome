"""Supplemental retrieval regressions with synthetic records, not benchmark labels."""
import pytest
from rubbish_rag.local_search import QuestionIndex
from rubbish_rag.pipeline import fuse_hits


@pytest.fixture
def index():
    return QuestionIndex({
        6001: {'Question':'بانک ملی با رتبه A2 وام می دهد؟', 'Keyword':'تسهیلات', 'BriefAnswer':'پاسخ ملی A2'},
        6002: {'Question':'بانک صادرات با رتبه E3 وام می دهد؟', 'Keyword':'تسهیلات', 'BriefAnswer':'پاسخ صادرات E3'},
        6003: {'Question':'گزارش اعتباری شرکت شامل چه مواردی است؟', 'Keyword':'شرکت', 'BriefAnswer':'گزارش شرکت'},
    })


def test_search_handles_reordered_paraphrase_and_script(index):
    hits = index.search('وام با رتبه a۲ در بانك ملي')
    assert hits[0]['doc_id'] == 6001
    assert hits[0]['source'] == 'local_corpus'
    assert 'local_bm25' in hits[0]
    assert 'bm25' not in hits[0] and 'dense' not in hits[0] and 'fused' not in hits[0]


def test_entity_swap_changes_candidate_order(index):
    assert index.search('رتبه E۳ در بانک صادرات وام')[0]['doc_id'] == 6002
    assert index.search('رتبه A۱ در بانک ملی وام')[0]['doc_id'] == 6001


def test_unsupported_or_empty_query_has_no_candidates(index):
    assert index.search('') == []
    assert index.search('برف هواشناسی فردا') == []


def test_local_search_deterministic_and_bounded(index):
    assert index.search('وام بانک',limit=1) == index.search('وام بانک',limit=1)
    assert len(index.search('وام بانک',limit=1)) <= 1


def test_mixed_rrf_preserves_score_provenance():
    hits = [{'doc_id':1,'bm25':8.0,'dense':0.6,'fused':0.8,'source':'remote_api'},
            {'doc_id':2,'local_bm25':120.0,'source':'local_corpus'}]
    fused = fuse_hits(hits)
    assert all(h['rrf_score'] > 0 for h in fused)
    local = next(h for h in fused if h['doc_id']==2)
    assert 'bm25' not in local and 'fused' not in local
    assert hits[0].get('rrf_score') is None


def test_answer_terms_retrieve_a_paraphrased_question():
    index = QuestionIndex({
        6001:{'Question':'رتبه C1 یعنی چیست؟','Keyword':'رتبه','BriefAnswer':'امتیاز بین 560 و 579 است'},
        6002:{'Question':'روش دریافت گزارش چیست؟','Keyword':'گزارش','BriefAnswer':'گزارش شامل اطلاعات مالی است'},
    })
    assert index.search('برای رتبه c۱ بازه امتیاز چند است؟')[0]['doc_id'] == 6001


def test_negation_is_not_a_stopword():
    from rubbish_rag.normalize_fa import content_tokens
    assert 'نمی' in content_tokens('وام نمی شود دریافت کرد')
    assert 'نیست' in content_tokens('دریافت وام تضمینی نیست')


@pytest.mark.parametrize('query,expected',[
    ('رتبه a۲ چه معنایی دارد؟',101),
    ('برای رتبه c۱ بازه امتیاز چند است؟',106),
    ('رتبه c۳ چه بازه امتیازی دارد؟',108),
    ('گزارش اعتباری شرکت شامل کدام اطلاعات است؟',0),
    ('تجدید محاسبه امتیاز اعتباری امکان دارد؟',294),
    ('آیا سفته در اعتبارسنجی لحاظ می شود؟',271),
])
def test_public_corpus_paraphrase_regression(query,expected):
    from rubbish_rag.local_search import local_candidates
    from rubbish_rag.pipeline import rerank,resolve
    result = resolve(query,rerank(query,local_candidates(query))[:5])
    assert result['cites'][0] == expected


def test_cli_probe_prints_local_scores_without_fabricating_remote_scores(monkeypatch,tmp_path,capsys):
    import argparse
    import rubbish
    import server._hidden_retriever as remote
    from rubbish_rag import execution
    monkeypatch.setattr(remote,'remote_retrieve',lambda *args,**kwargs:[])
    monkeypatch.setattr(execution,'TRACE',tmp_path/'trace.jsonl')
    rubbish.cmd_bepar(argparse.Namespace(query='رتبه C1 چیست؟',topk=5,key='demo-key'))
    output=capsys.readouterr().out
    assert 'source=local_corpus' in output
    assert 'bm25=n/a' in output
    assert '--- resolve ---' in output
