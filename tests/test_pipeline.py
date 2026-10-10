"""Unit tests for the RubbishRAG pipeline stages and decorators."""
import pytest
import rubbish_rag as R
import rubbish_rag.pipeline


def test_decorator_registration():
    assert R.get("chunker") is not None
    assert R.get("retriever") is not None
    assert R.get("reranker") is not None
    assert R.get("resolver") is not None


def test_chunker_structure():
    chunker = R.get("chunker")
    mock_docs = [
        {
            "doc_id": 42,
            "Question": "سوال تستی؟",
            "Category": "مفاهیم",
            "BriefAnswer": "پاسخ کوتاه",
            "Answer": "پاسخ تفصیلی برای تست سیستم تقطیع چانک‌ها که باید پردازش شود.",
            "Keyword": "تست",
        }
    ]
    chunks = chunker(mock_docs)
    assert len(chunks) >= 1
    c = chunks[0]
    assert "doc_id" in c and c["doc_id"] == 42
    assert "text" in c
    assert "chunk_id" in c


def test_reranker_contract():
    reranker = R.get("reranker")
    from rubbish_rag.pipeline import corpus
    mock_hits = [
        {"doc_id": 106, "text": corpus()[106]["BriefAnswer"], "fused": 0.8},
        {"doc_id": 91, "text": corpus()[91]["BriefAnswer"], "fused": 0.5},
    ]
    reranked = reranker("رتبه C1", mock_hits)
    assert len(reranked) == 2
    assert "rerank_score" in reranked[0]
    # First item has more overlap with query
    assert reranked[0]["doc_id"] == 106


def test_resolver_contract():
    resolver = R.get("resolver")
    mock_hits = [
        {"doc_id": 106, "text": "رتبه C1 در بازه ۵۶۰ تا ۵۷۹ است", "fused": 0.9},
        {"doc_id": 91, "text": "اطلاعات رتبه‌بندی", "fused": 0.7},
    ]
    res = resolver("رتبه C1", mock_hits)
    assert isinstance(res, dict)
    assert "status" in res
    assert res["status"] in ("answer", "clarify")
    if res["status"] == "answer":
        assert "brief" in res
        assert "cites" in res
        assert isinstance(res["cites"], list)
        assert len(res["cites"]) > 0
