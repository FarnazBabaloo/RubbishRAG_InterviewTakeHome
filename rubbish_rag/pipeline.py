"""Candidate-owned, grounded Persian retrieval pipeline.

Remote scores remain observable. Client rank fusion, metadata-aware ranking and
resolution operate on API candidates and a public-corpus index, never on
benchmark answer IDs.
"""
import csv
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import rubbish_rag as R
from .normalize_fa import normalize_fa, tokens, content_tokens
from .slots import extract_slots
from .clarification import clarification

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def corpus() -> dict[int, dict[str, str]]:
    """Cache the public reference corpus; retain original CSV row IDs."""
    with (ROOT / "corpus/test.csv").open(encoding="utf-8-sig") as stream:
        return dict(enumerate(csv.DictReader(stream)))


@lru_cache(maxsize=1)
def inverse_document_frequency() -> dict[str, float]:
    counts: Counter = Counter()
    docs = corpus()
    for doc in docs.values():
        counts.update(set(tokens(" ".join(doc.get(k, "") for k in ("Question", "Keyword", "Answer")))))
    return {t: math.log(1 + (len(docs) + 1) / (n + 1)) for t, n in counts.items()}


@R.chunker
def chunk(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pack sentences into bounded chunks, preserving category and question.

    Long sentences fall back to word boundaries; numbers and their units stay
    together whenever the source sentence fits. Body text is never truncated.
    """
    result = []
    for doc in docs:
        body = doc.get("Answer") or doc.get("BriefAnswer") or ""
        prefix = "[" + " | ".join(str(doc.get(k, "")) for k in ("Category", "Keyword")) + "]"
        segments = []
        for sentence in re.split(r"(?<=[.!؟?؛])\s+|\n+", body):
            current = ""
            for word in sentence.split():
                if current and len(current) + len(word) + 1 > 400:
                    segments.append(current)
                    current = ""
                current = (current + " " + word).strip()
            if current:
                segments.append(current)
        packed = []
        for segment in segments:
            if packed and len(packed[-1]) + len(segment) + 1 <= 400:
                packed[-1] += " " + segment
            else:
                packed.append(segment)
        for index, part in enumerate(packed):
            result.append({"doc_id": doc["doc_id"], "chunk_id": f'{doc["doc_id"]}:{index}',
                           "category": doc.get("Category", ""), "question": doc.get("Question", ""),
                           "text": prefix + "\n" + part, "body": part})
    return result


def fuse_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """RRF over sparse and dense ranks; retain server fused scores unchanged."""
    result = [dict(h) for h in hits]
    if not result:
        return result
    # Rank each channel within its own candidate population. Local BM25 must
    # never masquerade as a score returned by the sealed server.
    contributions = [[] for _ in result]
    for field in ("bm25", "dense", "local_bm25"):
        values = [float(h[field]) for h in result if field in h]
        ranks = {v: 1 + sum(other > v for other in values) for v in set(values)}
        for i,h in enumerate(result):
            if field in h:
                contributions[i].append(1 / (60 + ranks[float(h[field])]))
    for h,parts in zip(result,contributions):
        # Normalize for channel availability so one local channel and two
        # remote channels have comparable rank-based contributions.
        h["rrf_score"] = 2 * sum(parts) / len(parts) if parts else 0.0
    return sorted(result, key=lambda h: h["rrf_score"], reverse=True)


@R.retriever
def retrieve(query: str, topk: int = 5, api_key: str = "demo-key") -> list[dict[str, Any]]:
    """One remote call per factual query, with a bounded candidate pool.

    Extra candidates improve reranking recall without making additional calls.
    """
    from server._hidden_retriever import remote_retrieve
    if not 1 <= topk <= 20:
        raise ValueError("topk must be between 1 and 20")
    normalized = normalize_fa(query)
    # Public traces show ASCII-only rewriting loses Persian numeric matches.
    # Preserve both script representations within the same remote request.
    rewritten = query if normalized == query else query + " " + normalized
    hits = remote_retrieve(rewritten, topk=min(20, max(10, topk * 2)), api_key=api_key)
    from .local_search import local_candidates
    remote_hits = [{**h, "source": "remote_api"} for h in hits]
    return fuse_hits(remote_hits + local_candidates(query))


def grounded(hit: dict[str, Any], doc: dict[str, str] | None) -> bool:
    """Require both corpus membership and corroborated content.

    This rejects synthetic IDs and forged content attached to a legitimate ID.
    """
    if doc is None:
        return False
    text = normalize_fa(hit.get("body", hit.get("text", ""))).strip()
    if not text:
        return False
    reference = "\n".join(doc.get(k, "") for k in ("BriefAnswer", "Answer"))
    return text in normalize_fa(reference)


@R.reranker
def rerank(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank verified evidence using IDF overlap and explicit entity agreement."""
    q_tokens = set(content_tokens(query))
    idf = inverse_document_frequency()
    total_weight = sum(idf.get(t, 1.0) for t in q_tokens) or 1.0
    slots = extract_slots(query)
    requested_ranks = set(re.findall(r"\b[A-E][1-3]\b", normalize_fa(query).upper()))
    ranked = []
    for hit in fuse_hits(hits):
        h = dict(hit)
        doc = corpus().get(h["doc_id"])
        if not grounded(h, doc):
            continue
        source = " ".join(doc.get(k, "") for k in ("Question", "Keyword", "BriefAnswer"))
        overlap = sum(idf.get(t, 1.0) for t in q_tokens & set(tokens(source))) / total_weight
        doc_slots = extract_slots(doc.get("Question", ""))
        conflicts = []
        agreement = 0
        for key in ("RANK", "BANK", "PERSONA"):
            wanted, actual = slots[key], doc_slots[key]
            if key == "RANK" and actual in requested_ranks:
                agreement += 1
                continue
            if wanted and actual and wanted != "both" and actual != "both":
                if wanted != actual:
                    conflicts.append(key)
                else:
                    agreement += 1
        if conflicts:
            continue
        # Semantic agreement dominates rank fusion; both components are logged.
        h["lexical_score"] = overlap
        h["slot_agreement"] = agreement
        # Entity agreement reinforces actual relevance; generic persona matches
        # must not dominate more specific question overlap.
        h["rerank_score"] = overlap * (1 + agreement / 3) + 5 * h["rrf_score"]
        ranked.append(h)
    ranked.sort(key=lambda h: (-h["rerank_score"], h["doc_id"], h.get("chunk_id", "")))
    unique = {}
    for h in ranked:
        unique.setdefault(h["doc_id"], h)
    return list(unique.values())


@R.resolver
def resolve(query: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a complete verified answer, or a structured clarification."""
    request = clarification(query)
    if request:
        return request
    verified = [h for h in hits if grounded(h, corpus().get(h["doc_id"]))]
    if not verified or verified[0].get("lexical_score", 0) <= 0:
        return {"status": "clarify", "question": "شواهد کافی پیدا نشد؛ موضوع پرسش را دقیق تر مشخص کنید.",
                "options": ["رتبه اعتباری", "تسهیلات", "چک", "مالیات"]}
    top = verified[0]
    doc = corpus()[top["doc_id"]]
    requested_ranks = set(re.findall(r"\b[A-E][1-3]\b", normalize_fa(query).upper()))
    if len(requested_ranks) > 1:
        by_rank = {}
        for h in verified:
            record = corpus()[h["doc_id"]]
            rank = extract_slots(record.get("Question", ""))["RANK"]
            if rank in requested_ranks:
                by_rank.setdefault(rank, (h["doc_id"], record))
        if requested_ranks.issubset(by_rank) and len(requested_ranks) <= 3:
            return {"status": "answer",
                    "brief": "\n".join(f"{rank}: {by_rank[rank][1].get('BriefAnswer') or by_rank[rank][1].get('Answer', '')}" for rank in sorted(requested_ranks)),
                    "cites": [by_rank[rank][0] for rank in sorted(requested_ranks)],
                    "slots": extract_slots(query)}
    return {"status": "answer", "brief": doc.get("BriefAnswer") or doc.get("Answer", ""),
            "cites": [h["doc_id"] for h in verified[:3] if h.get("lexical_score", 0) > 0],
            "slots": extract_slots(query)}
