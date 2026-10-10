"""Supplement remote retrieval with a BM25 index of the public reference corpus.

No benchmark queries or answer labels are loaded. Document IDs come from CSV
row positions. Local scores remain separate from remote API scores in telemetry.
"""
import math
from collections import Counter
from functools import lru_cache
from typing import Any

from .normalize_fa import content_tokens


class QuestionIndex:
    """Small fielded question index with query coverage for candidate admission."""

    def __init__(self, docs: dict[int, dict[str, str]]):
        self.docs = docs
        self.frequencies = {}
        self.question_terms = {}
        self.postings: dict[str, dict[int, int]] = {}
        for doc_id, doc in docs.items():
            terms = content_tokens(doc.get('Question', ''))
            self.question_terms[doc_id] = set(terms)
            # Include verified answer terminology: paraphrases often ask about
            # facts expressed in an answer but absent from the FAQ question.
            freq = Counter(terms * 2 + content_tokens(doc.get('Keyword', ''))
                           + content_tokens(doc.get('BriefAnswer', '')))
            self.frequencies[doc_id] = freq
            for term, count in freq.items():
                self.postings.setdefault(term, {})[doc_id] = count
        self.lengths = {i: sum(f.values()) for i,f in self.frequencies.items()}
        self.average_length = sum(self.lengths.values()) / max(1,len(docs)) or 1.0
        self.idf = {t: math.log(1 + (len(docs)-len(p)+0.5)/(len(p)+0.5)) for t,p in self.postings.items()}

    def search(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        # Stable summation order across Python hash seeds and processes.
        q = sorted(set(content_tokens(query)))
        if not q:
            return []
        weight = sum(self.idf.get(t, math.log(1+len(self.docs))) for t in q) or 1.0
        scores: Counter = Counter()
        matched: Counter = Counter()
        for term in q:
            for doc_id, frequency in self.postings.get(term, {}).items():
                denominator = frequency + 1.2 * (0.25 + 0.75*self.lengths[doc_id]/self.average_length)
                scores[doc_id] += self.idf[term]*frequency*2.2/denominator
                matched[doc_id] += self.idf[term]
        admitted = [(i,score,matched[i]/weight) for i,score in scores.items() if matched[i]/weight >= 0.35]
        admitted.sort(key=lambda item:(-item[1],item[0]))
        return [{'doc_id':i, 'chunk_id':f'{i}:local', 'source':'local_corpus',
                 'text':self.docs[i].get('BriefAnswer') or self.docs[i].get('Answer',''),
                 'category':self.docs[i].get('Category',''),
                 'local_bm25':score,'term_coverage':coverage}
                for i,score,coverage in admitted[:limit]]


@lru_cache(maxsize=1)
def question_index() -> QuestionIndex:
    from .pipeline import corpus
    return QuestionIndex(corpus())


def local_candidates(query: str, limit: int = 6) -> list[dict[str, Any]]:
    return question_index().search(query,limit)
