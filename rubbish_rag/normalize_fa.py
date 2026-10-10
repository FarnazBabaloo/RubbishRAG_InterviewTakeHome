"""Persian text normalization for Credit-Scoring QA.

Handles:
  1. Persian / Arabic digits -> ASCII numerals (e.g. ۲۵۰ / ٢٥٠ -> 250)
  2. Arabic character variants -> Canonical Persian (ي -> ی, ك -> ک, ة -> ه)
  3. Half-spaces (ZWNJ \u200c) -> standardized whitespace
  4. Common colloquial speech and enclitic suffixes (رتبم -> رتبه من, میشه -> می‌شود)
  5. Whitespace collapsing and punctuation strip
"""
import re
import unicodedata
from functools import lru_cache

FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EN_DIGITS = "0123456789"
_DIGIT_MAP = {ord(a): b for a, b in zip(FA_DIGITS + AR_DIGITS, EN_DIGITS * 2)}

COLLOQUIAL_MAP = {
    "میشه": "می‌شود",
    "نمیشه": "نمی‌شود",
    "چیه": "چیست",
    "چقد": "چقدر",
    "رتبم": "رتبه من",
    "رتبه‌م": "رتبه من",
    "امتیازم": "امتیاز من",
    "سابقه‌م": "سابقه من",
    "سابقم": "سابقه من",
    "شرکتم": "شرکت من",
    "چکم": "چک من",
    "وامم": "وام من",
    "قسطام": "اقساط من",
    "قسطامو": "اقساط من را",
    "گزارشم": "گزارش من",
    "گزارشمو": "گزارش من را",
    "مالیاتم": "مالیات من",
    "مالیاتمو": "مالیات من را",
}


@lru_cache(maxsize=8192)
def normalize_fa(text: str) -> str:
    """Normalize text; bounded memoization reuses immutable string results.

    Cache keys contain the full input, never just a document ID, so changed or
    forged excerpts cannot inherit verification of a different text.
    """
    if not text:
        return ""
    # 1. Digits mapping
    s = unicodedata.normalize("NFKC", text).translate(_DIGIT_MAP)
    # 2. Arabic letter variants
    s = s.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه").replace("ـ", "")
    # Expand complete words before folding ZWNJ, including enclitic forms.
    for k, v in COLLOQUIAL_MAP.items():
        s = re.sub(rf"\b{k}\b", v, s)
    s = s.replace("\u200c", " ")
    s = re.sub(r"[\u064b-\u065f\u0670]", "", s)
    # Preserve decimal and grouping separators; punctuation is tokenized separately.
    s = s.replace("٫", ".").replace("٬", ",")
    return re.sub(r"\s+", " ", s).strip()


def tokens(text: str) -> list[str]:
    """Tokenize without discarding negation or decimal numeric meaning."""
    return re.findall(r"[a-z]+[0-9]*|[0-9]+(?:[.,][0-9]+)*|[\u0621-\u06ff]+", normalize_fa(text).lower())


# Question phrasing should not outweigh domain concepts. Negation is deliberately
# absent from this set; removing it could reverse the meaning of a regulation.
QUESTION_STOPWORDS = frozenset("از به در با برای که چه چند کدام چگونه چطور آیا است هست باشد شود می یعنی چی من شما آن این یک را و".split())


def content_tokens(text: str) -> list[str]:
    """Terms used in lexical retrieval, preserving ranks, numbers and negation."""
    return [term for term in tokens(text) if term not in QUESTION_STOPWORDS]
