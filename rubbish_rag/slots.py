"""Query entities and slot extraction for Credit-Scoring QA.

Detects:
  - BANK: Known Iranian banks (ملی, صادرات, ملت, تجارت, سپه, پاسارگاد, ...)
  - RANK: Credit score grade A1..E3
  - PERSONA: حقیقی (individual) vs حقوقی (corporate entity)
  - DEBT: Category of debt (چک, مالیاتی, تسهیلاتی, محکومیت)
  - AMOUNT: Numerical figures and threshold amounts
  - TIME: Temporal indicators (ساعت, روز, ماه, سال)
  - NEGATED: Boolean negation indicator
"""
import re
from typing import Any, Dict

from .normalize_fa import normalize_fa

BANKS = [
    "ملی", "صادرات", "تجارت", "توسعه صادرات", "ملت", "سپه",
    "پاسارگاد", "دیجی پی", "دیجی‌پی", "اسنپ پی", "اسنپ‌پی", "تارا",
    "دیجیکالا", "دیجی‌کالا",
]

RANKS = [f"{L}{N}" for L in "ABCDE" for N in "123"]

# Fixed vocabulary: prepare longest-name-first matching once at module load.
_BANK_PATTERNS = tuple(
    (bank, re.compile(rf"(?<!\w){re.escape(normalize_fa(bank))}(?!\w)"))
    for bank in sorted(BANKS, key=lambda b: len(normalize_fa(b)), reverse=True)
)

DEBT_PATTERNS = [
    ("چک", ["چک", "صیادی", "برگشتی", "رفع سوءاثر", "رفع سواثر"]),
    ("مالیاتی", ["مالیات", "جریمه مالیاتی", "امور مالیاتی"]),
    ("تسهیلاتی", ["تسهیلات", "وام", "قسط", "اقساط", "معوق"]),
    ("محکومیت", ["محکومیت", "ورشکستگی", "دادگاه", "قضایی"]),
]


def extract_slots(query: str) -> Dict[str, Any]:
    """Extract domain entity slots from query."""
    norm = normalize_fa(query or "")
    slots: Dict[str, Any] = {
        "BANK": None,
        "RANK": None,
        "PERSONA": None,
        "DEBT": None,
        "AMOUNT": None,
        "TIME": False,
        "NUMS": [],
        "NEGATED": False,
    }

    # Bank
    for b, pattern in _BANK_PATTERNS:
        if pattern.search(norm):
            # Export/import volume is not a reference to Bank Saderat.
            if b == "صادرات" and re.search(r"صادرات\s*[/و]\s*واردات|صادرات و واردات", norm):
                continue
            slots["BANK"] = b
            break

    # Rank (A1..E3)
    rank_match = re.search(r"\b([A-Ea-e])\s*([1-3])\b", norm)
    if rank_match:
        slots["RANK"] = f"{rank_match.group(1).upper()}{rank_match.group(2)}"

    # Persona
    is_individual = any(w in norm for w in ["حقیقی", "شخصی", "فرد", "خودم"])
    is_legal = any(w in norm for w in ["حقوقی", "شرکت", "کسب و کار", "سازمان", "شرکت‌ها"])
    if is_individual and is_legal:
        slots["PERSONA"] = "both"
    elif is_individual:
        slots["PERSONA"] = "حقیقی"
    elif is_legal:
        slots["PERSONA"] = "حقوقی"

    # Debt type
    for cat, keywords in DEBT_PATTERNS:
        if any(w in norm for w in keywords):
            slots["DEBT"] = cat
            break

    # Numbers
    numeric_text = re.sub(r"\b[A-Ea-e]\s*[1-3]\b", "", norm)
    nums = re.findall(r"(?<![\w])\d+(?:[,.]\d+)*(?![\w])", numeric_text)
    slots["NUMS"] = [n.replace(",", "") for n in nums]
    if slots["NUMS"]:
        slots["AMOUNT"] = slots["NUMS"][0]

    # Time frame
    if any(w in norm for w in ["سال", "ماه", "روز", "ساعت", "مدت", "بازه"]):
        slots["TIME"] = True

    # Negation
    if any(w in norm for w in ["نمی", "نیست", "فاقد", "بدون", "ندارم", "ندارد", "معاف", "عدم"]):
        slots["NEGATED"] = True

    return slots


def extract_slots_broken(query: str) -> dict:
    """Starter stub (detects nothing). Used to demonstrate baseline broken behavior."""
    return {"BANK": None, "RANK": None, "PERSONA": None, "DEBT": None,
            "AMOUNT": None, "TIME": None}
