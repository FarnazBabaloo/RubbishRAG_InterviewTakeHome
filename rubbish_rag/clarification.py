"""Conservative missing-information guards before retrieval.

These rules identify incomplete requests, not facts or answers. Explicit topic
anchors keep demonstratives in complete factual questions from blocking retrieval.
"""
import re
from typing import Any

from .normalize_fa import normalize_fa
from .slots import extract_slots


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text) is not None


def _request(question: str, options: list[str], *missing: str) -> dict[str, Any]:
    return {'status': 'clarify', 'question': question, 'options': options,
            'missing_slots': list(missing)}


def clarification(query: str) -> dict[str, Any] | None:
    """Ask for absent referents or decision slots; leave factual topics retrievable."""
    norm = normalize_fa(query)
    slots = extract_slots(query)
    # Years, amounts and elapsed days are not evidence of a supplied score.
    score_supplied = bool(slots['RANK'] or _has(
        r'(?:امتیاز|رتبه)(?:\s+(?:اعتباری|من|ما|فعلی|عددی|برابر|با|شده|است))*\s*[:=]?\s*\d+|\d+\s*امتیاز|^\d+(?:[.,]\d+)?\s*(?:شدم|دارم|است|هستم|$)', norm))
    # An installment plan does not identify whether a debt is tax, loan, etc.
    debt_source = _has(r'مالیات|چک|صیادی|محکومیت|دادگاه|قضایی|ورشکست|تامین|تأمین|وام|تسهیلات|بانکی', norm)
    debt_definition = not _has(r'\bمن\b|\bما\b|دارم|داریم', norm) and _has(r'بدهی.*(?:چیست|چه معن|چه نوع|چه انواع)|(?:انواع|تعریف).*بدهی', norm)
    if 'بدهی' in norm and not debt_source and not debt_definition:
        return _request('این بدهی مربوط به کدام مورد است؟',
                        ['بدهی تسهیلاتی / وام', 'بدهی مالیاتی', 'چک برگشتی', 'محکومیت مالی'], 'debt_type')

    # Detect a request to decide eligibility, rather than general repayment,
    # guarantee, or credit-history questions containing the word "loan".
    loan = _has(r'وام|تسهیلات', norm)
    eligibility = _has(r'بگیرم|بگیریم|دریافت کنم|دریافت کنیم|می\s*ده[دند]*|میده|قبول|تایید|تأیید|واجد شرایط|شرایط من|شرایط ما', norm)
    eligibility = eligibility or (_has(r'امکان|می\s*توانم|می\s*توانیم', norm) and _has(r'برای من|برای ما|می\s*توانم|می\s*توانیم', norm))
    report_display = _has(r'نشان می\s*ده|نمایش', norm)
    if loan and eligibility and not report_display:
        missing = ([] if slots['BANK'] else ['bank']) + ([] if score_supplied else ['rank'])
        if missing == ['bank']:
            return _request('برای دریافت تسهیلات از کدام بانک درخواست دارید؟',
                            ['بانک ملی', 'بانک صادرات', 'بانک ملت', 'بانک تجارت'], *missing)
        if missing == ['rank']:
            return _request('رتبه اعتباری یا امتیاز عددی فعلی شما چیست؟',
                            ['رتبه را وارد می کنم', 'امتیاز عددی را وارد می کنم', 'رتبه و امتیاز را نمی دانم'], *missing)
        if missing:
            return _request('نام بانک و رتبه اعتباری یا امتیاز عددی فعلی را مشخص کنید.',
                            ['نام بانک و رتبه را وارد می کنم', 'بانک را می دانم، رتبه را نمی دانم', 'هنوز بانک مشخص نیست'], *missing)

    score_topic = _has(r'رتبه|امتیاز|سطح ریسک', norm)
    personal_score = _has(r'امتیاز(?:ی)?\s*(?:من|ما|که گرفتم)|رتبه(?: اعتباری)?\s*(?:من|ما|ام|مان)|رتبهام|سطح ریسک\s*(?:من|ما)|کد رتبه', norm)
    low_score = _has(r'افت کرده|پایینه|پایین است|بد شده', norm)
    interpretation = _has(r'تفسیر|کد رتبه|سطح ریسک|\bخوب\b|\bبد\b|کدام گروه|چه گروه|یعنی|چیست', norm)
    causal_income = 'درآمد' in norm and _has(r'چرا|تاثیر|تأثیر|ارتباط', norm)
    if score_topic and (personal_score and interpretation or low_score) and not score_supplied and not causal_income:
        return _request('کد رتبه یا مقدار امتیاز موردنظر را وارد کنید تا بتوان آن را تفسیر کرد.',
                        ['کد رتبه را وارد می کنم', 'امتیاز عددی را وارد می کنم', 'مقدار دقیق را نمی دانم'], 'rank')

    # A deictic phrase is only ambiguous when no event/product is named.
    # "این چک برگشتی" is complete; "این مورد در گزارش" is not.
    unresolved_reference = _has(r'(?:این|آن|همین|همون|اون)\s+(?:مورد|موضوع|وضعیت|شرایط|مشکل|اتفاق|امتیاز)|حذفش|رفعش|اصلاحش|اثر آن|اثرش', norm)
    explicit_topic = debt_source or _has(r'ضامن|ضمانت|اقساط|تاخیر|تأخیر|تسویه|پرداخت|تعطیل|تولید|آدرس|نشانی|صادرات|واردات|سودآور|اسنادی|سفته', norm)
    if unresolved_reference and not explicit_topic and not score_supplied:
        return _request('کدام مورد یا وضعیت را می گویید؟ متن آن بخش از گزارش یا شرح اتفاق را بنویسید.',
                        ['متن مورد گزارش را وارد می کنم', 'اتفاق موردنظر را توضیح می دهم', 'اطلاعات دقیق در دسترس نیست'], 'context')
    return None
