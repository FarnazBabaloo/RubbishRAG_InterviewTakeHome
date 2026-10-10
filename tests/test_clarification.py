"""Missing-slot guards and nearby complete questions; no API calls."""
import pytest
from rubbish_rag.clarification import clarification


@pytest.mark.parametrize('query,missing', [
    ('اگر بدهی را قسط بندی کنیم چه می شود؟', ['debt_type']),
    ('بانک ملی به من وام می دهد؟', ['rank']),
    ('رتبه D1 دارم و وام بگیرم؟', ['bank']),
    ('می توانم وام بگیرم؟', ['bank', 'rank']),
    ('کد رتبه من را تفسیر کن', ['rank']),
    ('این مشکل توی گزارش یعنی چی؟', ['context']),
    ('حذفش کنم چه اثری دارد؟', ['context']),
])
def test_requests_exactly_missing_information(query, missing):
    result = clarification(query)
    assert result['missing_slots'] == missing
    assert result['question'] and len(result['options']) >= 2


@pytest.mark.parametrize('query', [
    'بانک ملی با رتبه B3 به من وام می دهد؟',
    'بانک ملت با امتیاز ۶۲۰ به من وام می دهد؟',
    'اگر بدهی مالیاتی را قسط بندی کنم چه می شود؟',
    'اگر اقساط وام پرداخت نشود چه اثری دارد؟',
    'این چک برگشتی چه اثری بر گزارش من دارد؟',
    'اثر آن بدهی بانکی بر اعتبار چیست؟',
    'تعطیلی تولید شرکت در گزارش ثبت می شود؟',
    'سطح ریسک رتبه A1 چیست؟',
    'پرداخت بدهی مالیاتی امتیاز من را افزایش می دهد؟',
    'امتیاز من بعد از پرداخت اقساط چرا افزایش پیدا نکرده؟',
])
def test_complete_topics_are_not_blocked(query):
    assert clarification(query) is None


def test_does_not_request_an_already_supplied_rank():
    result = clarification('با رتبه C3 می توانم وام بگیرم؟')
    assert result['missing_slots'] == ['bank']
    assert 'رتبه' not in result['question']


def test_ambiguous_request_stops_before_retrieval(monkeypatch, tmp_path):
    import rubbish_rag as R
    from rubbish_rag import execution
    monkeypatch.setattr(execution, 'TRACE', tmp_path/'trace.jsonl')
    def forbidden(*args, **kwargs):
        pytest.fail('Missing information must be requested before retrieval')
    monkeypatch.setitem(R._registry, 'retriever', forbidden)
    _, result = execution.execute('این وضعیت در گزارش من چیست؟')
    assert result['missing_slots'] == ['context']
