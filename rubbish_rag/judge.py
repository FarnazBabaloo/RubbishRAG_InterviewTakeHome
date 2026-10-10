"""CPU decision judge: calibrated routing, bounded context, verified evidence.

Runtime reads only fitted calibration parameters, never evaluation questions or
labels. Evidence probabilities are heuristic support estimates, not calibrated
probabilities of semantic truth. Conversation state is currently not resolved.
"""
import dataclasses
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from .normalize_fa import normalize_fa, content_tokens
from .slots import extract_slots

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / 'rubbish_rag/judge_calibration.json'
ACTIONS = ('no_retrieval', 'clarify', 'retrieve')


@dataclasses.dataclass
class PreRetrievalDecision:
    action: str
    needs_retrieval: bool
    needs_clarification: bool
    missing_slots: list[str]
    retrieval_strategy: str
    query_rewrite: str
    target_entities: list[str]
    target_facets: list[str]
    top_k: int
    confidence: float
    retrieve_probability: float
    clarify_probability: float
    reason_codes: list[str]

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class EvidenceDecision:
    sufficiency: str
    evidence_score: float
    conflicting_items: list[int]
    poison_detected: bool
    repetition_detected: bool
    novel_evidence_chunks: int
    next_action: str
    confidence: float
    sufficiency_probability: float
    reason_codes: list[str]

    def to_dict(self):
        return dataclasses.asdict(self)


class RAGDecisionJudge(Protocol):
    def pre_retrieval(self, query: str, conversation=None, slots=None) -> PreRetrievalDecision: ...
    def post_retrieval(self, query: str, hits: list[dict[str, Any]], state=None) -> EvidenceDecision: ...


def clarification_request(query: str):
    """Reuse core guards and ask for missing details of personal events."""
    from .clarification import clarification, _request
    norm = normalize_fa(query)
    # A causal question already names the factor being investigated.
    if 'درآمد' in norm and re.search(r'چرا|تاثیر|تأثیر|ارتباط', norm):
        return None
    if re.search(r'بدون اعتبارسنجی', norm) and re.search(r'تضمین|100', norm):
        return None
    request = clarification(query)
    if request:
        return request
    norm = normalize_fa(query)
    s = extract_slots(query)
    short = len(norm.split()) <= 9
    if short and re.search(r'قسط|اقساط', norm) and re.search(r'عقب|تاخیر|تأخیر', norm) and not s['TIME']:
        return _request('چند روز از سررسید گذشته و تسهیلات متعلق به کدام بانک است؟', ['مدت و بانک را وارد می کنم', 'زمان دقیق را نمی دانم'], 'delay_days', *([] if s['BANK'] else ['bank']))
    if short and 'شرکت' in norm and re.search(r'امتیاز.*(?:ندار|نداره)', norm) and not re.search(r'سود|سابقه|سال|ماه|تاسیس|تأسیس', norm):
        return _request('شرکت چه مدت فعالیت و سابقه تسهیلات دارد؟', ['تازه تأسیس است', 'سابقه فعالیت و تسهیلات دارد', 'سابقه تسهیلات ندارد'], 'company_age')
    if short and 'سابقه' in norm and re.search(r'خراب|بد|مشکل', norm) and not s['DEBT']:
        return _request('کدام سابقه یا بدهی را می گویید؟', ['چک برگشتی', 'بدهی مالیاتی', 'تسهیلات معوق', 'سایر موارد'], 'debt_type')
    if short and 'رتبه' in norm and 'پایین' in norm and not s['RANK'] and not s['NUMS']:
        return _request('رتبه یا امتیاز فعلی و رویداد اخیر را مشخص کنید.', ['کد رتبه را وارد می کنم', 'امتیاز را وارد می کنم', 'اطلاعات دقیق ندارم'], 'rank')
    if short and re.search(r'وام|تسهیلات', norm) and re.search(r'می\s*خواهم|میخوام|می\s*خواهیم', norm):
        missing = ([] if s['BANK'] else ['bank']) + ([] if s['RANK'] else ['rank'])
        if missing:
            return _request('نام بانک و رتبه فعلی را مشخص کنید.', ['اطلاعات را وارد می کنم', 'رتبه را نمی دانم'], *missing)
    if short and 'چک' in norm and re.search(r'چیکار|چه کار|چکار', norm) and not s['TIME']:
        return _request('از برگشت چک چه مدت گذشته و آیا رفع سوءاثر شده است؟', ['هنوز رفع سوءاثر نشده', 'رفع سوءاثر شده', 'زمان دقیق را نمی دانم'], 'time_elapsed', 'check_status')
    if short and 'ضامن' in norm and re.search(r'شدم|شده ام', norm) and not re.search(r'تاخیر|تأخیر|قسط|اقساط|پرداخت', norm):
        return _request('وضعیت بازپرداخت وامی که ضمانت کرده اید چیست؟', ['به موقع پرداخت می شود', 'تأخیر دارد', 'اطلاع ندارم'], 'guarantee_status')
    if short and 'مالیات' in norm and re.search(r'ندادم|نپرداختم', norm) and not re.search(r'وصول|اجرا|مرحله', norm):
        return _request('بدهی مالیاتی در کدام مرحله است؟', ['اعلام بدهی', 'وصول و اجرا', 'مرحله را نمی دانم'], 'tax_stage')
    if short and 'وام' in norm and 'گرفتم' in norm and 'تغییر نکرده' in norm and not s['TIME']:
        return _request('چه زمانی وام گرفته اید و وضعیت پرداخت اقساط چگونه است؟', ['تازه دریافت کرده ام', 'اقساط را پرداخت کرده ام', 'اقساط تأخیر دارد'], 'time_elapsed')
    return None


def route_probabilities(decision):
    return [max(0.0, 1-decision.retrieve_probability-decision.clarify_probability),
            decision.clarify_probability, decision.retrieve_probability]


def confidence_group(query, action):
    """Fixed observable groups; no question IDs, answers or dataset lookup."""
    norm = normalize_fa(query)
    domain = bool(re.search(r'رتبه|امتیاز|بانک|وام|تسهیلات|چک|مالیات|بدهی|ضامن|شرکت|اعتبار|قسط|اقساط', norm))
    return action + (':domain' if domain else ':unanchored')


class HeuristicFeatureJudge:
    """Typed policy with optional temperature fitted outside the runtime."""
    def __init__(self, confidence_threshold=0.75, temperature=1.0, calibration=None):
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.calibration = calibration

    def pre_retrieval(self, query, conversation=None, slots=None):
        norm = normalize_fa(query)
        s = slots or extract_slots(query)
        entities = [s[k] for k in ('RANK','BANK','PERSONA') if s.get(k)]
        ranks = sorted(set(re.findall(r'\b[A-Ea-e][1-3]\b', norm.upper())))
        entities = sorted(set(entities+ranks))
        domain = re.search(r'رتبه|امتیاز|بانک|وام|تسهیلات|چک|مالیات|بدهی|ضامن|شرکت', norm)
        conversational = not domain and bool(re.search(r'^(سلام|درود|ممنون|مرسی|تشکر|خداحافظ|وقت.*بخیر|صبح.*بخیر|شب.*بخیر)|چطور کار می|چه سوالاتی|چه کاری می|چه کمکی|راهنمایی کن|کمک(?:م|مان)?\s*(?:کنی|کنید|کن)', norm))
        request = None if conversational else clarification_request(query)
        reasons = []
        if conversational:
            action, k = 'no_retrieval', 0
            reasons.append('CONVERSATION_ONLY')
        elif request:
            action, k = 'clarify', 0
            reasons.append('MISSING_INFORMATION')
        else:
            action = 'retrieve'
            if len(norm.split()) >= 35:
                k, reason = 8, 'LONG_MULTI_FACET'
            elif len(entities) >= 2 or re.search(r'تفاوت|مقایسه|عوامل|درصد', norm):
                k, reason = 5, 'MULTI_ENTITY_OR_COMPARISON'
            elif len(norm.split()) <= 9:
                k, reason = 2, 'FOCUSED_QUERY'
            else:
                k, reason = 3, 'STANDARD_QUERY'
            reasons.append(reason)
        raw = [0.03, 0.03, 0.03]
        raw[ACTIONS.index(action)] = 0.94
        scaled = [p**(1/self.temperature) for p in raw]
        probs = [p/sum(scaled) for p in scaled]
        if self.calibration:
            group = confidence_group(query, action)
            probs = self.calibration['groups'].get(group, self.calibration['fallback'])
            reasons.append('EMPIRICAL_ROUTE_CONFIDENCE')
        confidence = probs[ACTIONS.index(action)]
        if action=='retrieve' and confidence < self.confidence_threshold:
            k = max(k,5)
            reasons.append('LOW_ROUTE_CONFIDENCE_EXPAND_CONTEXT')
        return PreRetrievalDecision(action, action=='retrieve', action=='clarify',
                                    request['missing_slots'] if request else [],
                                    'hybrid' if action=='retrieve' else 'none', norm, entities,
                                    ranks or (['credit_regulations'] if domain else ['conversation']),
                                    k, confidence, probs[2], probs[1], reasons)

    def post_retrieval(self, query, hits, state=None):
        from .pipeline import corpus, grounded, rerank
        state = state or {}
        norm = normalize_fa(query)
        source_hits = state.get('raw_hits', hits)
        invalid = [h.get('doc_id', -1) for h in source_hits if not grounded(h, corpus().get(h.get('doc_id')))]
        repeated = any(max(Counter(content_tokens(h.get('text',''))).values(), default=0)>=8 for h in source_hits)
        verified = rerank(query, hits)
        relevant = [h for h in verified if h.get('lexical_score',0)>0]
        score = max((h['lexical_score'] for h in relevant), default=0.0)
        ranks = set(re.findall(r'\b[A-E][1-3]\b', norm.upper()))
        covered = set()
        for hit in relevant:
            covered.update(re.findall(r'\b[A-E][1-3]\b', normalize_fa(corpus()[hit['doc_id']].get('Question','')).upper()))
        conflict_ids = []
        for h in source_hits:
            d = corpus().get(h.get('doc_id'))
            if d and grounded(h,d):
                candidate_rank = extract_slots(d.get('Question',''))['RANK']
                if ranks and candidate_rank and candidate_rank not in ranks:
                    conflict_ids.append(h['doc_id'])
        novel = len({h['doc_id'] for h in relevant}-set(state.get('seen_doc_ids',[])))
        spam = bool(re.search(r'رتبه\s+a\s+برای همه|وام فوری بدون ضامن و بدون اعتبارسنجی', norm, re.I) or ('بدون اعتبارسنجی' in norm and re.search(r'تضمین|100', norm)))
        reasons = []
        if invalid: reasons.append('UNVERIFIED_EVIDENCE_FILTERED')
        if repeated: reasons.append('REPETITION_DETECTED')
        if spam:
            suff, action, prob = 'irrelevant','abstain',0.02
            reasons.append('SPAM_QUERY')
        elif not relevant:
            suff = 'conflicting' if conflict_ids else 'insufficient'
            action, prob = ('abstain' if conflict_ids else 'retrieve_more'),0.05
            reasons.append('NO_RELEVANT_VERIFIED_EVIDENCE')
        elif len(ranks)>1 and not ranks.issubset(covered):
            suff, action, prob = 'insufficient','retrieve_more',0.25
            reasons.append('MISSING_COMPARISON_FACET')
        elif score >= 0.30 or (ranks and ranks.issubset(covered)):
            suff, action, prob = 'sufficient','answer',min(0.96,0.80+0.16*score)
            reasons.append('VERIFIED_RELEVANT_SUPPORT')
        else:
            suff, action, prob = 'insufficient','retrieve_more',0.3
            reasons.append('WEAK_LEXICAL_SUPPORT')
        if action=='retrieve_more' and (state.get('round',1)>=state.get('max_rounds',2) or (state.get('round',1)>1 and novel==0)):
            action='abstain'
            reasons.append('ROUND_LIMIT_OR_NO_NEW_EVIDENCE')
        return EvidenceDecision(suff, score, sorted(set(conflict_ids)), bool(invalid), repeated,
                                novel, action, max(prob,1-prob), prob, reasons)


def get_judge():
    config = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {}
    return HeuristicFeatureJudge(temperature=config.get('temperature',1.0),
                                 calibration=config if config.get('method')=='dirichlet_route_groups' else None)


class OpenWeightLLMJudge:
    """Compatibility adapter; delegates to CPU judge, no model weights loaded."""
    def __init__(self, model_name='local-unloaded', fallback=None):
        self.model_name, self.fallback = model_name, fallback or get_judge()
    def pre_retrieval(self, query, conversation=None, slots=None):
        return self.fallback.pre_retrieval(query, conversation, slots)
    def post_retrieval(self, query, hits, state=None):
        return self.fallback.post_retrieval(query, hits, state)


def adaptive_select_hits(query, ranked_hits, judge, max_k=8, min_k=2, state=None):
    from .pipeline import rerank
    safe = rerank(query,ranked_hits)[:max_k]
    selected=[]
    decision=judge.post_retrieval(query,[],state)
    for hit in safe:
        selected.append(hit)
        if len(selected)>=min(min_k,len(safe)):
            decision=judge.post_retrieval(query,selected,state)
            if decision.next_action=='answer' and decision.confidence>=judge.confidence_threshold:
                break
    return selected,decision


def _validate(probabilities, outcomes):
    if not probabilities or len(probabilities)!=len(outcomes):
        raise ValueError('Nonempty, equally sized probabilities and outcomes required')
    if any(not math.isfinite(p) or not 0<=p<=1 for p in probabilities) or any(y not in (0,1) for y in outcomes):
        raise ValueError('Probabilities must be in [0,1], outcomes binary')


def compute_brier_score(probabilities, outcomes):
    _validate(probabilities,outcomes)
    return sum((p-y)**2 for p,y in zip(probabilities,outcomes))/len(outcomes)


def compute_ece(probabilities, outcomes, n_bins=10):
    _validate(probabilities,outcomes)
    if n_bins<1: raise ValueError('n_bins must be positive')
    buckets=[[] for _ in range(n_bins)]
    for p,y in zip(probabilities,outcomes): buckets[min(int(p*n_bins),n_bins-1)].append((p,y))
    return sum(abs(sum(p-y for p,y in b))/len(outcomes) for b in buckets)


def expected_route(item):
    # Dataset labels describe eventual knowledge need; operational clarification
    # takes priority and deliberately performs no retrieval at this turn.
    if item.get('expected_clarification')=='CLARIFY': return 'clarify'
    return 'no_retrieval' if item.get('expected_retrieval')=='NO_RETRIEVAL' else 'retrieve'


def evaluate_judge(dataset_path=None, judge=None, split=None):
    judge=judge or get_judge()
    data=json.loads(Path(dataset_path or ROOT/'data/judge_dataset.json').read_text())
    data=[d for d in data if not split or d.get('split')==split]
    if not data: return {'error':'No records for requested split'}
    predictions=[judge.pre_retrieval(d['query']) for d in data]
    actual=[expected_route(d) for d in data]
    tp=sum(p.action=='clarify' and y=='clarify' for p,y in zip(predictions,actual))
    fp=sum(p.action=='clarify' and y!='clarify' for p,y in zip(predictions,actual))
    fn=sum(p.action!='clarify' and y=='clarify' for p,y in zip(predictions,actual))
    probabilities=[p.retrieve_probability for p in predictions]; outcomes=[int(y=='retrieve') for y in actual]
    conf=[p.confidence for p in predictions]; correct=[int(p.action==y) for p,y in zip(predictions,actual)]
    mean_k=sum(p.top_k for p in predictions)/len(data)
    return {'total_queries':len(data),'split':split or 'all','routing_accuracy':sum(correct)/len(data),
            'clarification':{'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn),'tp':tp,'fp':fp,'fn':fn},
            'sufficiency_accuracy':None,'sufficiency_note':'Not measured by this routing-only dataset; no fabricated hits used.',
            'context_budget':{'mean_top_k':mean_k,'baseline_fixed_k':5,'budget_reduction_pct':100*(1-mean_k/5)},
            'calibration':{'brier_score':compute_brier_score(probabilities,outcomes),'expected_calibration_error':compute_ece(probabilities,outcomes),
                           'decision_brier':compute_brier_score(conf,correct),'decision_ece':compute_ece(conf,correct),
                           'clarify_brier':compute_brier_score([p.clarify_probability for p in predictions],[int(y=='clarify') for y in actual])},
            'results':[{'id':d['id'],'expected':y,'prediction':p.to_dict(),'correct':bool(c)} for d,y,p,c in zip(data,actual,predictions,correct)]}
