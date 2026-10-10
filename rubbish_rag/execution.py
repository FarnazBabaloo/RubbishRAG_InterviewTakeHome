"""Shared bounded adaptive execution with per-round, provenance-aware traces."""
import json
import time
import uuid
from pathlib import Path
from typing import Any

import rubbish_rag as R
from . import pipeline
from .judge import get_judge, clarification_request, adaptive_select_hits
from .normalize_fa import normalize_fa, tokens
from .slots import extract_slots

TRACE = Path(__file__).resolve().parents[1] / 'traces/trace.jsonl'


def execute(query: str, topk: int = 5, api_key: str = 'demo-key', *, adaptive: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use at most two retrieval calls; topk caps selected answer context."""
    if not 1 <= topk <= 20:
        raise ValueError('topk must be between 1 and 20')
    started = time.perf_counter()
    trace_id = 'client_' + uuid.uuid4().hex
    raw_hits, ranked, selected, rounds = [], [], [], []
    calls=0
    judge=get_judge() if adaptive else None
    pre=judge.pre_retrieval(query) if adaptive else None
    if not adaptive:
        request=pipeline.clarification(query)
        if request:
            prediction=request
        else:
            calls=1
            raw_hits=R.get('retriever')(query,topk=topk,api_key=api_key)
            ranked=R.get('reranker')(query,raw_hits)
            selected=ranked[:topk]
            prediction=R.get('resolver')(query,selected)
    elif pre.action=='no_retrieval':
        prediction={'status':'answer','brief':'سلام! می توانید درباره رتبه اعتباری، گزارش، چک و تسهیلات سؤال کنید.', 'cites':[], 'decision':'no_retrieval'}
    elif pre.action=='clarify':
        prediction=clarification_request(query)
    else:
        context_k=min(topk,pre.top_k)
        seen=set()
        for round_number in (1,2):
            request_k=context_k if round_number==1 else min(20,max(10,context_k*2))
            calls+=1
            incoming=R.get('retriever')(query,topk=request_k,api_key=api_key)
            # Preserve distinct excerpts/provenance, while avoiding repeated pool entries.
            merged={(h['doc_id'],h.get('chunk_id'),h.get('source'),h.get('text')):h for h in raw_hits+incoming}
            raw_hits=list(merged.values())
            ranked=R.get('reranker')(query,raw_hits)
            state={'raw_hits':raw_hits,'round':round_number,'max_rounds':2,'seen_doc_ids':sorted(seen)}
            selected,post=adaptive_select_hits(query,ranked,judge,max_k=context_k,min_k=min(2,context_k),state=state)
            rounds.append({'round':round_number,'request_topk':request_k,'context_limit':context_k,
                           'raw_hits':incoming,'selected_doc_ids':[h['doc_id'] for h in selected],
                           'decision':post.to_dict()})
            if post.next_action=='answer':
                prediction=R.get('resolver')(query,selected)
                break
            if post.next_action!='retrieve_more':
                break
            seen.update(h['doc_id'] for h in ranked)
            context_k=min(topk,8)
        if not rounds or rounds[-1]['decision']['next_action']!='answer':
            prediction={'status':'clarify','decision':'abstain',
                        'question':'شواهد کافی و سازگار پیدا نشد؛ موضوع و جزئیات پرسش را دقیق تر مشخص کنید.',
                        'options':['موضوع را دقیق تر توضیح می دهم','متن گزارش را وارد می کنم']}
    entry={'trace_id':trace_id,'event':'pipeline','ts':time.time(),'key':api_key,
           'q':query,'normalized_query':normalize_fa(query),'slots':extract_slots(query),
           'topk':topk,'hits':raw_hits,'ranked_hits':ranked,'selected_hits':selected,
           'remote_hits':[h for h in raw_hits if h.get('source')!='local_corpus'],
           'local_hits':[h for h in raw_hits if h.get('source')=='local_corpus'],
           'pipeline_version':'adaptive-staff-v2' if adaptive else 'fixed-comparison',
           'pre_decision':pre.to_dict() if pre else None,'rounds':rounds,
           'context_token_estimate':sum(len(tokens(h.get('text',''))) for h in selected),
           'fixed5_token_estimate_same_pool':sum(len(tokens(h.get('text',''))) for h in ranked[:5]),
           'prediction':prediction,'latency_ms':(time.perf_counter()-started)*1000,'retrieval_calls':calls}
    TRACE.parent.mkdir(exist_ok=True)
    with TRACE.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
    return selected,{**prediction,'trace_id':trace_id}
