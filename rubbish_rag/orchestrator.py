"""Shared grounded pipeline for the Web Studio and terminal chat.

The invoke interface accepts dictionaries or strings without requiring LangChain.
It uses the same registered stages as the benchmark, avoiding two divergent
implementations of normalization and resolution.
"""
from typing import Any
from .execution import execute


class RubbishRAGOrchestrator:
    def __init__(self, api_key: str = "demo-key"):
        self.api_key = api_key

    def run(self, query: str, topk: int = 5) -> dict[str, Any]:
        hits, result = execute(query, topk=topk, api_key=self.api_key)
        return {**result, "hits": hits}

    def invoke(self, input_data: Any) -> dict[str, Any]:
        if isinstance(input_data, dict):
            query = input_data.get("query") or input_data.get("question") or ""
        else:
            query = str(input_data)
        return self.run(query)


def get_orchestrator(api_key: str = "demo-key") -> RubbishRAGOrchestrator:
    return RubbishRAGOrchestrator(api_key)
