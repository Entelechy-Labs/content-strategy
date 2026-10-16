"""Typed definition of the content-strategy agent.

The ``AgentConfig`` is the definition-time contract of this agent: mission,
model role, allowed/forbidden tools, output contracts, and the JSON schema
its LLM calls must satisfy. The router validates model output against this
schema (fail closed) before the agent ever sees it.
"""

from __future__ import annotations

from typing import Any

from agent_framework.config import AgentConfig
from linkgen_contracts.llm.models import LLMRole, ResponseFormat

CONTENT_STRATEGY_SLUG = "content-strategy"

# JSON schema for the model's structured output. Restricted to the router's
# documented schema subset (type/properties/items/required/enum/const) so it
# fails closed rather than being silently ignored. Ids and timestamps are
# NEVER requested from the model — deterministic Python derives them.
CONTENT_STRATEGY_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pillars": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "sample_angles": {"type": "array", "items": {"type": "string"}},
                    "suggested_frequency": {"type": "integer"},
                },
            },
        },
        "messaging_priorities": {"type": "array", "items": {"type": "string"}},
        "goals": {"type": "array", "items": {"type": "string"}},
        "guardrails": {"type": "array", "items": {"type": "string"}},
    },
}

_SYSTEM_PROMPT = """You are the content-strategy agent of the LinkGen content system.
Mission: build the strategic content system for the requested focus — the
pillars, messaging priorities, goals, and guardrails that later phases
(calendar, writing, review) must follow.
Use varied pillars that progressively build authority across the 30-day
arc: days 1-7 attention/expertise, days 8-14 credibility, days 15-21
engagement/trust, days 22-30 commercial intent without excessive promotion.
Ground every pillar in the supplied research, opportunity map, brand, and
audience artifacts. Do not copy creator wording or distinctive material, and
never invent statistics, case studies, or personal results.
Respond only with JSON matching the requested schema."""


def build_content_strategy_config() -> AgentConfig:
    """Build the definition-time configuration of the strategy agent."""
    return AgentConfig(
        slug=CONTENT_STRATEGY_SLUG,
        model_role=LLMRole.REASONING,
        output_contracts=[
            "ContentStrategy",
            "ContentPillar",
        ],
        system_prompt=_SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=3072,
        max_steps=8,
        tools=["web-search"],
        forbidden_tools=["publish", "schedule", "linkedin-post"],
        response_format=ResponseFormat.JSON,
        json_schema=CONTENT_STRATEGY_OUTPUT_SCHEMA,
        description="Build the strategic content system (strategy phase).",
        version="1.0.0",
    )
