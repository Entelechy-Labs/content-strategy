"""Shared fixtures for content-strategy tests.

Deterministic by construction: fixed clock, mock-only settings, and fixed id
constants — no system clock, no randomness, no network.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]

# Sibling repos on sys.path (mirrors parent compose-time resolution).
# This repo lives at agents/content-strategy → agents → parent root.
for _child in (
    "../../contracts/src",
    "../../config/src",
    "../../providers/llm/src",
    "../../agent-framework/src",
):
    _src = (_ROOT / _child).resolve()
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

import pytest

from agent_content_strategy.agent import ContentStrategyAgent
from agent_framework.context import AgentContext
from linkgen_contracts.common.ids import RunId, TaskId
from linkgen_config.settings import (
    CostPolicy,
    ModelRoleConfig,
    ProviderInfo,
    ProviderKind,
    Settings,
)

FIXED_NOW = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)

# Fixed contract ids (32 lowercase hex suffixes) for golden determinism.
RUN_ID = "run_" + "a1" * 16
TASK_ID = "task_" + "b2" * 16
REQUEST_ID = "request_" + "c3" * 16
BRAND_ID = "profile_" + "aa" * 16
AUDIENCE_ID = "profile_" + "f6" * 16
OPPORTUNITY_ID = "result_" + "12" * 16


class RecordingSleeper:
    """Records backoff sleeps; never actually sleeps."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = dict(overrides)
    base.setdefault(
        "providers",
        {
            "mock": ProviderInfo(
                id="mock",
                kind=ProviderKind.LLM,
                paid=False,
                description="test mock",
            )
        },
    )
    base.setdefault(
        "model_roles",
        {
            role: ModelRoleConfig(provider="mock", model=f"dev_{role}_model")
            for role in ("reasoning", "writer", "fast", "vision", "review")
        },
    )
    base.setdefault("cost", CostPolicy())
    return Settings(**base)


def make_context(**overrides: Any) -> AgentContext:
    """Build an AgentContext with mock router, fixed clock, injected sleeper."""
    from provider_llm.router import ModelRouter

    settings = overrides.get("settings") or make_settings()
    router = overrides.get("router") or ModelRouter(settings)
    return AgentContext(
        settings=settings,
        router=router,
        clock=lambda: FIXED_NOW,
        sleeper=overrides.get("sleeper"),
    )


def make_request(**overrides: Any) -> dict[str, Any]:
    """A valid ResearchRequest payload (as a plain dict)."""
    data: dict[str, Any] = {
        "request_id": REQUEST_ID,
        "run_id": RUN_ID,
        "focus": "mid-market SaaS teams struggling with onboarding",
        "objectives": ["build authority", "win trust"],
        "creators": ["jane-doe"],
        "domains": ["linkedin.com"],
    }
    data.update(overrides)
    return data


def make_brand() -> dict[str, Any]:
    """A valid BrandProfile payload (as a plain dict)."""
    return {
        "brand_id": BRAND_ID,
        "name": "Jane Doe",
        "mission": "Help ops leads ship onboarding that works.",
        "values": ["specificity"],
        "tone": "direct and practical",
        "voice_guidelines": ["no hype"],
        "topics_we_own": ["onboarding"],
        "topics_to_avoid": ["vanity metrics"],
        "captured_at": FIXED_NOW.isoformat(),
    }


def make_audience() -> dict[str, Any]:
    """A valid AudienceProfile payload (as a plain dict)."""
    return {
        "profile_id": AUDIENCE_ID,
        "run_id": RUN_ID,
        "name": "Ops leads at mid-market SaaS",
        "description": "Own onboarding rollout; measured on time-to-value.",
        "demographics": {"role": "Operations lead"},
        "interests": ["activation"],
        "needs": ["fit-for-purpose checklists"],
        "preferred_formats": ["carousel"],
        "confidence": 0.7,
        "captured_at": FIXED_NOW.isoformat(),
    }


def make_opportunity_map() -> dict[str, Any]:
    """A valid OpportunityMap payload (as a plain dict)."""
    return {
        "opportunity_id": OPPORTUNITY_ID,
        "run_id": RUN_ID,
        "generated_at": FIXED_NOW.isoformat(),
        "signals": [],
        "opportunities": [
            {
                "angle_id": "angle_" + "34" * 16,
                "title": "Onboarding checklists that stall",
                "description": "Teams copy templates without fitting them.",
                "audience_match": 0.77,
                "novelty": 0.61,
            }
        ],
        "top_opportunity": "Onboarding checklists that stall",
        "confidence": 0.6,
    }


def make_input(**overrides: Any) -> dict[str, Any]:
    """A valid content-strategy input payload."""
    data: dict[str, Any] = {
        "request": make_request(),
        "brand_profile": make_brand(),
        "audience_profiles": [make_audience()],
        "opportunity_map": make_opportunity_map(),
    }
    data.update(overrides)
    return data


@pytest.fixture
def run_id() -> RunId:
    return RunId(RUN_ID)


@pytest.fixture
def task_id() -> TaskId:
    return TaskId(TASK_ID)


@pytest.fixture
def context() -> AgentContext:
    return make_context()


@pytest.fixture
def agent(context: AgentContext) -> ContentStrategyAgent:
    return ContentStrategyAgent(context=context)
