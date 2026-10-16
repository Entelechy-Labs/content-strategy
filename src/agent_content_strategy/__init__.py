"""content-strategy — builds the strategic content system (Phase 09).

Strategy-phase agent of the LinkGen content-intelligence system. Consumes the
research/strategy artifacts (``ResearchRequest`` plus optional
``BrandProfile``, ``AudienceProfile[]``, and ``OpportunityMap``) and produces
the contract-conforming ``ContentStrategy`` (``ContentPillar[]``, messaging
priorities, goals, guardrails) through the shared agent framework and
ModelRouter (mock mode by default).
"""

from __future__ import annotations

from agent_content_strategy.config import (
    CONTENT_STRATEGY_OUTPUT_SCHEMA,
    CONTENT_STRATEGY_SLUG,
    build_content_strategy_config,
)
from agent_content_strategy.agent import ContentStrategyAgent
from agent_content_strategy.version import __version__

__all__ = [
    "__version__",
    "CONTENT_STRATEGY_OUTPUT_SCHEMA",
    "CONTENT_STRATEGY_SLUG",
    "ContentStrategyAgent",
    "build_content_strategy_config",
]
