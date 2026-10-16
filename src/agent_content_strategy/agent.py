"""Content-strategy agent — Phase 09 strategy system.

Consumes the research/strategy phase artifacts and produces the
contract-conforming ``ContentStrategy`` (its ``ContentPillar[]`` plus
messaging priorities, goals, and guardrails). The strategy gate
(``STRATEGY_COMPLETE``) requires this artifact.

Design rules (shared with the Phase 08 research agents):
- The LLM is used only for content judgment; ids, timestamps, dedupe, capping,
  and confidence bounds are deterministic Python.
- Model output is untrusted: it arrives router-validated as JSON, and every
  field is sanitized before entering a contract model.
- Ids are derived deterministically (uuid5) so identical inputs produce
  identical artifacts (idempotent replay, golden tests, checkpoint recovery).
- No network, clock, or randomness: time comes from the injected context
  clock; failures come from the router as duck-typed-normalized errors.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from linkgen_contracts.common.ids import (
    PillarId,
    ProfileId,
    RunId,
    StrategyId,
    TaskId,
)
from linkgen_contracts.errors.envelope import ErrorCategory
from linkgen_contracts.research.models import OpportunityMap, ResearchRequest
from linkgen_contracts.strategy.models import (
    AudienceProfile,
    BrandProfile,
    ContentPillar,
    ContentStrategy,
)
from pydantic import ValidationError

from agent_content_strategy.config import (
    CONTENT_STRATEGY_OUTPUT_SCHEMA,
    build_content_strategy_config,
)
from agent_framework.base import AgentBase
from agent_framework.config import AgentConfig
from agent_framework.context import AgentContext
from agent_framework.errors import (
    AgentError,
    AgentInputError,
    AgentOutputError,
)
from agent_framework.ids import deterministic_id
from agent_framework.sanitize import (
    bounded_str_list,
    clean_text,
    normalize_key,
    stable_probability,
)
from agent_framework.structured import extract_json_dict

# Bounded resource usage (AGENTS rule 11).
MAX_PILLARS = 8
MAX_ANGLES = 10
MAX_LIST_ITEMS = 10

# Suggested frequency is a per-week/posts-per-week hint: anything outside
# 1..30 is meaningless (and the mock's schema-driven integers are huge), so
# it is dropped rather than propagated.
MAX_SUGGESTED_FREQUENCY = 30

# Spec-derived fallback guardrails (SOURCE_SPECIFICATION: originality and
# honesty rules) — used only when the model proposes none. These come from
# the written specification, not from model invention.
DEFAULT_GUARDRAILS = [
    "Do not copy exact wording, sentences, paragraphs, unique metaphors, "
    "story structures, or distinctive phrases from other creators.",
    "Never invent statistics, fabricate case studies, or claim personal "
    "results that were not supplied.",
]

__all__ = [
    "ContentStrategyAgent",
    "brand_reference_id",
    "build_pillars",
    "build_prompt",
    "build_strategy",
    "dedupe_pillars",
    "normalize_payload",
    "safe_frequency",
    "validate_input",
    "normalize_router_failure",
]


# ---------------------------------------------------------------------------
# Untrusted-input normalization (agent-specific; sanitizers are shared).
# ---------------------------------------------------------------------------

def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the top-level shape of the untrusted model payload."""
    pillars = payload.get("pillars")
    if pillars is None:
        pillars = []
    if not isinstance(pillars, list):
        raise AgentOutputError("model output field 'pillars' must be an array")
    return {
        "pillars": [entry for entry in pillars if isinstance(entry, dict)],
        "messaging_priorities": payload.get("messaging_priorities"),
        "goals": payload.get("goals"),
        "guardrails": payload.get("guardrails"),
    }


def dedupe_pillars(
    candidates: list[dict[str, Any]], *, cap: int = MAX_PILLARS
) -> list[dict[str, Any]]:
    """Dedupe by normalized name; drop unusable entries; cap deterministically.

    A pillar needs both a name and a description to be actionable downstream
    (calendar, writing), so entries missing either are dropped.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for entry in candidates:
        name = clean_text(entry.get("name"))
        description = clean_text(entry.get("description"))
        key = normalize_key(name)
        if not key or not description:
            continue  # not actionable — skip deterministically
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
        if len(out) >= cap:
            break
    return out


def safe_frequency(value: Any) -> int | None:
    """Return a plausible per-period frequency, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if 1 <= value <= MAX_SUGGESTED_FREQUENCY:
        return value
    return None


def _validate_model(raw: Any, model: type, label: str):
    if raw is None:
        return None
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first.get("loc", ()))
        raise AgentInputError(
            f"{label} failed contract validation at '{location}': "
            f"{first.get('msg', 'invalid')}"
        ) from exc


def _validate_model_list(raw: Any, model: type, label: str) -> list[Any]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AgentInputError(f"content-strategy input '{label}' must be a list")
    validated: list[Any] = []
    for index, item in enumerate(raw):
        validated.append(
            _validate_model(item, model, f"{label}[{index}]")
        )
    return validated


def validate_input(
    input_data: Any,
) -> tuple[
    ResearchRequest,
    BrandProfile | None,
    list[AudienceProfile],
    OpportunityMap | None,
]:
    """Validate the agent input against the phase contracts.

    Input shape::

        {"request": ResearchRequest,                # required
         "brand_profile": BrandProfile,              # optional
         "audience_profiles": [AudienceProfile, ...], # optional
         "opportunity_map": OpportunityMap}          # optional
    """
    if not isinstance(input_data, dict):
        raise AgentInputError(
            "content-strategy input must be a dict with a 'request' key"
        )
    raw_request = input_data.get("request")
    if raw_request is None:
        raise AgentInputError("content-strategy input is missing the 'request' object")
    try:
        request = ResearchRequest.model_validate(raw_request)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first.get("loc", ()))
        raise AgentInputError(
            f"research request failed contract validation at '{location}': "
            f"{first.get('msg', 'invalid')}"
        ) from exc

    brand = _validate_model(input_data.get("brand_profile"), BrandProfile, "brand_profile")
    audiences = _validate_model_list(
        input_data.get("audience_profiles"), AudienceProfile, "audience_profiles"
    )
    opportunity_map = _validate_model(
        input_data.get("opportunity_map"), OpportunityMap, "opportunity_map"
    )
    return request, brand, audiences, opportunity_map


class _RouterFailure(AgentError):
    """Agent-level wrapper for a router failure (category/code preserved)."""


def normalize_router_failure(exc: Exception) -> AgentError:
    """Normalize a router failure without importing provider internals.

    Router errors carry ``category``/``code``/``retryable``; they are preserved
    by duck-typing so ``TaskResult`` mapping keeps retry semantics honest.
    """
    failure = _RouterFailure(f"router call failed: {getattr(exc, 'message', None) or exc}")
    raw_category = getattr(exc, "category", ErrorCategory.INTERNAL)
    failure.category = raw_category if isinstance(raw_category, ErrorCategory) else ErrorCategory.INTERNAL
    failure.code = str(getattr(exc, "code", "router_failure"))
    failure.retryable = bool(getattr(exc, "retryable", False))
    return failure


def build_prompt(
    request: ResearchRequest,
    brand: BrandProfile | None,
    audiences: list[AudienceProfile],
    opportunity_map: OpportunityMap | None,
) -> str:
    """Deterministic user prompt for the content-strategy LLM call."""
    lines = [
        "Build the strategic content system for a LinkedIn content system.",
        f"Focus: {request.focus}",
    ]
    if request.objectives:
        lines.append("Objectives: " + "; ".join(request.objectives))
    if brand is not None:
        lines.append(f"Brand: {brand.name} — {brand.mission}")
        lines.append(f"Tone: {brand.tone}")
        if brand.topics_we_own:
            lines.append("Topics we own: " + ", ".join(brand.topics_we_own))
        if brand.topics_to_avoid:
            lines.append("Topics to avoid: " + ", ".join(brand.topics_to_avoid))
    if audiences:
        lines.append("Target audience segments:")
        lines.extend(f"- {a.name}: {a.description}" for a in audiences[:6])
    if opportunity_map is not None:
        if opportunity_map.top_opportunity:
            lines.append(f"Top opportunity: {opportunity_map.top_opportunity}")
        if opportunity_map.opportunities:
            lines.append("Candidate angles:")
            lines.extend(
                f"- {a.title}: {a.description}"
                for a in opportunity_map.opportunities[:10]
            )
    lines.append(
        "Define varied content pillars that progressively build authority "
        "across the 30-day arc (days 1-7 attention/expertise, days 8-14 "
        "credibility, days 15-21 engagement/trust, days 22-30 commercial "
        "intent without excessive promotion). Ground every pillar in the "
        "supplied artifacts; keep messaging priorities, goals, and "
        "guardrails concrete; never invent statistics or results."
    )
    lines.append(
        "Return JSON with 'pillars', 'messaging_priorities', 'goals', "
        "and 'guardrails'."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Artifact builders (pure functions of their inputs).
# ---------------------------------------------------------------------------

def brand_reference_id(run_id: RunId, request: ResearchRequest, brand: BrandProfile | None) -> ProfileId:
    """The strategy's brand reference: the supplied brand, else derived.

    When no ``BrandProfile`` was supplied, the reference id is derived from
    the run and focus (never model text) so the artifact stays deterministic;
    it is replaced by the real brand id as soon as brand artifacts flow in.
    """
    if brand is not None:
        return brand.brand_id
    return ProfileId(
        deterministic_id(
            ProfileId, str(run_id), "brand", normalize_key(request.focus) or "unspecified"
        )
    )


def build_pillars(
    *,
    run_id: RunId,
    candidates: list[dict[str, Any]],
    audiences: list[AudienceProfile],
) -> list[ContentPillar]:
    """Map untrusted, deduped pillar entries into contract-valid pillars.

    ``candidates`` must already be deduped (``dedupe_pillars``). Every pillar
    is attached to the run's audience segments (deterministically).
    """
    audience_ids = [profile.profile_id for profile in audiences]
    pillars: list[ContentPillar] = []
    for entry in candidates:
        name = clean_text(entry.get("name"))
        description = clean_text(entry.get("description"))
        if not name or not description:
            continue
        pillars.append(
            ContentPillar(
                pillar_id=deterministic_id(
                    PillarId, str(run_id), "pillar", normalize_key(name)
                ),
                name=name,
                description=description,
                audience_ids=audience_ids,
                sample_angles=bounded_str_list(
                    entry.get("sample_angles"), limit=MAX_ANGLES
                ),
                suggested_frequency=safe_frequency(entry.get("suggested_frequency")),
            )
        )
    return pillars


def baseline_pillar(
    *, run_id: RunId, request: ResearchRequest, audiences: list[AudienceProfile]
) -> ContentPillar:
    """The pillar derived from the request focus itself.

    Used only when the model produced no usable pillar, so downstream phases
    (calendar, writing) always receive at least one *input-derived* pillar.
    """
    focus = clean_text(request.focus) or "unspecified"
    return ContentPillar(
        pillar_id=deterministic_id(PillarId, str(run_id), "pillar", normalize_key(focus)),
        name=f"Focus: {focus}",
        description=f"Content pillar anchored on the research focus '{focus}'.",
        audience_ids=[profile.profile_id for profile in audiences],
        sample_angles=bounded_str_list(request.objectives),
        suggested_frequency=None,
    )


def build_strategy(
    *,
    run_id: RunId,
    request: ResearchRequest,
    brand: BrandProfile | None,
    audiences: list[AudienceProfile],
    payload: dict[str, Any],
    generated_at: datetime,
) -> ContentStrategy:
    """Build the run's ``ContentStrategy`` from sanitized model payload."""
    pillars = build_pillars(
        run_id=run_id,
        candidates=dedupe_pillars(payload.get("pillars", [])),
        audiences=audiences,
    )
    if not pillars:
        pillars = [baseline_pillar(run_id=run_id, request=request, audiences=audiences)]

    goals = bounded_str_list(payload.get("goals"), limit=MAX_LIST_ITEMS)
    if not goals:
        # Request objectives are real input, not model invention.
        goals = bounded_str_list(request.objectives, limit=MAX_LIST_ITEMS)

    guardrails = bounded_str_list(payload.get("guardrails"), limit=MAX_LIST_ITEMS)
    if not guardrails:
        guardrails = list(DEFAULT_GUARDRAILS)

    return ContentStrategy(
        strategy_id=deterministic_id(
            StrategyId, str(run_id), "strategy", str(request.request_id)
        ),
        run_id=run_id,
        brand_profile_id=brand_reference_id(run_id, request, brand),
        audience_profile_ids=[profile.profile_id for profile in audiences],
        pillars=pillars,
        messaging_priorities=bounded_str_list(
            payload.get("messaging_priorities"), limit=MAX_LIST_ITEMS
        ),
        goals=goals,
        guardrails=guardrails,
        generated_at=generated_at,
        confidence=stable_probability(
            "strategy", str(run_id), str(request.request_id)
        ),
    )


# ---------------------------------------------------------------------------
# The agent.
# ---------------------------------------------------------------------------

class ContentStrategyAgent(AgentBase):
    """Build the strategic content system (strategy phase, Phase 09).

    Input (validated against phase contracts)::

        {"request": ResearchRequest, "brand_profile"?: BrandProfile,
         "audience_profiles"?: [AudienceProfile],
         "opportunity_map"?: OpportunityMap}

    Output::

        {"strategy": ContentStrategy, "pillars": [ContentPillar]}
    """

    def __init__(
        self, context: AgentContext, config: AgentConfig | None = None
    ) -> None:
        super().__init__(
            config if config is not None else build_content_strategy_config(),
            context,
        )

    def process(
        self,
        *,
        run_id: RunId,
        task_id: TaskId,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        request, brand, audiences, opportunity_map = validate_input(input_data)
        payload = normalize_payload(
            self._gather(
                run_id=run_id,
                task_id=task_id,
                request=request,
                brand=brand,
                audiences=audiences,
                opportunity_map=opportunity_map,
            )
        )
        strategy = build_strategy(
            run_id=run_id,
            request=request,
            brand=brand,
            audiences=audiences,
            payload=payload,
            generated_at=self.context.now(),
        )
        return {
            "strategy": strategy.model_dump(mode="json"),
            "pillars": [pillar.model_dump(mode="json") for pillar in strategy.pillars],
        }

    def _gather(
        self,
        *,
        run_id: RunId,
        task_id: TaskId,
        request: ResearchRequest,
        brand: BrandProfile | None,
        audiences: list[AudienceProfile],
        opportunity_map: OpportunityMap | None,
    ) -> dict[str, Any]:
        """Call the router for the strategic content system (single LLM hop).

        Fault markers in the prompt (mock mode) or a failing adapter surface
        as normalized agent errors; they never escape as raw provider errors.
        """
        prompt = build_prompt(request, brand, audiences, opportunity_map)
        llm_request = self.build_llm_request(
            run_id=run_id,
            task_id=task_id,
            prompt=prompt,
            json_schema=CONTENT_STRATEGY_OUTPUT_SCHEMA,
        )
        try:
            result = self.context.router.complete_json(
                llm_request, CONTENT_STRATEGY_OUTPUT_SCHEMA
            )
        except AgentError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalized below, never swallowed
            raise normalize_router_failure(exc) from exc
        return extract_json_dict(result)
