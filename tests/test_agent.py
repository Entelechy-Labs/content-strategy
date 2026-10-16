"""Agent-level tests: normal cases, edge cases, and malformed inputs."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent_content_strategy.agent import (
    ContentStrategyAgent,
    build_pillars,
    build_prompt,
    build_strategy,
    normalize_payload,
    safe_frequency,
    validate_input,
)
from agent_content_strategy.config import CONTENT_STRATEGY_SLUG
from agent_framework.errors import AgentInputError, AgentOutputError
from linkgen_contracts.errors.envelope import ErrorCategory
from linkgen_contracts.research.models import OpportunityMap, ResearchRequest
from linkgen_contracts.strategy.models import (
    AudienceProfile,
    BrandProfile,
    ContentPillar,
    ContentStrategy,
)
from linkgen_contracts.workflow.models import TaskStatus

from conftest import (
    AUDIENCE_ID,
    BRAND_ID,
    RUN_ID,
    TASK_ID,
    make_audience,
    make_brand,
    make_input,
    make_opportunity_map,
    make_request,
)

FIXED_AT = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)

CANDIDATES = [
    {
        "name": "Practical playbooks",
        "description": "Step-by-step onboarding teardowns.",
        "sample_angles": ["checklist audit", "day-1 setup"],
        "suggested_frequency": 3,
    },
    {
        "name": "practical playbooks",  # duplicate
        "description": "Duplicate — dropped by dedupe.",
    },
    {"name": "", "description": "dropped — no name"},
    {"name": "no description", "description": "  "},
]


class TestInputValidation:
    def test_valid_input_round_trips(self) -> None:
        request, brand, audiences, opportunity_map = validate_input(make_input())
        assert isinstance(request, ResearchRequest)
        assert isinstance(brand, BrandProfile)
        assert str(brand.brand_id) == BRAND_ID
        assert len(audiences) == 1 and isinstance(audiences[0], AudienceProfile)
        assert isinstance(opportunity_map, OpportunityMap)
        assert opportunity_map.top_opportunity

    def test_minimal_input_accepted(self) -> None:
        request, brand, audiences, opportunity_map = validate_input(
            {"request": make_request()}
        )
        assert brand is None and audiences == [] and opportunity_map is None

    def test_non_dict_input_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(None)

    def test_missing_request_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input({"brand_profile": make_brand()})

    def test_empty_focus_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(make_input(request=make_request(focus="   ")))

    def test_invalid_brand_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(make_input(brand_profile={"mission": ""}))

    def test_invalid_audience_list_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(make_input(audience_profiles="junk"))

    def test_invalid_opportunity_map_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(make_input(opportunity_map={"opportunity_id": "bad"}))


class TestNormalRun:
    def test_process_produces_contract_valid_output(
        self, agent: ContentStrategyAgent, run_id, task_id
    ) -> None:
        output = agent.process(
            run_id=run_id, task_id=task_id, input_data=make_input()
        )
        assert set(output) == {"strategy", "pillars"}
        strategy = ContentStrategy.model_validate(output["strategy"])
        pillars = [ContentPillar.model_validate(item) for item in output["pillars"]]
        assert strategy.run_id == run_id
        assert strategy.pillars == pillars
        assert pillars  # never an empty strategy (baseline fallback)
        # References point at the artifacts supplied in the input.
        assert str(strategy.brand_profile_id) == BRAND_ID
        assert [str(a) for a in strategy.audience_profile_ids] == [AUDIENCE_ID]
        assert strategy.goals  # model goals, else request objectives
        assert strategy.guardrails  # model guardrails, else spec defaults
        assert 0.0 <= strategy.confidence <= 1.0
        for pillar in pillars:
            assert str(pillar.pillar_id).startswith("pillar_")
            assert pillar.name and pillar.description
            assert [str(a) for a in pillar.audience_ids] == [AUDIENCE_ID]
            if pillar.suggested_frequency is not None:
                assert 1 <= pillar.suggested_frequency <= 30

    def test_minimal_input_still_yields_valid_strategy(
        self, agent: ContentStrategyAgent, run_id, task_id
    ) -> None:
        output = agent.process(
            run_id=run_id,
            task_id=task_id,
            input_data={"request": make_request()},
        )
        strategy = ContentStrategy.model_validate(output["strategy"])
        assert strategy.pillars
        assert strategy.audience_profile_ids == []
        assert str(strategy.brand_profile_id).startswith("profile_")
        assert strategy.guardrails  # spec-derived defaults

    def test_run_reports_success(
        self, agent: ContentStrategyAgent, run_id, task_id
    ) -> None:
        result = agent.run(run_id=run_id, task_id=task_id, input_data=make_input())
        assert result.status is TaskStatus.SUCCEEDED
        assert result.errors == []

    def test_config_declares_expected_boundaries(
        self, agent: ContentStrategyAgent
    ) -> None:
        assert agent.config.slug == CONTENT_STRATEGY_SLUG
        assert "publish" in agent.config.forbidden_tools
        assert "schedule" in agent.config.forbidden_tools
        assert agent.config.response_format.value == "json"
        assert set(agent.config.output_contracts) == {
            "ContentStrategy",
            "ContentPillar",
        }


class TestRunFailures:
    def test_malformed_input_maps_to_failed_result(
        self, agent: ContentStrategyAgent, run_id, task_id
    ) -> None:
        result = agent.run(run_id=run_id, task_id=TASK_ID, input_data="junk")
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].category is ErrorCategory.VALIDATION

    def test_missing_request_maps_to_failed_result(
        self, agent: ContentStrategyAgent, run_id, task_id
    ) -> None:
        result = agent.run(run_id=run_id, task_id=task_id, input_data={})
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].code == "agent_input_invalid"


class TestArtifactBuilders:
    def test_normalize_payload_shape(self) -> None:
        payload = normalize_payload({"pillars": [{"name": "a"}, "junk"]})
        assert payload["pillars"] == [{"name": "a"}]
        assert normalize_payload({})["pillars"] == []
        with pytest.raises(AgentOutputError):
            normalize_payload({"pillars": "nope"})

    def test_dedupe_keeps_first_occurrence_and_drops_unusable(self) -> None:
        from agent_content_strategy.agent import dedupe_pillars

        kept = dedupe_pillars(CANDIDATES)
        assert len(kept) == 1
        assert kept[0]["name"] == "Practical playbooks"

    def test_safe_frequency_bounds(self) -> None:
        assert safe_frequency(3) == 3
        assert safe_frequency(0) is None
        assert safe_frequency(31) is None
        assert safe_frequency(10**9) is None  # mock integers are huge
        assert safe_frequency(True) is None
        assert safe_frequency("3") is None
        assert safe_frequency(None) is None

    def test_build_pillars_maps_fields(self) -> None:
        from agent_content_strategy.agent import dedupe_pillars

        audiences = [AudienceProfile.model_validate(make_audience())]
        pillars = build_pillars(
            run_id=RUN_ID,
            candidates=dedupe_pillars(CANDIDATES),
            audiences=audiences,
        )
        assert len(pillars) == 1
        pillar = pillars[0]
        assert pillar.name == "Practical playbooks"
        assert pillar.sample_angles == ["checklist audit", "day-1 setup"]
        assert pillar.suggested_frequency == 3
        assert [str(a) for a in pillar.audience_ids] == [AUDIENCE_ID]

    def test_build_strategy_fallbacks(self) -> None:
        request = ResearchRequest.model_validate(make_request())
        audiences = [AudienceProfile.model_validate(make_audience())]
        brand = BrandProfile.model_validate(make_brand())
        strategy = build_strategy(
            run_id=RUN_ID,
            request=request,
            brand=brand,
            audiences=audiences,
            payload={"pillars": []},
            generated_at=FIXED_AT,
        )
        # No usable pillars → focus-derived baseline pillar.
        assert len(strategy.pillars) == 1
        assert request.focus in strategy.pillars[0].name
        # No model goals → request objectives (real input, not invention).
        assert strategy.goals == request.objectives
        # No model guardrails → spec-derived defaults.
        assert strategy.guardrails
        assert any("Never invent statistics" in g for g in strategy.guardrails)
        assert str(strategy.brand_profile_id) == BRAND_ID

    def test_prompt_contains_focus_and_ground_rules(self) -> None:
        request = ResearchRequest.model_validate(make_request())
        brand = BrandProfile.model_validate(make_brand())
        opportunity_map = OpportunityMap.model_validate(make_opportunity_map())
        prompt = build_prompt(
            request, brand, [AudienceProfile.model_validate(make_audience())], opportunity_map
        )
        assert request.focus in prompt
        assert "Brand: Jane Doe" in prompt
        assert "Top opportunity: Onboarding checklists that stall" in prompt
        assert "never invent statistics" in prompt
        assert "'messaging_priorities'" in prompt
