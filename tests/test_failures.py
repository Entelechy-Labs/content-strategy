"""Failure-path tests: model failures, timeouts, schema violations.

Mock-provider fault markers (RATE_LIMIT, TIMEOUT, NETWORK_FAILURE,
AUTH_FAILURE, INVALID_OUTPUT) make provider failures deterministic; stub
routers cover paths the mock cannot reach.
"""

from __future__ import annotations

from typing import Any

from agent_content_strategy.agent import ContentStrategyAgent
from agent_framework.context import AgentContext
from agent_framework.errors import AgentTimeoutError
from linkgen_contracts.errors.envelope import ErrorCategory
from linkgen_contracts.strategy.models import ContentStrategy
from linkgen_contracts.workflow.models import TaskStatus
from provider_llm.router import ModelRouter

from conftest import RUN_ID, TASK_ID, make_input, make_request, make_settings


class _StubResponse:
    def __init__(self, *, parsed: Any = None, text: Any = None) -> None:
        self.parsed = parsed
        self.text = text


class _StubResult:
    def __init__(self, response: _StubResponse) -> None:
        self.response = response


class StubRouter:
    """Duck-typed router returning canned results or raising canned errors."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def complete_json(self, request, schema):  # noqa: ANN001
        return self.fn(request, schema)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _agent_with_router(router: Any, sleeper=None) -> ContentStrategyAgent:
    context = AgentContext(settings=make_settings(), router=router, sleeper=sleeper)
    return ContentStrategyAgent(context=context)


def _input_with_focus(focus: str) -> dict[str, Any]:
    payload = make_input()
    payload["request"] = make_request(focus=focus)
    return payload


class TestMockProviderFaults:
    def test_rate_limit_exhausts_retries_with_backoff(self) -> None:
        sleeper = _Recorder()
        agent = _agent_with_router(
            ModelRouter(make_settings(), sleeper=sleeper), sleeper=sleeper
        )
        result = agent.run(
            run_id=RUN_ID,
            task_id=TASK_ID,
            input_data=_input_with_focus("RATE_LIMIT storm"),
        )
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].category is ErrorCategory.PROVIDER
        assert result.errors[0].code == "router_exhausted"
        assert len(sleeper.calls) == 2  # max_attempts_per_provider=3

    def test_timeout_marker_fails_after_retries(self) -> None:
        sleeper = _Recorder()
        agent = _agent_with_router(
            ModelRouter(make_settings(), sleeper=sleeper), sleeper=sleeper
        )
        result = agent.run(
            run_id=RUN_ID,
            task_id=TASK_ID,
            input_data=_input_with_focus("TIMEOUT slow research"),
        )
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].code == "router_exhausted"
        assert len(sleeper.calls) == 2

    def test_auth_failure_is_not_retried(self) -> None:
        sleeper = _Recorder()
        agent = _agent_with_router(
            ModelRouter(make_settings(), sleeper=sleeper), sleeper=sleeper
        )
        result = agent.run(
            run_id=RUN_ID,
            task_id=TASK_ID,
            input_data=_input_with_focus("AUTH_FAILURE locked out"),
        )
        assert result.status is TaskStatus.FAILED
        assert sleeper.calls == []  # non-retryable → no backoff sleeps

    def test_invalid_output_marker_fails_closed(self) -> None:
        agent = _agent_with_router(ModelRouter(make_settings()))
        result = agent.run(
            run_id=RUN_ID,
            task_id=TASK_ID,
            input_data=_input_with_focus("INVALID_OUTPUT garbage"),
        )
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].code == "router_exhausted"


class TestStubRouterPaths:
    def test_unparseable_model_json_maps_to_validation_failure(self) -> None:
        agent = _agent_with_router(
            StubRouter(lambda request, schema: _StubResult(_StubResponse(text="not json")))
        )
        result = agent.run(run_id=RUN_ID, task_id=TASK_ID, input_data=make_input())
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].category is ErrorCategory.VALIDATION
        assert result.errors[0].code == "agent_output_invalid"

    def test_non_object_json_maps_to_validation_failure(self) -> None:
        agent = _agent_with_router(
            StubRouter(lambda request, schema: _StubResult(_StubResponse(text="[1,2]")))
        )
        result = agent.run(run_id=RUN_ID, task_id=TASK_ID, input_data=make_input())
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].category is ErrorCategory.VALIDATION

    def test_non_array_pillars_fails_closed(self) -> None:
        agent = _agent_with_router(
            StubRouter(
                lambda request, schema: _StubResult(
                    _StubResponse(parsed={"pillars": "nope"})
                )
            )
        )
        result = agent.run(run_id=RUN_ID, task_id=TASK_ID, input_data=make_input())
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].category is ErrorCategory.VALIDATION

    def test_unexpected_router_exception_maps_to_internal(self) -> None:
        def boom(request, schema):  # noqa: ANN001
            raise RuntimeError("unexpected")

        agent = _agent_with_router(StubRouter(boom))
        result = agent.run(run_id=RUN_ID, task_id=TASK_ID, input_data=make_input())
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].category is ErrorCategory.INTERNAL

    def test_policy_error_category_is_preserved(self) -> None:
        class FakePolicyBlocked(Exception):
            category = ErrorCategory.PERMISSIONS
            code = "policy_blocked"
            retryable = False

            def __init__(self) -> None:
                self.message = "paid provider blocked"
                super().__init__(self.message)

        def deny(request, schema):  # noqa: ANN001
            raise FakePolicyBlocked()

        agent = _agent_with_router(StubRouter(deny))
        result = agent.run(run_id=RUN_ID, task_id=TASK_ID, input_data=make_input())
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].category is ErrorCategory.PERMISSIONS
        assert result.errors[0].code == "policy_blocked"

    def test_agent_errors_pass_through_unchanged(self) -> None:
        def timeout(request, schema):  # noqa: ANN001
            raise AgentTimeoutError("agent budget exhausted")

        agent = _agent_with_router(StubRouter(timeout))
        result = agent.run(run_id=RUN_ID, task_id=TASK_ID, input_data=make_input())
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].category is ErrorCategory.TIMEOUT
        assert result.errors[0].retryable is True

    def test_empty_model_output_yields_baseline_strategy(self) -> None:
        # Fail-soft on content, fail-closed on format: schema-valid but empty
        # model output still produces a usable, input-derived strategy.
        agent = _agent_with_router(
            StubRouter(
                lambda request, schema: _StubResult(
                    _StubResponse(parsed={})
                )
            )
        )
        output = agent.process(
            run_id=RUN_ID, task_id=TASK_ID, input_data=make_input()
        )
        strategy = ContentStrategy.model_validate(output["strategy"])
        assert strategy.pillars
        assert make_input()["request"]["focus"] in strategy.pillars[0].name
        assert strategy.goals  # request objectives
        assert strategy.guardrails  # spec-derived defaults
