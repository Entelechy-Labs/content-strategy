"""Determinism and id-stability regression tests (golden path).

The same input, clock, and router must produce identical artifacts; different
run ids must produce different derived ids (no cross-run bleed).
"""

from __future__ import annotations

from agent_content_strategy.agent import ContentStrategyAgent
from agent_framework.ids import deterministic_id
from linkgen_contracts.common.ids import RunId, StrategyId, TaskId

from conftest import RUN_ID, TASK_ID, make_context, make_input, make_request


def _run_agent(payload: dict, *, run_id: str = RUN_ID) -> dict:
    agent = ContentStrategyAgent(context=make_context())
    return agent.process(
        run_id=RunId(run_id), task_id=TaskId(TASK_ID), input_data=payload
    )


def test_same_input_same_output() -> None:
    first = _run_agent(make_input())
    second = _run_agent(make_input())
    assert first == second


def test_strategy_id_is_derived_deterministically() -> None:
    from conftest import REQUEST_ID

    output = _run_agent(make_input())
    expected = deterministic_id(StrategyId, RUN_ID, "strategy", REQUEST_ID)
    assert output["strategy"]["strategy_id"] == expected


def test_different_run_id_yields_different_ids() -> None:
    other_run = "run_" + "e4" * 16
    base = _run_agent(make_input())
    other = _run_agent(make_input(run_id=other_run), run_id=other_run)
    assert base["strategy"]["strategy_id"] != other["strategy"]["strategy_id"]
    base_pillars = {p["pillar_id"] for p in base["pillars"]}
    other_pillars = {p["pillar_id"] for p in other["pillars"]}
    if base_pillars and other_pillars:
        assert base_pillars != other_pillars


def test_strategy_id_depends_only_on_run_and_request() -> None:
    # strategy_id derives from (run_id, request_id) — not from the focus or
    # the model's pillar text — so a re-run under the same run/request is
    # idempotent at the artifact level.
    first = _run_agent(make_input())
    second = _run_agent(make_input(request=make_request(focus="a new focus")))
    assert (
        first["strategy"]["strategy_id"] == second["strategy"]["strategy_id"]
    )
