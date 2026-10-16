# content-strategy

Strategy-phase agent of the **LinkGen** content-intelligence and publishing
system.

## Purpose

The specialist agent responsible for building the strategic content system:
it consumes the phase artifacts (`ResearchRequest` plus optional
`BrandProfile`, `AudienceProfile[]`, and `OpportunityMap`) and produces the
contract-conforming `ContentStrategy` — its `ContentPillar[]` plus messaging
priorities, goals, and guardrails. This is the artifact the workflow's
`STRATEGY_COMPLETE` gate requires.

Design highlights (see `AGENTS.md` in the parent for the master rules):

- **Deterministic by construction** — `strategy_id` derives from
  (run, request) and pillar ids from (run, normalized name) via uuid5; time
  comes from the injected context clock; outputs are pure functions of input.
- **LLM output is untrusted** — model JSON arrives router-validated and is
  sanitized field-by-field; pillars need name + description, duplicates are
  deduped, lists are capped, and `suggested_frequency` is dropped unless it
  is a plausible 1..30.
- **Input-derived fallbacks, never invention** — empty model output yields a
  focus-derived baseline pillar, the request's objectives as goals, and the
  spec-derived originality/honesty guardrails.
- **Grounded prompt** — brand, audience segments, and opportunity angles flow
  into the prompt; the 30-day authority arc (days 1–7 attention, 8–14
  credibility, 15–21 engagement/trust, 22–30 commercial intent) is stated as
  the sequencing rule.
- **Mock-first ($0 development)** — the default provider is the offline
  deterministic `MockLLMProvider` via the `ModelRouter`.
- **No privileged tools** — no publishing/scheduling authority.

## Input / output contract

Input (validated against phase contracts):

```json
{"request": {"request_id": "...", "run_id": "...", "focus": "..."},
 "brand_profile": {...optional BrandProfile...},
 "audience_profiles": [{...AudienceProfile...}],
 "opportunity_map": {...optional OpportunityMap...}}
```

`request` is required; everything else is optional.

Output:

```json
{"strategy": {...ContentStrategy...},
 "pillars": [{...ContentPillar...}]}
```

## Boundaries

- Output contracts: `ContentStrategy`, `ContentPillar`.
- Forbidden tools: `publish`, `schedule`, `linkedin-post`.
- Depends only on `linkgen_contracts`, `linkgen_config`, `agent-framework`,
  and the `ModelRouter` interface from `providers/llm`.

## Testing

Run local tests with:

    python -m pytest

## Status

Phase 09 (strategy system) — implemented at 1.0.0.

## Publishing authority

No publishing/scheduling authority; publishing privilege belongs to the
publishing subsystem (publisher) only.
