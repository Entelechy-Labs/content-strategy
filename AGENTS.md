# AGENTS.md — content-strategy (child repository)

## Owns
This repository owns its implementation, local tests, local documentation,
local configuration defaults, and local version history.

It is responsible for: building the strategic content system — the
contract-conforming `ContentStrategy` (`ContentPillar[]`, messaging
priorities, goals, guardrails) that the `STRATEGY_COMPLETE` gate requires
(Phase 09).

## Does NOT own
- Composition, orchestration, global configuration, system-wide documentation
  (the parent repository owns these).
- System-wide contracts and schemas (established in Phase 01).
- Implementation of other child repositories.

## Dependency direction (allowed)
Depends on explicit shared layers only:

- `linkgen_contracts` (contract layer)
- `linkgen_config` (settings/modes/cost policy)
- `agent-framework` (AgentBase/AgentContext/AgentConfig — ADR-013 shared
  framework for all agents)
- `providers/llm` (ModelRouter only — agents never import provider internals
  or adapters directly)

MUST NOT import parent implementation code (e.g. linkgen.system, "../..", or
any upward filesystem import).
MUST NOT import sibling agent implementation code (e.g. agent_brand_positioning).
No circular dependencies.

## Determinism expectations
- Ids derived deterministically (uuid5 via `agent_framework.ids`); never
  `uuid4` in agent logic.
- `strategy_id` derives from (run, request); pillar ids from (run, normalized
  pillar name); the brand reference from the supplied brand, else (run, focus).
- Time only via the injected context clock; never the system clock.
- No network, randomness, or commercial SDK imports in source.
- Model output is untrusted: sanitized field-by-field before it may enter a
  contract model; unusable pillars are dropped, duplicates deduped, lists and
  frequencies bounded.

## Testing expectations
- Run local unit tests with: python -m pytest
- Never weaken tests; never fake a passing result.
- Add regression tests for every bug fixed.

## Security expectations
- Never commit keys, tokens, cookies, credentials, browser sessions, or
  personal data.
- Use environment variables or a supported secret mechanism.
- Never log secrets.

## Cost constraints
- Mock-first development; $0 external spend is the default.
- No provider may be called accidentally; paid paths fail closed.

## Publishing authority
No publishing/scheduling authority; publishing privilege belongs to the publishing subsystem (publisher) only.
