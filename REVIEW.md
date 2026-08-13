# Spice — Design Review

*A review of `spice-framework` (as of v0.1.0), the trajectory-first Rust test
harness for nondeterministic LLM agents, with a prioritized roadmap to v0.2.*

## TL;DR

Spice is an **excellent trace-assertion library** with the right core instinct:
it asserts on an agent's *trajectory* (which tools it called, with what
arguments, in what order, across how many turns) rather than only on final text.
Security/RBAC is a first-class concept, nondeterminism is handled deliberately
(retries + consensus), and the whole thing is `cargo test`-native and
provider-agnostic. For Rust agents it is effectively best-in-class, because the
entire agent-eval ecosystem is otherwise Python.

Its gaps versus "industry standard" harnesses (promptfoo, DeepEval, Inspect,
LangSmith/Braintrust, tau-bench) are all in **one dimension: statistical and
semantic scoring** — no LLM-as-judge, no dataset fan-out, boolean-only results,
no cost/token metrics, and no baseline/regression tracking. Close those and it
graduates from "trace-assertion crate" to "eval framework."

## What's genuinely good

- **The right primitive.** Trajectory-first from the ground up
  (`tools_called`, per-`Turn` `tool_calls`/`tool_results`, arg matching,
  ordering, turn ranges). This is where agent correctness actually lives.
- **Security/RBAC as a first-class citizen.** `expect_tools_within_allowlist`,
  `is_security` tagging, and an `RbacMatrix` that *auto-generates* per-role
  allowlist + injection tests. Most generic harnesses make you build this.
- **Nondeterminism handled deliberately.** Retries (pass-if-any) *and* M-of-N
  consensus are both built in.
- **CI-native and provider-agnostic.** One trait (`AgentUnderTest`), runs in
  `cargo test`, parallel runner, JSON reports + per-run traces.

## Gaps vs. industry standard

1. **No LLM-as-judge / model-graded assertions** — the biggest gap. The only
   text check is substring `expect_text_contains`. Serious harnesses grade
   free-form output semantically against a rubric.
2. **No datasets / parametrization.** No way to run the same assertions over N
   rows of a JSONL/CSV dataset and report accuracy over a distribution.
3. **Everything is boolean.** No scores (0–1), no partial credit, no
   suite-level score.
4. **No cost/token/latency metrics.** `duration` is captured per turn but
   tokens and cost are discarded.
5. **No baseline / regression tracking.** Reports are written to disk but never
   diffed against a prior run.
6. **"Multi-turn" is agent turns, not a conversation.** `run` takes a single
   `user_message`; there is no simulated user driving follow-ups.
7. **State/environment assertions aren't a primitive.** Tool calls are asserted
   as a *proxy* for correct action; grading end-state of the world requires a
   raw closure.

## Bugs / rough edges found in v0.1

- **`ExpectGatheringBeforeAction` copy-paste bug**: `last_gather` is computed
  from `action_strs` instead of the gather tools (dead code, suppressed, but
  misleading). The "neither gather nor action called" case is also treated as a
  failure rather than vacuously true.
- **`retries` semantics are lenient**: a retry passes if *any* attempt passes
  (pass@N used as a gate), which inflates pass rates and masks flakiness.
  Consensus is the honest control; the distinction should be loud.
- **Consensus discards its own distribution**: it early-breaks at `required` and
  reports `attempts: runs` regardless — throwing away the pass^k signal it just
  paid to compute.
- **Two sources of truth for tool calls**: `ExpectTools` reads the flat
  `tools_called` while count/order read per-turn `turns[].tool_calls`; the
  adapter populates them separately, so they can diverge.
- **No "exactly these tools" assertion** — `expect_tools` is a subset check.

## Prioritized roadmap → v0.2

Delivered in this order:

1. **`Judge` trait + `.expect_judge(rubric)`** returning `{score, reason}`
   (with `MockJudge` for offline tests and an optional `OpenAiJudge`).
2. **Dataset-driven cases** (`Dataset::from_jsonl`, `map_tests`) applying the
   same assertions across N inputs, reporting an aggregate pass rate.
3. **Scores, not just bools** — per-test and per-suite `score` (0–1), with judge
   thresholds as the gate.
4. **Cost/token/latency metrics** — `Usage` on `AgentOutput`, suite totals +
   p50/p95 latency in the report.
5. **Baseline diffing** — `SuiteReport::load_from_file` + `diff_against` +
   `regressions`, so CI can fail on regression.
6. **pass^k / variance** — consensus runs the full N and records the
   distribution; retries-vs-consensus semantics documented.

Deferred to a later cut (bigger swings): simulated-user multi-turn conversations
and first-class post-run environment/state assertions.

See [CHANGELOG.md](CHANGELOG.md) for what actually shipped in 0.2.
