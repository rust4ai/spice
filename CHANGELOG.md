# Changelog

## 0.2.0

The 0.2 release turns Spice from a trace-assertion library into an eval
framework, closing the statistical/semantic-scoring gap called out in
[REVIEW.md](REVIEW.md). Priorities were delivered in review order.

### Added

1. **LLM-as-judge** (`judge` module). New `Judge` trait, `JudgeRequest` /
   `JudgeVerdict`, and builders `TestCaseBuilder::expect_judge(rubric)` /
   `expect_judge_threshold(rubric, t)`. Judge assertions are evaluated by the
   runner via `Runner::with_judge(...)`. Ships `MockJudge` (deterministic,
   offline, keyword-based) and, behind the `openai` feature, `OpenAiJudge`
   (chat-completions, strict-JSON verdict). Judges only run when the
   deterministic assertions already pass, so they never cost a call on an
   already-failing test.
2. **Dataset-driven evals** (`dataset` module). `Dataset::from_jsonl` /
   `from_json`, `DatasetRow`, and `Dataset::map_tests(|row, builder| ...)` to
   fan one set of assertions across N inputs.
3. **Scores, not just pass/fail.** `TestReport::score` and `SuiteReport::score`
   (mean `0.0..=1.0` over assertions + judges), plus `SuiteReport::pass_rate`.
4. **Cost / token / latency metrics.** New `Usage` type on `AgentOutput`;
   `SuiteReport::metrics` (`SuiteMetrics`) aggregates token/cost totals and
   `latency_p50_ms` / `latency_p95_ms` / `latency_mean_ms`.
5. **Baseline diffing.** `SuiteReport::load_from_file`, `diff_against` →
   `SuiteDiff` (`regressions`, `newly_passing`, `added`, `removed`, score/pass-rate
   deltas), `has_regressions()`, and `RunnerConfig::baseline_path` to print the
   diff automatically.
6. **pass^k from consensus.** Consensus mode now runs the full N (no early
   break) and records the distribution in `TestReport::consensus`
   (`ConsensusStats { runs, passed, required }`, `pass_rate()`).
7. **`expect_exact_tools(&[..])`** — assert the exact set of tools called.

### Fixed

- `ExpectGatheringBeforeAction`: removed the dead `last_gather` computed from
  the wrong tool list, and the "neither gathered nor acted" case is now
  vacuously true instead of a failure.
- Tool-presence assertions (`expect_tools`, `forbid_tools`, `expect_no_tools`,
  allowlist, …) now derive tool names from the turn trace via
  `AgentOutput::tool_names()`, giving a single source of truth with the
  per-turn assertions instead of trusting the flat `tools_called` field.
- Consensus reporting no longer claims `attempts: runs` while short-circuiting;
  it reports the true distribution.

### Changed / breaking

- `AgentOutput` gained a `usage: Option<Usage>` field. Struct-literal
  constructions must add `usage: None` (or `..Default::default()` — `AgentOutput`
  now implements `Default`).
- `TestReport` and `SuiteReport` gained fields (all `#[serde(default)]`, so old
  JSON reports still deserialize). `SuiteReport` is now built via
  `SuiteReport::new(...)`, which computes scores and metrics.
- `TestCase` gained a `judges: Vec<JudgeSpec>` field.

## 0.1.0

Initial release: fluent trajectory assertions, concurrent runner, retries +
consensus, RBAC matrix, JSON traces + reports, `MockAgent`.
