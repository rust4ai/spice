//! Tests for the 0.2 eval features against the in-box `MockAgent` / `MockJudge`.

use std::sync::Arc;

use spice_framework::dataset::{Dataset, DatasetRow};
use spice_framework::report::TestReport;
use spice_framework::{
    suite, test, MockAgent, MockJudge, MockResponse, Runner, RunnerConfig, ToolCall,
};

fn quiet_config() -> RunnerConfig {
    RunnerConfig {
        console_output: false,
        ..Default::default()
    }
}

fn tool(name: &str) -> ToolCall {
    ToolCall {
        id: "t".into(),
        name: name.into(),
        arguments: serde_json::json!({}),
    }
}

#[tokio::test]
async fn judge_gates_pass_and_records_score() {
    let agent = MockAgent::new("a")
        .on("q", MockResponse::text("The answer is 42."))
        .with_tools(vec![]);

    // Judge requires "answer" — output contains it → passes.
    let good = Runner::new(quiet_config())
        .with_judge(Arc::new(MockJudge::new().require(&["answer"])))
        .run(
            suite(
                "j",
                vec![test("t", "q")
                    .expect_judge("must contain the answer")
                    .build()],
            ),
            Arc::new(agent),
        )
        .await;
    assert!(good.tests[0].passed);
    assert_eq!(good.tests[0].judge_results.len(), 1);
    assert!(good.tests[0].judge_results[0].score >= 0.7);
    assert_eq!(good.tests[0].score, 1.0);
}

#[tokio::test]
async fn judge_below_threshold_fails() {
    let agent = MockAgent::new("a").on("q", MockResponse::text("nope"));
    let report = Runner::new(quiet_config())
        .with_judge(Arc::new(MockJudge::new().require(&["answer", "refund"])))
        .run(
            suite("j", vec![test("t", "q").expect_judge("needs both").build()]),
            Arc::new(agent),
        )
        .await;
    assert!(!report.tests[0].passed);
    // 0 of 2 keywords present → score 0.
    assert_eq!(report.tests[0].judge_results[0].score, 0.0);
}

#[tokio::test]
async fn judge_assertion_fails_without_a_judge() {
    let agent = MockAgent::new("a").on("q", MockResponse::text("The answer is 42."));
    let report = Runner::new(quiet_config())
        // no .with_judge(...)
        .run(
            suite("j", vec![test("t", "q").expect_judge("anything").build()]),
            Arc::new(agent),
        )
        .await;
    assert!(!report.tests[0].passed);
    assert!(report.tests[0].judge_results[0]
        .reason
        .contains("no judge configured"));
}

#[tokio::test]
async fn dataset_fans_out_and_aggregates() {
    let data = Dataset::new(vec![
        DatasetRow::new("hello"),
        DatasetRow::new("hi"),
        DatasetRow::new("hey"),
    ]);
    let tests = data.map_tests(|_row, b| b.expect_no_tools());
    assert_eq!(tests.len(), 3);

    let agent = MockAgent::new("a").default_response(MockResponse::text("hi there"));
    let report = Runner::new(quiet_config())
        .run(suite("d", tests), Arc::new(agent))
        .await;
    assert_eq!(report.total, 3);
    assert_eq!(report.passed, 3);
    assert_eq!(report.pass_rate, 1.0);
    assert_eq!(report.score, 1.0);
}

#[test]
fn dataset_loads_jsonl() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("spice-ds-{}.jsonl", std::process::id()));
    std::fs::write(
        &path,
        "{\"input\":\"a\",\"id\":\"one\"}\n\n{\"input\":\"b\",\"expected\":{\"rubric\":\"r\"}}\n",
    )
    .unwrap();
    let data = Dataset::from_jsonl(&path).unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!(data.rows[0].id.as_deref(), Some("one"));
    assert_eq!(data.rows[1].expected_str("rubric"), Some("r"));
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn exact_tools_rejects_extra_and_missing() {
    let agent = MockAgent::new("a")
        .on(
            "both",
            MockResponse::with_tools("ok", vec![tool("read"), tool("write")]),
        )
        .on("one", MockResponse::with_tools("ok", vec![tool("read")]));

    let report = Runner::new(quiet_config())
        .run(
            suite(
                "x",
                vec![
                    // exactly {read, write} → both present, passes.
                    test("ok", "both")
                        .expect_exact_tools(&["read", "write"])
                        .build(),
                    // expected {read, write} but only read called → fails (missing).
                    test("missing", "one")
                        .expect_exact_tools(&["read", "write"])
                        .build(),
                    // expected {read} but read+write called → fails (extra).
                    test("extra", "both").expect_exact_tools(&["read"]).build(),
                ],
            ),
            Arc::new(agent),
        )
        .await;
    let by_id = |id: &str| report.tests.iter().find(|t| t.test_id == id).unwrap();
    assert!(by_id("ok").passed);
    assert!(!by_id("missing").passed);
    assert!(!by_id("extra").passed);
}

#[tokio::test]
async fn consensus_records_full_distribution() {
    // Deterministic agent → all 5 runs pass; consensus stats reflect 5/5.
    let agent = MockAgent::new("a").on("q", MockResponse::text("hi"));
    let report = Runner::new(quiet_config())
        .run(
            suite(
                "c",
                vec![test("t", "q").expect_no_tools().consensus(5, 4).build()],
            ),
            Arc::new(agent),
        )
        .await;
    let c = report.tests[0].consensus.as_ref().unwrap();
    assert_eq!(c.runs, 5);
    assert_eq!(c.passed, 5);
    assert_eq!(c.required, 4);
    assert_eq!(c.pass_rate(), 1.0);
    assert_eq!(report.tests[0].attempts, 5);
}

#[test]
fn baseline_diff_detects_regression_and_fix() {
    fn report(name: &str, cases: &[(&str, bool)]) -> spice_framework::SuiteReport {
        let tests: Vec<TestReport> = cases
            .iter()
            .map(|(id, passed)| TestReport {
                test_id: (*id).into(),
                test_name: None,
                tags: vec![],
                passed: *passed,
                attempts: 1,
                assertion_results: vec![],
                judge_results: vec![],
                score: if *passed { 1.0 } else { 0.0 },
                consensus: None,
                usage: None,
                run_duration: None,
                duration: std::time::Duration::ZERO,
                error: None,
            })
            .collect();
        let total = tests.len();
        spice_framework::SuiteReport::new(
            name.into(),
            tests,
            total,
            std::time::Duration::ZERO,
            chrono::Utc::now(),
        )
    }

    let baseline = report("base", &[("a", true), ("b", false), ("removed", true)]);
    let current = report("now", &[("a", false), ("b", true), ("added", true)]);
    let diff = current.diff_against(&baseline);

    assert_eq!(diff.regressions, vec!["a".to_string()]);
    assert_eq!(diff.newly_passing, vec!["b".to_string()]);
    assert_eq!(diff.added, vec!["added".to_string()]);
    assert_eq!(diff.removed, vec!["removed".to_string()]);
    assert!(diff.has_regressions());
}

#[test]
fn metrics_aggregate_tokens_and_latency() {
    use spice_framework::Usage;
    let mk = |ms: u64, tok: u64| TestReport {
        test_id: "t".into(),
        test_name: None,
        tags: vec![],
        passed: true,
        attempts: 1,
        assertion_results: vec![],
        judge_results: vec![],
        score: 1.0,
        consensus: None,
        usage: Some(Usage::tokens(tok, 0)),
        run_duration: Some(std::time::Duration::from_millis(ms)),
        duration: std::time::Duration::from_millis(ms),
        error: None,
    };
    let tests = vec![mk(10, 100), mk(20, 200), mk(30, 300)];
    let report = spice_framework::SuiteReport::new(
        "m".into(),
        tests,
        3,
        std::time::Duration::ZERO,
        chrono::Utc::now(),
    );
    assert_eq!(report.metrics.total_usage.total_tokens, Some(600));
    assert_eq!(report.metrics.timed_runs, 3);
    assert_eq!(report.metrics.latency_p50_ms, 20.0);
    assert!(report.metrics.latency_mean_ms > 19.0 && report.metrics.latency_mean_ms < 21.0);
}
