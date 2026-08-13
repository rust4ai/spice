//! End-to-end demo of the 0.2 eval features, fully offline (no API key):
//!
//!   1. LLM-as-judge      — `.expect_judge(...)` graded by a `MockJudge`
//!   2. Dataset fan-out   — one set of assertions mapped over N rows
//!   3. Scores            — per-test + suite score in the report
//!   4. Cost/latency      — `Usage` on the agent output, aggregated
//!   5. Baseline diffing  — second run diffed against the first
//!   6. Consensus (pass^k)— one test run N times, distribution recorded
//!
//! Run: `cargo run --example eval_dataset`

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use spice_framework::agent::{AgentConfig, AgentOutput, AgentUnderTest, ToolCall, Turn, Usage};
use spice_framework::dataset::{Dataset, DatasetRow};
use spice_framework::error::SpiceError;
use spice_framework::{suite, MockJudge, Runner, RunnerConfig, TestCase};

/// A toy "support agent": echoes a helpful answer and calls `lookup` for
/// anything that looks like a question. Reports fake token usage so the metrics
/// path has something to aggregate.
struct SupportAgent;

#[async_trait]
impl AgentUnderTest for SupportAgent {
    async fn run(&self, msg: &str, _c: &AgentConfig) -> Result<AgentOutput, SpiceError> {
        let is_question = msg.contains('?');
        let tool_calls = if is_question {
            vec![ToolCall {
                id: "1".into(),
                name: "lookup".into(),
                arguments: serde_json::json!({ "query": msg }),
            }]
        } else {
            vec![]
        };
        let final_text = if is_question {
            format!("Here is the answer to your question about {msg} Refund processed.")
        } else {
            "Thanks for reaching out!".to_string()
        };
        let turn = Turn {
            index: 0,
            output_text: Some(final_text.clone()),
            tool_calls: tool_calls.clone(),
            tool_results: vec![],
            stop_reason: Some("stop".into()),
            duration: Duration::from_millis(40),
        };
        Ok(AgentOutput {
            final_text,
            turns: vec![turn],
            tools_called: tool_calls.iter().map(|t| t.name.clone()).collect(),
            duration: Duration::from_millis(40),
            error: None,
            usage: Some(Usage::tokens(120, 45).with_cost(0.0002)),
        })
    }

    fn available_tools(&self, _c: &AgentConfig) -> Vec<String> {
        vec!["lookup".into()]
    }

    fn name(&self) -> &str {
        "support-agent"
    }
}

fn dataset() -> Dataset {
    // (2) Dataset — normally `Dataset::from_jsonl("cases.jsonl")`; inline here.
    Dataset::new(vec![
        DatasetRow {
            input: "Where is my refund?".into(),
            id: Some("refund".into()),
            expected: serde_json::json!({ "rubric": "The answer mentions a refund." }),
            tags: vec!["support".into()],
        },
        DatasetRow {
            input: "How do I reset my password?".into(),
            id: Some("password".into()),
            expected: serde_json::json!({ "rubric": "The answer is helpful." }),
            tags: vec!["support".into()],
        },
        DatasetRow::new("Thanks!"),
    ])
}

/// Rebuild the full test vec (TestCase isn't Clone, so we regenerate per run).
fn build_tests() -> Vec<TestCase> {
    // (1)+(2)+(3): map each dataset row → assertions + a judged rubric.
    let mut tests = dataset().map_tests(|row, b| {
        let b = b.name(format!("case: {}", row.input));
        if row.input.contains('?') {
            b.expect_tools(&["lookup"]).expect_no_error().expect_judge(
                row.expected_str("rubric")
                    .unwrap_or("The answer is correct."),
            )
        } else {
            b.expect_no_tools().expect_no_error()
        }
    });

    // (6) one consensus test: run 5×, require 4 to pass (records pass^k).
    tests.push(
        spice_framework::test("stable-greeting", "Thanks!")
            .name("greeting is stable (consensus 4/5)")
            .expect_no_tools()
            .consensus(5, 4)
            .build(),
    );
    tests
}

fn make_judge() -> Arc<MockJudge> {
    // Grades `.expect_judge(...)`: needs "answer", must not contain "error".
    Arc::new(MockJudge::new().require(&["answer"]).forbid(&["error"]))
}

#[tokio::main]
async fn main() {
    let report_path = std::env::temp_dir().join("spice-eval-report.json");

    // ---- RUN 1: writes the baseline ----
    println!("\n===== RUN 1 (writing baseline) =====");
    let first = Runner::new(RunnerConfig {
        report_path: Some(report_path.clone()),
        ..Default::default()
    })
    .with_judge(make_judge())
    .run(suite("Support Eval", build_tests()), Arc::new(SupportAgent))
    .await;

    // (4) metrics live on the report:
    println!(
        "aggregate score={:.2}  pass_rate={:.0}%  tokens={:?}  cost=${:.4}  p95={:.0}ms",
        first.score,
        first.pass_rate * 100.0,
        first.metrics.total_usage.total_tokens,
        first.metrics.total_usage.cost_usd.unwrap_or(0.0),
        first.metrics.latency_p95_ms
    );

    // ---- RUN 2: diffed against the baseline we just wrote ----
    println!("\n===== RUN 2 (diff vs baseline) =====");
    Runner::new(RunnerConfig {
        baseline_path: Some(report_path.clone()),
        ..Default::default()
    })
    .with_judge(make_judge())
    .run(suite("Support Eval", build_tests()), Arc::new(SupportAgent))
    .await;
}
