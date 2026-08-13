use crate::agent::{AgentConfig, AgentOutput, AgentUnderTest};
use crate::assertion::AssertionResult;
use crate::judge::{Judge, JudgeRequest};
use crate::report::{ConsensusStats, JudgeResult, SuiteReport, TestReport};
use crate::test_case::{JudgeSpec, TestCase, TestSuite};
use crate::trace::Trace;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

/// Configuration for the test runner.
pub struct RunnerConfig {
    /// Max concurrent tests.
    pub concurrency: usize,
    /// Default timeout per test.
    pub default_timeout: Duration,
    /// Filter: only run tests whose id or name contains this substring.
    pub filter: Option<String>,
    /// Filter: only run tests with any of these tags.
    pub tag_filter: Option<Vec<String>>,
    /// Directory to write trace files.
    pub trace_dir: Option<PathBuf>,
    /// Path to write JSON report.
    pub report_path: Option<PathBuf>,
    /// Baseline report to diff against; the diff is printed after the run.
    pub baseline_path: Option<PathBuf>,
    /// Print console output.
    pub console_output: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            concurrency: 4,
            default_timeout: Duration::from_secs(60),
            filter: None,
            tag_filter: None,
            trace_dir: None,
            report_path: None,
            baseline_path: None,
            console_output: true,
        }
    }
}

/// The test runner.
pub struct Runner {
    pub config: RunnerConfig,
    judge: Option<Arc<dyn Judge>>,
}

impl Runner {
    pub fn new(config: RunnerConfig) -> Self {
        Self {
            config,
            judge: None,
        }
    }

    /// Install the judge used to evaluate `.expect_judge(...)` expectations.
    /// Without one, judge assertions fail with "no judge configured".
    pub fn with_judge(mut self, judge: Arc<dyn Judge>) -> Self {
        self.judge = Some(judge);
        self
    }

    /// Run a test suite against an agent, returning the suite report.
    pub async fn run(&self, suite: TestSuite, agent: Arc<dyn AgentUnderTest>) -> SuiteReport {
        let start = Instant::now();
        let semaphore = Arc::new(Semaphore::new(self.config.concurrency));

        let tests: Vec<TestCase> = suite
            .tests
            .into_iter()
            .filter(|t| self.matches_filter(t))
            .collect();

        let total = tests.len();
        let mut handles = Vec::with_capacity(total);

        for test_case in tests {
            let sem = semaphore.clone();
            let agent = agent.clone();
            let judge = self.judge.clone();
            let default_timeout = suite.default_timeout.unwrap_or(self.config.default_timeout);
            let default_retries = suite.default_retries;
            let default_config = suite.default_config.clone();
            let trace_dir = self.config.trace_dir.clone();

            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                run_single_test(
                    test_case,
                    &*agent,
                    judge.as_ref(),
                    default_timeout,
                    default_retries,
                    &default_config,
                    trace_dir.as_ref(),
                )
                .await
            });
            handles.push(handle);
        }

        let mut reports = Vec::with_capacity(total);
        for handle in handles {
            match handle.await {
                Ok(report) => reports.push(report),
                Err(e) => {
                    reports.push(TestReport {
                        test_id: "unknown".into(),
                        test_name: None,
                        tags: vec![],
                        passed: false,
                        attempts: 0,
                        assertion_results: vec![],
                        judge_results: vec![],
                        score: 0.0,
                        consensus: None,
                        usage: None,
                        run_duration: None,
                        duration: Duration::ZERO,
                        error: Some(format!("Task panicked: {}", e)),
                    });
                }
            }
        }

        let suite_report = SuiteReport::new(
            suite.name,
            reports,
            total,
            start.elapsed(),
            chrono::Utc::now(),
        );

        if self.config.console_output {
            suite_report.print_console();
        }

        if let Some(path) = &self.config.report_path {
            if let Err(e) = suite_report.save_to_file(path) {
                eprintln!("Failed to save report: {}", e);
            }
        }

        if let Some(path) = &self.config.baseline_path {
            match SuiteReport::load_from_file(path) {
                Ok(baseline) => {
                    let diff = suite_report.diff_against(&baseline);
                    if self.config.console_output {
                        diff.print_console();
                    }
                }
                Err(e) => {
                    if self.config.console_output {
                        eprintln!("  (no baseline diff: {})", e);
                    }
                }
            }
        }

        suite_report
    }

    fn matches_filter(&self, test: &TestCase) -> bool {
        if let Some(filter) = &self.config.filter {
            let id_match = test.id.contains(filter.as_str());
            let name_match = test
                .name
                .as_ref()
                .map(|n| n.contains(filter.as_str()))
                .unwrap_or(false);
            if !id_match && !name_match {
                return false;
            }
        }
        if let Some(tag_filter) = &self.config.tag_filter {
            if !test.tags.iter().any(|t| tag_filter.contains(t)) {
                return false;
            }
        }
        true
    }
}

/// Evaluate the judge specs against one output. Returns the per-judge results
/// and whether all of them passed (vacuously true when there are no judges).
async fn eval_judges(
    judges: &[JudgeSpec],
    user_message: &str,
    output: &AgentOutput,
    judge: Option<&Arc<dyn Judge>>,
) -> (Vec<JudgeResult>, bool) {
    let mut results = Vec::with_capacity(judges.len());
    let mut all_pass = true;
    for spec in judges {
        let jr = match judge {
            Some(j) => {
                let req = JudgeRequest {
                    rubric: &spec.rubric,
                    user_message,
                    output,
                    threshold: spec.threshold,
                };
                match j.score(req).await {
                    Ok(v) => JudgeResult {
                        rubric: spec.rubric.clone(),
                        score: v.score,
                        threshold: spec.threshold,
                        passed: v.score >= spec.threshold,
                        reason: v.reason,
                    },
                    Err(e) => JudgeResult {
                        rubric: spec.rubric.clone(),
                        score: 0.0,
                        threshold: spec.threshold,
                        passed: false,
                        reason: format!("judge error: {e}"),
                    },
                }
            }
            None => JudgeResult {
                rubric: spec.rubric.clone(),
                score: 0.0,
                threshold: spec.threshold,
                passed: false,
                reason: "no judge configured (Runner::with_judge)".into(),
            },
        };
        if !jr.passed {
            all_pass = false;
        }
        results.push(jr);
    }
    (results, all_pass)
}

/// Assertions + judges for a single agent output.
struct Evaluation {
    assertion_results: Vec<AssertionResult>,
    judge_results: Vec<JudgeResult>,
    passed: bool,
    score: f64,
    usage: Option<crate::agent::Usage>,
    run_duration: Duration,
}

async fn evaluate_output(
    test: &TestCase,
    output: &AgentOutput,
    available_tools: &[String],
    judge: Option<&Arc<dyn Judge>>,
) -> Evaluation {
    let assertion_results: Vec<AssertionResult> = test
        .assertions
        .iter()
        .map(|a| a.evaluate(output, available_tools))
        .collect();
    let assertions_pass = assertion_results.iter().all(|r| r.passed);

    // Only spend judge calls when the deterministic assertions already pass.
    let (judge_results, judges_pass) = if assertions_pass {
        eval_judges(&test.judges, &test.user_message, output, judge).await
    } else {
        (Vec::new(), false)
    };

    let score = TestReport::compute_score(&assertion_results, &judge_results);
    Evaluation {
        assertion_results,
        judge_results,
        passed: assertions_pass && judges_pass,
        score,
        usage: output.usage.clone(),
        run_duration: output.duration,
    }
}

async fn run_single_test(
    test: TestCase,
    agent: &dyn AgentUnderTest,
    judge: Option<&Arc<dyn Judge>>,
    default_timeout: Duration,
    default_retries: usize,
    default_config: &AgentConfig,
    trace_dir: Option<&PathBuf>,
) -> TestReport {
    let start = Instant::now();
    let timeout = test.timeout.unwrap_or(default_timeout);
    let max_retries = test.retries.max(default_retries);
    let config = if test.config.data.is_null() {
        default_config
    } else {
        &test.config
    };
    let available_tools = agent.available_tools(config);

    // Consensus mode.
    if let (Some(runs), Some(required)) = (test.consensus_runs, test.consensus_required) {
        return run_consensus(
            &test,
            agent,
            judge,
            config,
            &available_tools,
            timeout,
            runs,
            required,
            trace_dir,
            start,
        )
        .await;
    }

    // Standard retry mode.
    let mut last_eval: Option<Evaluation> = None;
    let mut last_error = None;
    let mut attempts = 0;

    for attempt in 0..=max_retries {
        attempts = attempt + 1;

        let run_result = tokio::time::timeout(timeout, agent.run(&test.user_message, config)).await;

        match run_result {
            Ok(Ok(output)) => {
                if let Some(dir) = trace_dir {
                    let trace =
                        Trace::new(test.id.clone(), test.user_message.clone(), output.clone());
                    let path = dir.join(format!("{}_attempt{}.json", test.id, attempt));
                    let _ = trace.save_to_file(&path);
                }

                let eval = evaluate_output(&test, &output, &available_tools, judge).await;
                if eval.passed {
                    return TestReport {
                        test_id: test.id,
                        test_name: test.name,
                        tags: test.tags,
                        passed: true,
                        attempts,
                        assertion_results: eval.assertion_results,
                        judge_results: eval.judge_results,
                        score: eval.score,
                        consensus: None,
                        usage: eval.usage,
                        run_duration: Some(eval.run_duration),
                        duration: start.elapsed(),
                        error: None,
                    };
                }
                last_eval = Some(eval);
                last_error = None;
            }
            Ok(Err(e)) => {
                last_error = Some(e.to_string());
                last_eval = None;
            }
            Err(_) => {
                last_error = Some(format!("Timeout after {:?}", timeout));
                last_eval = None;
            }
        }
    }

    let (assertion_results, judge_results, score, usage, run_duration) = match last_eval {
        Some(e) => (
            e.assertion_results,
            e.judge_results,
            e.score,
            e.usage,
            Some(e.run_duration),
        ),
        None => (vec![], vec![], 0.0, None, None),
    };

    TestReport {
        test_id: test.id,
        test_name: test.name,
        tags: test.tags,
        passed: false,
        attempts,
        assertion_results,
        judge_results,
        score,
        consensus: None,
        usage,
        run_duration,
        duration: start.elapsed(),
        error: last_error,
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_consensus(
    test: &TestCase,
    agent: &dyn AgentUnderTest,
    judge: Option<&Arc<dyn Judge>>,
    config: &AgentConfig,
    available_tools: &[String],
    timeout: Duration,
    runs: usize,
    required: usize,
    trace_dir: Option<&PathBuf>,
    start: Instant,
) -> TestReport {
    let mut pass_count = 0;
    let mut last_eval: Option<Evaluation> = None;
    let mut last_error = None;

    // Run the full N — never early-break — so the pass/fail distribution
    // (pass^k) is honest and complete.
    for i in 0..runs {
        let run_result = tokio::time::timeout(timeout, agent.run(&test.user_message, config)).await;

        match run_result {
            Ok(Ok(output)) => {
                if let Some(dir) = trace_dir {
                    let trace =
                        Trace::new(test.id.clone(), test.user_message.clone(), output.clone());
                    let path = dir.join(format!("{}_consensus{}.json", test.id, i));
                    let _ = trace.save_to_file(&path);
                }

                let eval = evaluate_output(test, &output, available_tools, judge).await;
                if eval.passed {
                    pass_count += 1;
                }
                last_eval = Some(eval);
            }
            Ok(Err(e)) => last_error = Some(e.to_string()),
            Err(_) => last_error = Some(format!("Timeout after {:?}", timeout)),
        }
    }

    let passed = pass_count >= required;
    let (assertion_results, judge_results, score, usage, run_duration) = match last_eval {
        Some(e) => (
            e.assertion_results,
            e.judge_results,
            e.score,
            e.usage,
            Some(e.run_duration),
        ),
        None => (vec![], vec![], 0.0, None, None),
    };

    TestReport {
        test_id: test.id.clone(),
        test_name: test.name.clone(),
        tags: test.tags.clone(),
        passed,
        attempts: runs,
        assertion_results,
        judge_results,
        score,
        consensus: Some(ConsensusStats {
            runs,
            passed: pass_count,
            required,
        }),
        usage,
        run_duration,
        duration: start.elapsed(),
        error: if passed {
            None
        } else {
            Some(format!(
                "Consensus: {}/{} passed, needed {}{}",
                pass_count,
                runs,
                required,
                last_error
                    .map(|e| format!(" (last error: {e})"))
                    .unwrap_or_default()
            ))
        },
    }
}
