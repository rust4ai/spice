use crate::agent::Usage;
use crate::assertion::AssertionResult;
use crate::error::SpiceError;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// The outcome of a single model-graded (judge) expectation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeResult {
    pub rubric: String,
    pub score: f64,
    pub threshold: f64,
    pub passed: bool,
    pub reason: String,
}

/// Distribution of pass/fail across consensus runs — the raw material for
/// pass^k reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusStats {
    pub runs: usize,
    pub passed: usize,
    pub required: usize,
}

impl ConsensusStats {
    pub fn pass_rate(&self) -> f64 {
        if self.runs == 0 {
            0.0
        } else {
            self.passed as f64 / self.runs as f64
        }
    }
}

/// Result of a single test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    pub test_id: String,
    pub test_name: Option<String>,
    pub tags: Vec<String>,
    pub passed: bool,
    pub attempts: usize,
    pub assertion_results: Vec<AssertionResult>,
    /// Model-graded results (0.2+).
    #[serde(default)]
    pub judge_results: Vec<JudgeResult>,
    /// Aggregate score in `0.0..=1.0` across assertions + judges (0.2+).
    #[serde(default)]
    pub score: f64,
    /// Consensus distribution, if this test ran in consensus mode (0.2+).
    #[serde(default)]
    pub consensus: Option<ConsensusStats>,
    /// Token / cost accounting for the reported run (0.2+).
    #[serde(default)]
    pub usage: Option<Usage>,
    /// Wall-clock of the reported agent run itself, excluding retries (0.2+).
    #[serde(default)]
    pub run_duration: Option<Duration>,
    pub duration: Duration,
    pub error: Option<String>,
}

impl TestReport {
    /// Compute a `0.0..=1.0` score from assertion pass/fail and judge scores.
    /// Boolean assertions contribute 1.0 (pass) or 0.0 (fail); judges
    /// contribute their raw score.
    pub fn compute_score(
        assertion_results: &[AssertionResult],
        judge_results: &[JudgeResult],
    ) -> f64 {
        let mut sum = 0.0;
        let mut n = 0.0;
        for a in assertion_results {
            sum += if a.passed { 1.0 } else { 0.0 };
            n += 1.0;
        }
        for j in judge_results {
            sum += j.score;
            n += 1.0;
        }
        if n == 0.0 {
            1.0
        } else {
            sum / n
        }
    }
}

/// Aggregate metrics across a suite: token/cost totals and latency percentiles.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SuiteMetrics {
    pub total_usage: Usage,
    /// Number of tests that reported an agent run duration.
    pub timed_runs: usize,
    pub latency_mean_ms: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
}

/// Result of an entire test suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteReport {
    pub suite_name: String,
    pub tests: Vec<TestReport>,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    /// Fraction of tests that passed (0.2+).
    #[serde(default)]
    pub pass_rate: f64,
    /// Mean per-test score in `0.0..=1.0` (0.2+).
    #[serde(default)]
    pub score: f64,
    /// Token/cost/latency aggregates (0.2+).
    #[serde(default)]
    pub metrics: SuiteMetrics,
    pub duration: Duration,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// The delta between a suite report and a baseline.
#[derive(Debug, Clone)]
pub struct SuiteDiff {
    /// Tests that passed in the baseline but fail now.
    pub regressions: Vec<String>,
    /// Tests that failed in the baseline but pass now.
    pub newly_passing: Vec<String>,
    /// Test ids present now but absent from the baseline.
    pub added: Vec<String>,
    /// Test ids present in the baseline but absent now.
    pub removed: Vec<String>,
    pub score_delta: f64,
    pub pass_rate_delta: f64,
}

impl SuiteDiff {
    pub fn has_regressions(&self) -> bool {
        !self.regressions.is_empty()
    }

    pub fn print_console(&self) {
        println!();
        println!("  \x1b[1mBaseline diff\x1b[0m");
        println!("  {}", "─".repeat(50));
        if self.regressions.is_empty()
            && self.newly_passing.is_empty()
            && self.added.is_empty()
            && self.removed.is_empty()
        {
            println!("  \x1b[32mNo changes vs. baseline\x1b[0m");
        } else {
            for id in &self.regressions {
                println!("  \x1b[31m▼ REGRESSED\x1b[0m  {}", id);
            }
            for id in &self.newly_passing {
                println!("  \x1b[32m▲ FIXED\x1b[0m      {}", id);
            }
            for id in &self.added {
                println!("  \x1b[36m+ NEW\x1b[0m        {}", id);
            }
            for id in &self.removed {
                println!("  \x1b[90m- REMOVED\x1b[0m    {}", id);
            }
        }
        println!(
            "  score {:+.3}   pass-rate {:+.1}%",
            self.score_delta,
            self.pass_rate_delta * 100.0
        );
        println!("  {}", "─".repeat(50));
    }
}

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    if sorted_ms.len() == 1 {
        return sorted_ms[0];
    }
    let rank = p * (sorted_ms.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    sorted_ms[lo] + (sorted_ms[hi] - sorted_ms[lo]) * frac
}

impl SuiteReport {
    /// Build a suite report from per-test reports, computing pass counts,
    /// aggregate score, pass rate, and metrics.
    pub fn new(
        suite_name: String,
        tests: Vec<TestReport>,
        total: usize,
        duration: Duration,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        let passed = tests.iter().filter(|r| r.passed).count();
        let failed = tests.len() - passed;
        let pass_rate = if tests.is_empty() {
            0.0
        } else {
            passed as f64 / tests.len() as f64
        };
        let score = if tests.is_empty() {
            0.0
        } else {
            tests.iter().map(|t| t.score).sum::<f64>() / tests.len() as f64
        };

        // Metrics.
        let mut total_usage = Usage::default();
        let mut latencies: Vec<f64> = Vec::new();
        for t in &tests {
            if let Some(u) = &t.usage {
                total_usage = total_usage.add(u);
            }
            if let Some(d) = t.run_duration {
                latencies.push(d.as_secs_f64() * 1000.0);
            }
        }
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let timed_runs = latencies.len();
        let latency_mean_ms = if latencies.is_empty() {
            0.0
        } else {
            latencies.iter().sum::<f64>() / latencies.len() as f64
        };
        let metrics = SuiteMetrics {
            total_usage,
            timed_runs,
            latency_mean_ms,
            latency_p50_ms: percentile(&latencies, 0.50),
            latency_p95_ms: percentile(&latencies, 0.95),
        };

        Self {
            suite_name,
            tests,
            total,
            passed,
            failed,
            pass_rate,
            score,
            metrics,
            duration,
            timestamp,
        }
    }

    /// Diff this report against a baseline, keyed by `test_id`.
    pub fn diff_against(&self, baseline: &SuiteReport) -> SuiteDiff {
        use std::collections::HashMap;
        let base: HashMap<&str, bool> = baseline
            .tests
            .iter()
            .map(|t| (t.test_id.as_str(), t.passed))
            .collect();
        let now: HashMap<&str, bool> = self
            .tests
            .iter()
            .map(|t| (t.test_id.as_str(), t.passed))
            .collect();

        let mut regressions = Vec::new();
        let mut newly_passing = Vec::new();
        let mut added = Vec::new();
        for t in &self.tests {
            match base.get(t.test_id.as_str()) {
                Some(&was_passed) => {
                    if was_passed && !t.passed {
                        regressions.push(t.test_id.clone());
                    } else if !was_passed && t.passed {
                        newly_passing.push(t.test_id.clone());
                    }
                }
                None => added.push(t.test_id.clone()),
            }
        }
        let removed: Vec<String> = baseline
            .tests
            .iter()
            .filter(|t| !now.contains_key(t.test_id.as_str()))
            .map(|t| t.test_id.clone())
            .collect();

        SuiteDiff {
            regressions,
            newly_passing,
            added,
            removed,
            score_delta: self.score - baseline.score,
            pass_rate_delta: self.pass_rate - baseline.pass_rate,
        }
    }

    /// Print colored console output.
    pub fn print_console(&self) {
        println!();
        println!(
            "  \x1b[1m{}\x1b[0m  ({} tests)",
            self.suite_name, self.total
        );
        println!("  {}", "─".repeat(50));

        for test in &self.tests {
            let display_name = test.test_name.as_deref().unwrap_or(&test.test_id);

            if test.passed {
                let mut line = format!("  \x1b[32m✓ PASS\x1b[0m  {}", display_name);
                if let Some(c) = &test.consensus {
                    line.push_str(&format!(
                        "  \x1b[90m[consensus {}/{}]\x1b[0m",
                        c.passed, c.runs
                    ));
                }
                println!("{}", line);
                for jr in &test.judge_results {
                    println!(
                        "          \x1b[90mjudge {:.2} ≥ {:.2}: {}\x1b[0m",
                        jr.score,
                        jr.threshold,
                        truncate(&jr.reason, 80)
                    );
                }
            } else {
                println!("  \x1b[31m✗ FAIL\x1b[0m  {}", display_name);
                for ar in &test.assertion_results {
                    if !ar.passed {
                        let prefix = if ar.is_security {
                            "\x1b[33m🔒\x1b[0m"
                        } else {
                            " "
                        };
                        println!(
                            "          {} {}",
                            prefix,
                            ar.message.as_deref().unwrap_or(&ar.description)
                        );
                    }
                }
                for jr in &test.judge_results {
                    if !jr.passed {
                        println!(
                            "          \x1b[33m⚖\x1b[0m  judge {:.2} < {:.2}: {}",
                            jr.score,
                            jr.threshold,
                            truncate(&jr.reason, 100)
                        );
                    }
                }
                if let Some(c) = &test.consensus {
                    println!(
                        "          consensus: {}/{} passed, needed {}",
                        c.passed, c.runs, c.required
                    );
                }
                if let Some(err) = &test.error {
                    println!("          error: {}", err);
                }
            }
        }

        // --- RBAC summary ---
        let rbac_tests: Vec<_> = self
            .tests
            .iter()
            .filter(|t| t.tags.iter().any(|tag| tag == "rbac"))
            .collect();

        if !rbac_tests.is_empty() {
            println!("  {}", "─".repeat(50));
            println!("  \x1b[1mRBAC Summary\x1b[0m");

            let mut role_results: std::collections::BTreeMap<String, (usize, usize)> =
                std::collections::BTreeMap::new();
            for t in &rbac_tests {
                let role = t
                    .test_name
                    .as_deref()
                    .and_then(|n| n.split(" — ").next())
                    .unwrap_or(&t.test_id)
                    .to_string();
                let entry = role_results.entry(role).or_insert((0, 0));
                entry.0 += 1;
                if t.passed {
                    entry.1 += 1;
                }
            }

            for (role, (total, passed)) in &role_results {
                let color = if passed == total {
                    "\x1b[32m"
                } else {
                    "\x1b[31m"
                };
                println!("    {}{}: {}/{} passed\x1b[0m", color, role, passed, total);
            }

            let rbac_passed = rbac_tests.iter().filter(|t| t.passed).count();
            let rbac_total = rbac_tests.len();
            let rbac_color = if rbac_passed == rbac_total {
                "\x1b[32m"
            } else {
                "\x1b[31m"
            };
            println!(
                "  {}RBAC Total: {}/{} passed\x1b[0m",
                rbac_color, rbac_passed, rbac_total
            );
        }

        println!("  {}", "─".repeat(50));

        let security_tests: Vec<_> = self
            .tests
            .iter()
            .filter(|t| t.assertion_results.iter().any(|a| a.is_security))
            .collect();

        if !security_tests.is_empty() {
            let sec_passed = security_tests.iter().filter(|t| t.passed).count();
            let sec_total = security_tests.len();
            let color = if sec_passed == sec_total {
                "\x1b[32m"
            } else {
                "\x1b[31m"
            };
            println!(
                "  {}Security: {}/{} passed\x1b[0m",
                color, sec_passed, sec_total
            );
        }

        // --- Metrics line ---
        let m = &self.metrics;
        if m.timed_runs > 0 || m.total_usage.total_tokens.is_some() {
            let mut parts: Vec<String> = Vec::new();
            if let Some(tot) = m.total_usage.total_tokens {
                parts.push(format!("{} tok", tot));
            }
            if let Some(cost) = m.total_usage.cost_usd {
                parts.push(format!("${:.4}", cost));
            }
            if m.timed_runs > 0 {
                parts.push(format!(
                    "p50 {:.0}ms p95 {:.0}ms",
                    m.latency_p50_ms, m.latency_p95_ms
                ));
            }
            if !parts.is_empty() {
                println!("  \x1b[90mMetrics: {}\x1b[0m", parts.join("  ·  "));
            }
        }

        let color = if self.failed == 0 {
            "\x1b[32m"
        } else {
            "\x1b[31m"
        };
        println!(
            "  {}Total: {}/{} passed\x1b[0m  score {:.2}  ({:.1}s)",
            color,
            self.passed,
            self.total,
            self.score,
            self.duration.as_secs_f64()
        );
        println!();
    }

    /// Save report to a JSON file.
    pub fn save_to_file(&self, path: &Path) -> Result<(), SpiceError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a previously saved report (e.g. a baseline).
    pub fn load_from_file(path: &Path) -> Result<Self, SpiceError> {
        let text = std::fs::read_to_string(path)?;
        let report: SuiteReport = serde_json::from_str(&text)?;
        Ok(report)
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{}...", truncated)
    }
}
