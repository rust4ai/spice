use crate::assertion::Assertion;
use std::ops::RangeInclusive;

use crate::agent::{AgentConfig, AgentOutput};

/// Default pass threshold for a judge assertion when none is specified.
pub const DEFAULT_JUDGE_THRESHOLD: f64 = 0.7;

/// A model-graded expectation: grade the output against `rubric` and pass iff
/// the judge's score is at least `threshold`.
#[derive(Debug, Clone)]
pub struct JudgeSpec {
    pub rubric: String,
    pub threshold: f64,
}

/// A single test case for an agent.
pub struct TestCase {
    pub id: String,
    pub name: Option<String>,
    pub user_message: String,
    pub config: AgentConfig,
    pub assertions: Vec<Assertion>,
    /// LLM-as-judge expectations, evaluated by the runner's judge.
    pub judges: Vec<JudgeSpec>,
    pub tags: Vec<String>,
    pub retries: usize,
    pub consensus_runs: Option<usize>,
    pub consensus_required: Option<usize>,
    pub timeout: Option<std::time::Duration>,
}

/// A collection of test cases.
pub struct TestSuite {
    pub name: String,
    pub tests: Vec<TestCase>,
    pub default_config: AgentConfig,
    pub default_retries: usize,
    pub default_timeout: Option<std::time::Duration>,
}

impl Default for TestSuite {
    fn default() -> Self {
        Self {
            name: "Test Suite".into(),
            tests: vec![],
            default_config: AgentConfig::empty(),
            default_retries: 0,
            default_timeout: None,
        }
    }
}

/// Builder for constructing test cases fluently.
pub struct TestCaseBuilder {
    id: String,
    user_message: String,
    name: Option<String>,
    config: Option<AgentConfig>,
    assertions: Vec<Assertion>,
    judges: Vec<JudgeSpec>,
    tags: Vec<String>,
    retries: usize,
    consensus_runs: Option<usize>,
    consensus_required: Option<usize>,
    timeout: Option<std::time::Duration>,
}

impl TestCaseBuilder {
    pub fn new(id: impl Into<String>, user_message: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            user_message: user_message.into(),
            name: None,
            config: None,
            assertions: vec![],
            judges: vec![],
            tags: vec![],
            retries: 0,
            consensus_runs: None,
            consensus_required: None,
            timeout: None,
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn tags(mut self, tags: &[&str]) -> Self {
        self.tags.extend(tags.iter().map(|s| s.to_string()));
        self
    }

    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn config_json(mut self, data: serde_json::Value) -> Self {
        self.config = Some(AgentConfig::new(data));
        self
    }

    pub fn retries(mut self, n: usize) -> Self {
        self.retries = n;
        self
    }

    pub fn consensus(mut self, runs: usize, required: usize) -> Self {
        self.consensus_runs = Some(runs);
        self.consensus_required = Some(required);
        self
    }

    pub fn timeout(mut self, duration: std::time::Duration) -> Self {
        self.timeout = Some(duration);
        self
    }

    // --- Assertion builders ---

    pub fn expect_tools(mut self, tools: &[&str]) -> Self {
        self.assertions.push(Assertion::ExpectTools(
            tools.iter().map(|s| s.to_string()).collect(),
        ));
        self
    }

    pub fn forbid_tools(mut self, tools: &[&str]) -> Self {
        self.assertions.push(Assertion::ForbidTools(
            tools.iter().map(|s| s.to_string()).collect(),
        ));
        self
    }

    pub fn expect_any_tool(mut self) -> Self {
        self.assertions.push(Assertion::ExpectAnyTool);
        self
    }

    pub fn expect_no_tools(mut self) -> Self {
        self.assertions.push(Assertion::ExpectNoTools);
        self
    }

    /// The set of tools called equals exactly this set (order/count ignored,
    /// but no extra tools and none missing).
    pub fn expect_exact_tools(mut self, tools: &[&str]) -> Self {
        self.assertions.push(Assertion::ExpectExactTools(
            tools.iter().map(|s| s.to_string()).collect(),
        ));
        self
    }

    /// Grade the agent's final output against a natural-language rubric using
    /// the runner's judge. Passes iff the judge's score ≥
    /// [`DEFAULT_JUDGE_THRESHOLD`].
    pub fn expect_judge(mut self, rubric: impl Into<String>) -> Self {
        self.judges.push(JudgeSpec {
            rubric: rubric.into(),
            threshold: DEFAULT_JUDGE_THRESHOLD,
        });
        self
    }

    /// Like [`Self::expect_judge`] but with an explicit pass threshold.
    pub fn expect_judge_threshold(mut self, rubric: impl Into<String>, threshold: f64) -> Self {
        self.judges.push(JudgeSpec {
            rubric: rubric.into(),
            threshold,
        });
        self
    }

    pub fn expect_text_contains(mut self, s: impl Into<String>) -> Self {
        self.assertions
            .push(Assertion::ExpectTextContains(s.into()));
        self
    }

    pub fn expect_text_not_contains(mut self, s: impl Into<String>) -> Self {
        self.assertions
            .push(Assertion::ExpectTextNotContains(s.into()));
        self
    }

    pub fn expect_turns(mut self, range: RangeInclusive<usize>) -> Self {
        self.assertions.push(Assertion::ExpectTurns(range));
        self
    }

    pub fn expect_tools_within_allowlist(mut self) -> Self {
        self.assertions.push(Assertion::ExpectToolsWithinAllowlist);
        self
    }

    pub fn expect_no_error(mut self) -> Self {
        self.assertions.push(Assertion::ExpectNoError);
        self
    }

    pub fn expect_tool_args(mut self, tool: impl Into<String>, args: serde_json::Value) -> Self {
        self.assertions
            .push(Assertion::ExpectToolArgs(tool.into(), args));
        self
    }

    pub fn expect_tool_args_contain(
        mut self,
        tool: impl Into<String>,
        partial: serde_json::Value,
    ) -> Self {
        self.assertions
            .push(Assertion::ExpectToolArgsContain(tool.into(), partial));
        self
    }

    pub fn expect_tool_arg(
        mut self,
        tool: impl Into<String>,
        param: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.assertions
            .push(Assertion::ExpectToolArg(tool.into(), param.into(), value));
        self
    }

    pub fn expect_tool_arg_exists(
        mut self,
        tool: impl Into<String>,
        param: impl Into<String>,
    ) -> Self {
        self.assertions
            .push(Assertion::ExpectToolArgExists(tool.into(), param.into()));
        self
    }

    pub fn expect_tool_call_count(mut self, tool: impl Into<String>, count: usize) -> Self {
        self.assertions
            .push(Assertion::ExpectToolCallCount(tool.into(), count));
        self
    }

    pub fn expect_tool_call_order(mut self, order: &[&str]) -> Self {
        self.assertions.push(Assertion::ExpectToolCallOrder(
            order.iter().map(|s| s.to_string()).collect(),
        ));
        self
    }

    pub fn expect_tool_on_turn(mut self, turn: usize, tool: impl Into<String>) -> Self {
        self.assertions
            .push(Assertion::ExpectToolOnTurn(turn, tool.into()));
        self
    }

    pub fn expect<F>(mut self, f: F) -> Self
    where
        F: Fn(&AgentOutput) -> Result<(), String> + Send + Sync + 'static,
    {
        self.assertions.push(Assertion::Custom(Box::new(f)));
        self
    }

    // --- Multi-turn assertion builders ---

    pub fn with_role(self, role: &str) -> Self {
        self.config_json(serde_json::json!({"role": role}))
    }

    pub fn expect_tools_in_turn_range(
        mut self,
        range: RangeInclusive<usize>,
        tools: &[&str],
    ) -> Self {
        self.assertions.push(Assertion::ExpectToolsInTurnRange(
            range,
            tools.iter().map(|s| s.to_string()).collect(),
        ));
        self
    }

    pub fn forbid_tools_in_turn_range(
        mut self,
        range: RangeInclusive<usize>,
        tools: &[&str],
    ) -> Self {
        self.assertions.push(Assertion::ForbidToolsInTurnRange(
            range,
            tools.iter().map(|s| s.to_string()).collect(),
        ));
        self
    }

    pub fn expect_final_tool(mut self, tool: &str) -> Self {
        self.assertions
            .push(Assertion::ExpectFinalTool(tool.to_string()));
        self
    }

    pub fn expect_final_tool_arg(
        mut self,
        tool: &str,
        param: &str,
        value: serde_json::Value,
    ) -> Self {
        self.assertions.push(Assertion::ExpectFinalToolArg(
            tool.to_string(),
            param.to_string(),
            value,
        ));
        self
    }

    /// Shorthand: gathering tools before default action tools (say_to_user, task_fully_completed).
    pub fn expect_gathering_phase(mut self, gather_tools: &[&str]) -> Self {
        self.assertions.push(Assertion::ExpectGatheringBeforeAction(
            gather_tools.iter().map(|s| s.to_string()).collect(),
            vec![
                "say_to_user".to_string(),
                "task_fully_completed".to_string(),
            ],
        ));
        self
    }

    /// Explicit: gathering tools before specified action tools.
    pub fn expect_gathering_before_action(
        mut self,
        gather_tools: &[&str],
        action_tools: &[&str],
    ) -> Self {
        self.assertions.push(Assertion::ExpectGatheringBeforeAction(
            gather_tools.iter().map(|s| s.to_string()).collect(),
            action_tools.iter().map(|s| s.to_string()).collect(),
        ));
        self
    }

    pub fn expect_tool_only_on_final_turn(mut self, tool: &str) -> Self {
        self.assertions
            .push(Assertion::ExpectToolOnlyOnFinalTurn(tool.to_string()));
        self
    }

    pub fn build(self) -> TestCase {
        TestCase {
            id: self.id,
            name: self.name,
            user_message: self.user_message,
            config: self.config.unwrap_or_else(AgentConfig::empty),
            assertions: self.assertions,
            judges: self.judges,
            tags: self.tags,
            retries: self.retries,
            consensus_runs: self.consensus_runs,
            consensus_required: self.consensus_required,
            timeout: self.timeout,
        }
    }
}
