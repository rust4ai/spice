use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::error::SpiceError;

/// Configuration passed to the agent under test. Wraps arbitrary JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentConfig {
    pub data: serde_json::Value,
}

impl AgentConfig {
    pub fn new(data: serde_json::Value) -> Self {
        Self { data }
    }

    pub fn empty() -> Self {
        Self {
            data: serde_json::Value::Null,
        }
    }
}

/// A single tool call made by the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// A single turn in the agent's execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub index: usize,
    pub output_text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<serde_json::Value>,
    pub stop_reason: Option<String>,
    pub duration: Duration,
}

/// Token / cost accounting for a single agent run.
///
/// Every field is optional so adapters can report whatever their provider
/// exposes. The runner aggregates these across a suite into totals and
/// per-run latency percentiles.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
}

impl Usage {
    pub fn tokens(input: u64, output: u64) -> Self {
        Self {
            input_tokens: Some(input),
            output_tokens: Some(output),
            total_tokens: Some(input + output),
            cost_usd: None,
        }
    }

    pub fn with_cost(mut self, usd: f64) -> Self {
        self.cost_usd = Some(usd);
        self
    }

    /// Element-wise sum, treating `None` as zero (but staying `None` if both
    /// operands omit a field).
    pub fn add(&self, other: &Usage) -> Usage {
        fn sum(a: Option<u64>, b: Option<u64>) -> Option<u64> {
            match (a, b) {
                (None, None) => None,
                _ => Some(a.unwrap_or(0) + b.unwrap_or(0)),
            }
        }
        fn sumf(a: Option<f64>, b: Option<f64>) -> Option<f64> {
            match (a, b) {
                (None, None) => None,
                _ => Some(a.unwrap_or(0.0) + b.unwrap_or(0.0)),
            }
        }
        Usage {
            input_tokens: sum(self.input_tokens, other.input_tokens),
            output_tokens: sum(self.output_tokens, other.output_tokens),
            total_tokens: sum(self.total_tokens, other.total_tokens),
            cost_usd: sumf(self.cost_usd, other.cost_usd),
        }
    }
}

/// The complete output of an agent run.
///
/// Construct with a struct literal and `..Default::default()`, or field by
/// field. `usage` was added in 0.2 and defaults to `None`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentOutput {
    pub final_text: String,
    pub turns: Vec<Turn>,
    pub tools_called: Vec<String>,
    pub duration: Duration,
    pub error: Option<String>,
    /// Token / cost accounting, if the adapter reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl AgentOutput {
    /// Collect all tool calls across all turns.
    pub fn all_tool_calls(&self) -> Vec<&ToolCall> {
        self.turns
            .iter()
            .flat_map(|t| t.tool_calls.iter())
            .collect()
    }

    /// Get tool calls filtered by name.
    pub fn tool_calls_by_name(&self, name: &str) -> Vec<&ToolCall> {
        self.all_tool_calls()
            .into_iter()
            .filter(|tc| tc.name == name)
            .collect()
    }

    /// The canonical list of tool names called, derived from the turn trace
    /// when available and falling back to the flat `tools_called` field.
    ///
    /// Assertions use this so tool-presence checks and per-turn checks share a
    /// single source of truth even if an adapter populates the two
    /// inconsistently.
    pub fn tool_names(&self) -> Vec<String> {
        let from_turns: Vec<String> = self
            .turns
            .iter()
            .flat_map(|t| t.tool_calls.iter().map(|tc| tc.name.clone()))
            .collect();
        if from_turns.is_empty() {
            self.tools_called.clone()
        } else {
            from_turns
        }
    }
}

/// Trait that the agent under test must implement.
#[async_trait]
pub trait AgentUnderTest: Send + Sync {
    /// Run the agent with a user message and config, return full output with trace.
    async fn run(
        &self,
        user_message: &str,
        config: &AgentConfig,
    ) -> Result<AgentOutput, SpiceError>;

    /// Return tool names available for this config (for allowlist assertions).
    fn available_tools(&self, config: &AgentConfig) -> Vec<String>;

    /// Human-readable agent name (for reports).
    fn name(&self) -> &str {
        "agent"
    }
}
