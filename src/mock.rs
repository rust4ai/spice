use async_trait::async_trait;
use std::time::Duration;

use crate::agent::{AgentConfig, AgentOutput, AgentUnderTest, ToolCall, Turn};
use crate::error::SpiceError;

/// A scripted response for the mock agent.
#[derive(Debug, Clone)]
pub struct MockResponse {
    pub final_text: String,
    pub tool_calls: Vec<ToolCall>,
    pub error: Option<String>,
}

impl MockResponse {
    /// Create a simple text response with no tool calls.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            final_text: text.into(),
            tool_calls: vec![],
            error: None,
        }
    }

    /// Create a response with tool calls.
    pub fn with_tools(text: impl Into<String>, tools: Vec<ToolCall>) -> Self {
        Self {
            final_text: text.into(),
            tool_calls: tools,
            error: None,
        }
    }

    /// Create an error response.
    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            final_text: String::new(),
            tool_calls: vec![],
            error: Some(msg.into()),
        }
    }
}

/// A single turn in a multi-turn scripted response.
#[derive(Debug, Clone)]
pub struct MockTurn {
    pub tool_calls: Vec<ToolCall>,
    pub output_text: Option<String>,
}

/// A multi-turn scripted response with multiple turns of tool calls.
#[derive(Debug, Clone)]
pub struct MockMultiTurnResponse {
    pub turns: Vec<MockTurn>,
    pub final_text: String,
}

impl MockMultiTurnResponse {
    pub fn new(final_text: impl Into<String>) -> Self {
        Self {
            turns: vec![],
            final_text: final_text.into(),
        }
    }

    /// Add a turn with tool calls.
    pub fn turn(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.turns.push(MockTurn {
            tool_calls,
            output_text: None,
        });
        self
    }

    /// Add a turn with tool calls and output text.
    pub fn turn_with_text(mut self, tool_calls: Vec<ToolCall>, text: impl Into<String>) -> Self {
        self.turns.push(MockTurn {
            tool_calls,
            output_text: Some(text.into()),
        });
        self
    }
}

/// A mock agent for deterministic testing.
pub struct MockAgent {
    name: String,
    responses: std::collections::HashMap<String, MockResponse>,
    multi_turn_responses: std::collections::HashMap<String, MockMultiTurnResponse>,
    default_response: MockResponse,
    tools: Vec<String>,
    role_tools: std::collections::HashMap<String, Vec<String>>,
}

impl MockAgent {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            responses: std::collections::HashMap::new(),
            multi_turn_responses: std::collections::HashMap::new(),
            default_response: MockResponse::text("I don't know how to help with that."),
            tools: vec![],
            role_tools: std::collections::HashMap::new(),
        }
    }

    /// Register a scripted response for a specific user message (exact match).
    pub fn on(mut self, message: impl Into<String>, response: MockResponse) -> Self {
        self.responses.insert(message.into(), response);
        self
    }

    /// Register a multi-turn scripted response for a specific user message.
    pub fn on_multi_turn(
        mut self,
        message: impl Into<String>,
        response: MockMultiTurnResponse,
    ) -> Self {
        self.multi_turn_responses.insert(message.into(), response);
        self
    }

    /// Set the default response for unmatched messages.
    pub fn default_response(mut self, response: MockResponse) -> Self {
        self.default_response = response;
        self
    }

    /// Set the available tools.
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools = tools;
        self
    }

    /// Set per-role tool lists.
    pub fn with_role_tools(mut self, role: &str, tools: &[&str]) -> Self {
        self.role_tools.insert(
            role.to_string(),
            tools.iter().map(|s| s.to_string()).collect(),
        );
        self
    }
}

#[async_trait]
impl AgentUnderTest for MockAgent {
    async fn run(
        &self,
        user_message: &str,
        _config: &AgentConfig,
    ) -> Result<AgentOutput, SpiceError> {
        // Check multi-turn responses first
        if let Some(mt) = self.multi_turn_responses.get(user_message) {
            let mut turns = Vec::new();
            let mut all_tools_called = Vec::new();

            for (i, mock_turn) in mt.turns.iter().enumerate() {
                for tc in &mock_turn.tool_calls {
                    all_tools_called.push(tc.name.clone());
                }
                turns.push(Turn {
                    index: i,
                    output_text: mock_turn.output_text.clone(),
                    tool_calls: mock_turn.tool_calls.clone(),
                    tool_results: vec![],
                    stop_reason: Some("tool_use".into()),
                    duration: Duration::from_millis(1),
                });
            }

            // Fix last turn's stop_reason
            if let Some(last) = turns.last_mut() {
                last.stop_reason = Some("stop".into());
                last.output_text = Some(mt.final_text.clone());
            }

            return Ok(AgentOutput {
                final_text: mt.final_text.clone(),
                turns,
                tools_called: all_tools_called,
                duration: Duration::from_millis(1),
                error: None,
                usage: None,
            });
        }

        // Fall back to single-turn responses
        let response = self
            .responses
            .get(user_message)
            .unwrap_or(&self.default_response);

        if let Some(err) = &response.error {
            return Err(SpiceError::AgentError(err.clone()));
        }

        let tools_called: Vec<String> = response
            .tool_calls
            .iter()
            .map(|tc| tc.name.clone())
            .collect();

        let turn = Turn {
            index: 0,
            output_text: Some(response.final_text.clone()),
            tool_calls: response.tool_calls.clone(),
            tool_results: vec![],
            stop_reason: Some("stop".into()),
            duration: Duration::from_millis(1),
        };

        Ok(AgentOutput {
            final_text: response.final_text.clone(),
            turns: vec![turn],
            tools_called,
            duration: Duration::from_millis(1),
            error: None,
            usage: None,
        })
    }

    fn available_tools(&self, config: &AgentConfig) -> Vec<String> {
        // Check for role-specific tools
        if let Some(role) = config.data.get("role").and_then(|v| v.as_str()) {
            if let Some(tools) = self.role_tools.get(role) {
                return tools.clone();
            }
        }
        self.tools.clone()
    }

    fn name(&self) -> &str {
        &self.name
    }
}
