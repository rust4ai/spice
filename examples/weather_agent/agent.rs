use async_trait::async_trait;
use serde_json::json;
use spice_framework::agent::{AgentConfig, AgentOutput, AgentUnderTest, ToolCall, Turn};
use spice_framework::error::SpiceError;
use std::time::Instant;

pub struct WeatherAgent {
    api_key: String,
    client: reqwest::Client,
}

impl WeatherAgent {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
        }
    }

    fn tool_definition() -> serde_json::Value {
        json!({
            "type": "function",
            "function": {
                "name": "getWeather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name"
                        }
                    },
                    "required": ["location"]
                }
            }
        })
    }

    fn execute_tool(name: &str, args: &serde_json::Value) -> String {
        match name {
            "getWeather" => {
                let location = args
                    .get("location")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown");
                format!(
                    "Weather in {}: 72°F, partly cloudy, humidity 45%, wind 8mph NW",
                    location
                )
            }
            _ => format!("Unknown tool: {}", name),
        }
    }

    async fn call_openai(
        &self,
        messages: &[serde_json::Value],
    ) -> Result<serde_json::Value, SpiceError> {
        let body = json!({
            "model": "gpt-4o-mini",
            "messages": messages,
            "tools": [Self::tool_definition()],
            "tool_choice": "auto"
        });

        let resp = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| SpiceError::AgentError(format!("HTTP error: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_else(|_| "no body".into());
            return Err(SpiceError::AgentError(format!(
                "OpenAI API error {}: {}",
                status, text
            )));
        }

        resp.json::<serde_json::Value>()
            .await
            .map_err(|e| SpiceError::AgentError(format!("JSON parse error: {}", e)))
    }
}

#[async_trait]
impl AgentUnderTest for WeatherAgent {
    async fn run(
        &self,
        user_message: &str,
        _config: &AgentConfig,
    ) -> Result<AgentOutput, SpiceError> {
        let start = Instant::now();

        let system_msg = json!({
            "role": "system",
            "content": "You are a weather assistant. Use the getWeather tool to look up weather for any location the user asks about. If the user is just greeting you or asking something unrelated to weather, respond normally without using tools."
        });
        let user_msg = json!({
            "role": "user",
            "content": user_message
        });

        let mut messages = vec![system_msg, user_msg];
        let mut turns: Vec<Turn> = Vec::new();
        let mut all_tools_called: Vec<String> = Vec::new();
        let max_turns = 5;

        for turn_idx in 0..max_turns {
            let turn_start = Instant::now();
            let response = self.call_openai(&messages).await?;

            let choice = response
                .get("choices")
                .and_then(|c| c.get(0))
                .ok_or_else(|| SpiceError::AgentError("No choices in response".into()))?;

            let message = choice
                .get("message")
                .ok_or_else(|| SpiceError::AgentError("No message in choice".into()))?;

            let finish_reason = choice
                .get("finish_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            let content = message
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let tool_calls_json = message
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            let mut tool_calls = Vec::new();
            let mut tool_results = Vec::new();

            if !tool_calls_json.is_empty() {
                // Add the assistant message with all tool calls once
                messages.push(json!({
                    "role": "assistant",
                    "tool_calls": tool_calls_json
                }));
            }

            for tc in &tool_calls_json {
                let id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let func = tc.get("function").cloned().unwrap_or(json!({}));
                let name = func
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let args_str = func
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");
                let arguments: serde_json::Value =
                    serde_json::from_str(args_str).unwrap_or(json!({}));

                let result = Self::execute_tool(&name, &arguments);

                tool_calls.push(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments,
                });
                all_tools_called.push(name.clone());

                tool_results.push(json!(result));

                // Add each tool result message
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": result
                }));
            }

            turns.push(Turn {
                index: turn_idx,
                output_text: content.clone(),
                tool_calls,
                tool_results,
                stop_reason: Some(finish_reason.clone()),
                duration: turn_start.elapsed(),
            });

            if finish_reason == "stop" || tool_calls_json.is_empty() {
                let final_text = content.unwrap_or_default();
                return Ok(AgentOutput {
                    final_text,
                    turns,
                    tools_called: all_tools_called,
                    duration: start.elapsed(),
                    error: None,
                    usage: None,
                });
            }
        }

        // If we exhausted max turns, return what we have
        let final_text = turns
            .last()
            .and_then(|t| t.output_text.clone())
            .unwrap_or_default();

        Ok(AgentOutput {
            final_text,
            turns,
            tools_called: all_tools_called,
            duration: start.elapsed(),
            error: None,
            usage: None,
        })
    }

    fn available_tools(&self, _config: &AgentConfig) -> Vec<String> {
        vec!["getWeather".into()]
    }

    fn name(&self) -> &str {
        "weather-agent"
    }
}
