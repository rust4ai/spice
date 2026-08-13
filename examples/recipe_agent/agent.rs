use async_trait::async_trait;
use serde_json::json;
use spice_framework::agent::{AgentConfig, AgentOutput, AgentUnderTest, ToolCall, Turn};
use spice_framework::error::SpiceError;
use spice_framework::toolkit::{PromptTemplate, Toolkit};
use std::path::Path;
use std::time::Instant;

pub struct RecipeAgent {
    api_key: String,
    client: reqwest::Client,
    toolkit: Toolkit,
    system_prompt: String,
}

impl RecipeAgent {
    pub fn new(api_key: String, example_dir: &Path) -> Self {
        let toolkit =
            Toolkit::from_dir(&example_dir.join("tools")).expect("Failed to load tool definitions");
        let template = PromptTemplate::from_file(&example_dir.join("prompt.md"))
            .expect("Failed to load prompt template");
        let system_prompt = template.render(&toolkit);

        Self {
            api_key,
            client: reqwest::Client::new(),
            toolkit,
            system_prompt,
        }
    }

    fn execute_tool(name: &str, args: &serde_json::Value) -> String {
        match name {
            "searchRecipes" => {
                let ingredients = args
                    .get("ingredients")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                format!(
                    "Found 3 recipes with {}: 1) Chicken Fried Rice, 2) Chicken Rice Bowl, 3) Arroz con Pollo",
                    ingredients
                )
            }
            "getNutrition" => {
                let food = args
                    .get("food")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                format!(
                    "Nutrition for {}: 160 calories, 2g protein, 15g fat, 9g carbs, rich in potassium and vitamin K",
                    food
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
            "tools": self.toolkit.to_openai_json(),
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
impl AgentUnderTest for RecipeAgent {
    async fn run(
        &self,
        user_message: &str,
        _config: &AgentConfig,
    ) -> Result<AgentOutput, SpiceError> {
        let start = Instant::now();

        let system_msg = json!({
            "role": "system",
            "content": self.system_prompt
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
        self.toolkit.tool_names()
    }

    fn name(&self) -> &str {
        "recipe-agent"
    }
}
