//! LLM-as-judge: model-graded assertions for free-form agent output.
//!
//! Substring checks can't tell you whether an answer is *correct*, *helpful*,
//! or *safe*. A [`Judge`] scores the agent's output against a natural-language
//! rubric and returns a `0.0..=1.0` score with a written reason. Tests declare
//! judge expectations with [`crate::TestCaseBuilder::expect_judge`], and the
//! [`crate::Runner`] evaluates them with whatever judge you install via
//! [`crate::Runner::with_judge`].
//!
//! Two implementations ship in the box:
//! - [`MockJudge`] — deterministic, offline, keyword-based. For testing suites
//!   (and Spice itself) without a network.
//! - [`OpenAiJudge`] — a real judge backed by OpenAI chat-completions, behind
//!   the `openai` feature flag.

use async_trait::async_trait;

use crate::agent::AgentOutput;
use crate::error::SpiceError;

/// Everything a judge needs to grade one agent run.
pub struct JudgeRequest<'a> {
    /// The rubric to grade against, e.g. "The answer correctly identifies the
    /// capital of France and is phrased as a complete sentence."
    pub rubric: &'a str,
    /// The original user message the agent was responding to.
    pub user_message: &'a str,
    /// The full agent output (final text + trajectory).
    pub output: &'a AgentOutput,
    /// The pass threshold the caller will apply (informational — a judge may
    /// use it to calibrate, but the runner is what enforces it).
    pub threshold: f64,
}

/// A judge's verdict on a single run.
#[derive(Debug, Clone)]
pub struct JudgeVerdict {
    /// Score in `0.0..=1.0`.
    pub score: f64,
    /// Human-readable justification.
    pub reason: String,
}

impl JudgeVerdict {
    pub fn new(score: f64, reason: impl Into<String>) -> Self {
        Self {
            score: score.clamp(0.0, 1.0),
            reason: reason.into(),
        }
    }
}

/// A model (or heuristic) that grades agent output against a rubric.
#[async_trait]
pub trait Judge: Send + Sync {
    async fn score(&self, req: JudgeRequest<'_>) -> Result<JudgeVerdict, SpiceError>;
}

/// A deterministic, offline judge for testing.
///
/// Scoring is a simple keyword heuristic over the agent's `final_text`: the
/// score is the fraction of `require` keywords present, minus a full penalty if
/// any `forbid` keyword is present. Not a substitute for a real model — just a
/// way to exercise judge plumbing in unit tests without a network call.
#[derive(Debug, Clone, Default)]
pub struct MockJudge {
    require: Vec<String>,
    forbid: Vec<String>,
}

impl MockJudge {
    pub fn new() -> Self {
        Self::default()
    }

    /// Keywords that should all appear in the output for a perfect score.
    pub fn require(mut self, kw: &[&str]) -> Self {
        self.require.extend(kw.iter().map(|s| s.to_lowercase()));
        self
    }

    /// Keywords whose presence forces the score to 0.
    pub fn forbid(mut self, kw: &[&str]) -> Self {
        self.forbid.extend(kw.iter().map(|s| s.to_lowercase()));
        self
    }
}

#[async_trait]
impl Judge for MockJudge {
    async fn score(&self, req: JudgeRequest<'_>) -> Result<JudgeVerdict, SpiceError> {
        let text = req.output.final_text.to_lowercase();

        if let Some(bad) = self.forbid.iter().find(|k| text.contains(k.as_str())) {
            return Ok(JudgeVerdict::new(
                0.0,
                format!("forbidden keyword present: {bad:?}"),
            ));
        }

        if self.require.is_empty() {
            return Ok(JudgeVerdict::new(1.0, "no requirements — vacuously pass"));
        }

        let present = self
            .require
            .iter()
            .filter(|k| text.contains(k.as_str()))
            .count();
        let score = present as f64 / self.require.len() as f64;
        let missing: Vec<&String> = self
            .require
            .iter()
            .filter(|k| !text.contains(k.as_str()))
            .collect();
        let reason = if missing.is_empty() {
            "all required keywords present".to_string()
        } else {
            format!("missing keywords: {missing:?}")
        };
        Ok(JudgeVerdict::new(score, reason))
    }
}

/// A real LLM-as-judge backed by the OpenAI chat-completions API.
///
/// Reads `OPENAI_API_KEY` from the environment (or takes it explicitly). The
/// judge is asked to return strict JSON `{"score": <0..1>, "reason": "..."}`.
///
/// Requires the `openai` feature.
#[cfg(feature = "openai")]
pub struct OpenAiJudge {
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
}

#[cfg(feature = "openai")]
impl OpenAiJudge {
    /// Build a judge, reading `OPENAI_API_KEY` from the environment.
    pub fn from_env() -> Result<Self, SpiceError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            SpiceError::ConfigError("OPENAI_API_KEY not set for OpenAiJudge".into())
        })?;
        Ok(Self::new(api_key))
    }

    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "gpt-4o-mini".into(),
            base_url: "https://api.openai.com/v1".into(),
            client: reqwest::Client::new(),
        }
    }

    /// Override the grading model (default `gpt-4o-mini`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the API base URL (for proxies / OpenAI-compatible endpoints).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[cfg(feature = "openai")]
#[async_trait]
impl Judge for OpenAiJudge {
    async fn score(&self, req: JudgeRequest<'_>) -> Result<JudgeVerdict, SpiceError> {
        let tools_called = req.output.tool_names();
        let system = "You are a strict evaluator of an AI agent's output. \
             Grade how well the output satisfies the given rubric. \
             Respond ONLY with a JSON object of the form \
             {\"score\": <number between 0 and 1>, \"reason\": <string>}. \
             1.0 means the rubric is fully satisfied, 0.0 means not at all.";
        let user = format!(
            "RUBRIC:\n{}\n\nUSER MESSAGE TO THE AGENT:\n{}\n\nAGENT FINAL OUTPUT:\n{}\n\nTOOLS THE AGENT CALLED:\n{:?}\n\nReturn the JSON verdict now.",
            req.rubric, req.user_message, req.output.final_text, tools_called
        );

        let body = serde_json::json!({
            "model": self.model,
            "temperature": 0.0,
            "response_format": { "type": "json_object" },
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": user },
            ]
        });

        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| SpiceError::AgentError(format!("judge request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(SpiceError::AgentError(format!(
                "judge HTTP {status}: {text}"
            )));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| SpiceError::AgentError(format!("judge decode failed: {e}")))?;

        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| SpiceError::AgentError("judge returned no content".into()))?;

        let parsed: serde_json::Value = serde_json::from_str(content).map_err(|e| {
            SpiceError::AgentError(format!("judge returned non-JSON: {e}: {content}"))
        })?;

        let score = parsed["score"].as_f64().ok_or_else(|| {
            SpiceError::AgentError(format!("judge JSON missing numeric score: {content}"))
        })?;
        let reason = parsed["reason"]
            .as_str()
            .unwrap_or("(no reason provided)")
            .to_string();

        Ok(JudgeVerdict::new(score, reason))
    }
}
