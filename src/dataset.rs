//! Dataset-driven evals: run the *same* assertions over N inputs and report an
//! aggregate pass rate, instead of hand-writing one test per prompt.
//!
//! Load rows from JSONL/JSON, then expand each into a [`crate::TestCase`] with a
//! closure that maps the row to its assertions. The row's `expected` field
//! (arbitrary JSON) is available to the closure so the same rubric can be
//! parameterized per row.
//!
//! ```no_run
//! use spice_framework::{suite, Runner, RunnerConfig};
//! use spice_framework::dataset::Dataset;
//! use std::sync::Arc;
//! # async fn demo(agent: Arc<dyn spice_framework::AgentUnderTest>) {
//! let data = Dataset::from_jsonl("cases.jsonl").unwrap();
//! let tests = data.map_tests(|row, b| {
//!     b.name(format!("case: {}", row.input))
//!         .expect_no_error()
//!         .expect_judge(row.expected_str("rubric").unwrap_or("The answer is correct."))
//! });
//! let report = Runner::new(RunnerConfig::default())
//!     .run(suite("dataset", tests), agent)
//!     .await;
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::SpiceError;
use crate::test_case::{TestCase, TestCaseBuilder};

/// A single dataset row: an input to send the agent plus optional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetRow {
    /// The user message to send the agent.
    pub input: String,
    /// Stable id for this row. Auto-filled with the row index if omitted.
    #[serde(default)]
    pub id: Option<String>,
    /// Arbitrary expected-value payload, available to the mapping closure
    /// (e.g. a rubric, a gold answer, expected tool names).
    #[serde(default)]
    pub expected: serde_json::Value,
    /// Tags applied to every generated test for this row.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl DatasetRow {
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            id: None,
            expected: serde_json::Value::Null,
            tags: vec![],
        }
    }

    /// Read a string field out of `expected` (top-level object key).
    pub fn expected_str(&self, key: &str) -> Option<&str> {
        self.expected.get(key).and_then(|v| v.as_str())
    }

    /// The row's id, falling back to the provided index-derived default.
    pub fn id_or(&self, fallback: impl Into<String>) -> String {
        self.id.clone().unwrap_or_else(|| fallback.into())
    }
}

/// A collection of dataset rows.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Dataset {
    pub rows: Vec<DatasetRow>,
}

impl Dataset {
    pub fn new(rows: Vec<DatasetRow>) -> Self {
        Self { rows }
    }

    /// Load from a JSONL file — one `DatasetRow` JSON object per line. Blank
    /// lines are skipped.
    pub fn from_jsonl(path: impl AsRef<Path>) -> Result<Self, SpiceError> {
        let text = std::fs::read_to_string(path)?;
        let mut rows = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let row: DatasetRow = serde_json::from_str(line)
                .map_err(|e| SpiceError::ConfigError(format!("dataset line {}: {e}", i + 1)))?;
            rows.push(row);
        }
        Ok(Self { rows })
    }

    /// Load from a JSON file containing either a top-level array of rows or an
    /// object `{ "rows": [...] }`.
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self, SpiceError> {
        let text = std::fs::read_to_string(path)?;
        let value: serde_json::Value = serde_json::from_str(&text)?;
        let rows: Vec<DatasetRow> = match value {
            serde_json::Value::Array(_) => serde_json::from_value(value)?,
            serde_json::Value::Object(ref o) if o.contains_key("rows") => {
                serde_json::from_value(value["rows"].clone())?
            }
            _ => {
                return Err(SpiceError::ConfigError(
                    "dataset JSON must be an array or an object with a `rows` field".into(),
                ))
            }
        };
        Ok(Self { rows })
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Expand every row into a [`TestCase`]. The closure receives the row and a
    /// pre-seeded [`TestCaseBuilder`] (id, input, and row tags already set) and
    /// returns the builder with its assertions attached.
    pub fn map_tests<F>(&self, mut f: F) -> Vec<TestCase>
    where
        F: FnMut(&DatasetRow, TestCaseBuilder) -> TestCaseBuilder,
    {
        self.rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let id = row.id_or(format!("row-{i}"));
                let mut builder = TestCaseBuilder::new(id, row.input.clone());
                if !row.tags.is_empty() {
                    let refs: Vec<&str> = row.tags.iter().map(|s| s.as_str()).collect();
                    builder = builder.tags(&refs);
                }
                f(row, builder).build()
            })
            .collect()
    }
}
