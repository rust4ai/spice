pub mod agent;
pub mod assertion;
pub mod dataset;
pub mod error;
pub mod judge;
pub mod mock;
pub mod multi_turn;
pub mod rbac;
pub mod report;
pub mod runner;
pub mod test_case;
pub mod toolkit;
pub mod trace;

pub use agent::{AgentConfig, AgentOutput, AgentUnderTest, ToolCall, Turn, Usage};
pub use assertion::Assertion;
pub use dataset::{Dataset, DatasetRow};
pub use error::SpiceError;
pub use judge::{Judge, JudgeRequest, JudgeVerdict, MockJudge};
pub use mock::{MockAgent, MockMultiTurnResponse, MockResponse, MockTurn};
pub use rbac::RbacMatrix;
pub use report::{ConsensusStats, JudgeResult, SuiteDiff, SuiteMetrics, SuiteReport, TestReport};
pub use runner::{Runner, RunnerConfig};
pub use test_case::{JudgeSpec, TestCase, TestCaseBuilder, TestSuite, DEFAULT_JUDGE_THRESHOLD};
pub use toolkit::{ParamDef, PromptTemplate, ToolDef, Toolkit};

#[cfg(feature = "openai")]
pub use judge::OpenAiJudge;

/// Convenience function to start building a test case.
pub fn test(id: impl Into<String>, user_message: impl Into<String>) -> TestCaseBuilder {
    TestCaseBuilder::new(id, user_message)
}

/// Convenience function to create a test suite.
pub fn suite(name: impl Into<String>, tests: Vec<TestCase>) -> TestSuite {
    TestSuite {
        name: name.into(),
        tests,
        ..Default::default()
    }
}
