use crate::agent::AgentConfig;
use crate::assertion::Assertion;
use crate::test_case::TestCase;
use crate::TestCaseBuilder;

/// Role-based access control matrix for test generation.
pub struct RbacMatrix {
    roles: Vec<(String, Vec<String>)>,
}

impl RbacMatrix {
    pub fn new() -> Self {
        Self { roles: vec![] }
    }

    /// Add a role with its allowed tools.
    pub fn role(mut self, name: &str, tools: &[&str]) -> Self {
        self.roles.push((
            name.to_string(),
            tools.iter().map(|s| s.to_string()).collect(),
        ));
        self
    }

    /// Get allowed tools for a role.
    pub fn tools_for(&self, role: &str) -> Vec<&str> {
        self.roles
            .iter()
            .find(|(r, _)| r == role)
            .map(|(_, tools)| tools.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get all unique tools across all roles.
    pub fn all_tools(&self) -> Vec<&str> {
        let mut all: Vec<&str> = self
            .roles
            .iter()
            .flat_map(|(_, tools)| tools.iter().map(|s| s.as_str()))
            .collect();
        all.sort();
        all.dedup();
        all
    }

    /// Build an AgentConfig for a specific role.
    pub fn config_for_role(&self, role: &str) -> AgentConfig {
        AgentConfig::new(serde_json::json!({"role": role}))
    }

    /// Auto-generate allowlist tests: one test per role verifying tools within allowlist.
    pub fn generate_allowlist_tests(&self, user_message: &str) -> Vec<TestCase> {
        let mut tests = Vec::new();

        for (role, role_tools) in &self.roles {
            let all = self.all_tools();
            let forbidden: Vec<&str> = all
                .iter()
                .filter(|t| !role_tools.iter().any(|rt| rt.as_str() == **t))
                .copied()
                .collect();

            let mut builder = TestCaseBuilder::new(
                format!("rbac-allowlist-{}", role.to_lowercase()),
                user_message,
            )
            .name(format!("{} — tools within allowlist", role))
            .tags(&["rbac", "security"])
            .config(self.config_for_role(role))
            .expect_tools_within_allowlist();

            if !forbidden.is_empty() {
                builder = builder.forbid_tools(&forbidden);
            }

            tests.push(builder.build());
        }

        tests
    }

    /// Generate scenario tests with per-role custom assertions.
    /// Takes ownership of assertions since `Assertion` is not Clone.
    pub fn generate_scenario_tests(
        &self,
        id: &str,
        msg: &str,
        role_assertions: Vec<(&str, Vec<Assertion>)>,
    ) -> Vec<TestCase> {
        let mut tests = Vec::new();

        for (role, assertions) in role_assertions {
            let mut tc = TestCase {
                id: format!("rbac-{}-{}", id, role.to_lowercase()),
                name: Some(format!("{} — {} scenario", role, id)),
                user_message: msg.to_string(),
                config: self.config_for_role(role),
                assertions,
                judges: vec![],
                tags: vec!["rbac".to_string(), "security".to_string()],
                retries: 0,
                consensus_runs: None,
                consensus_required: None,
                timeout: None,
            };
            // Always add allowlist check
            tc.assertions.push(Assertion::ExpectToolsWithinAllowlist);
            tests.push(tc);
        }

        tests
    }

    /// Generate injection tests: for each role × payload, verify tools stay within allowlist.
    pub fn generate_injection_tests(&self, payloads: &[(&str, &str)]) -> Vec<TestCase> {
        let mut tests = Vec::new();

        for (role, _) in &self.roles {
            for (payload_id, payload_msg) in payloads {
                tests.push(
                    TestCaseBuilder::new(
                        format!("rbac-injection-{}-{}", role.to_lowercase(), payload_id),
                        *payload_msg,
                    )
                    .name(format!("{} — injection: {}", role, payload_id))
                    .tags(&["rbac", "security", "injection"])
                    .config(self.config_for_role(role))
                    .expect_tools_within_allowlist()
                    .build(),
                );
            }
        }

        tests
    }
}

impl Default for RbacMatrix {
    fn default() -> Self {
        Self::new()
    }
}

// --- Convenience constructors for assertions ---

/// Create a ForbidTools assertion.
pub fn forbid_tools(tools: &[&str]) -> Assertion {
    Assertion::ForbidTools(tools.iter().map(|s| s.to_string()).collect())
}

/// Create an ExpectTools assertion.
pub fn expect_tools(tools: &[&str]) -> Assertion {
    Assertion::ExpectTools(tools.iter().map(|s| s.to_string()).collect())
}
