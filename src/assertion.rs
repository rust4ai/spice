use crate::agent::AgentOutput;
use crate::multi_turn;
use serde::{Deserialize, Serialize};
use std::ops::RangeInclusive;

/// Result of evaluating a single assertion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssertionResult {
    pub description: String,
    pub passed: bool,
    pub message: Option<String>,
    pub is_security: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
}

/// An assertion to evaluate against agent output.
pub enum Assertion {
    ExpectTools(Vec<String>),
    ForbidTools(Vec<String>),
    /// The exact set of tools called (no extras, none missing).
    ExpectExactTools(Vec<String>),
    ExpectAnyTool,
    ExpectNoTools,
    ExpectTextContains(String),
    ExpectTextNotContains(String),
    ExpectTurns(RangeInclusive<usize>),
    ExpectToolsWithinAllowlist,
    ExpectNoError,
    ExpectToolArgs(String, serde_json::Value),
    ExpectToolArgsContain(String, serde_json::Value),
    ExpectToolArg(String, String, serde_json::Value),
    ExpectToolArgExists(String, String),
    ExpectToolCallCount(String, usize),
    ExpectToolCallOrder(Vec<String>),
    ExpectToolOnTurn(usize, String),
    /// Tools must appear in the specified turn range.
    ExpectToolsInTurnRange(RangeInclusive<usize>, Vec<String>),
    /// Tools must NOT appear in the specified turn range.
    ForbidToolsInTurnRange(RangeInclusive<usize>, Vec<String>),
    /// Last turn must contain this tool call.
    ExpectFinalTool(String),
    /// Last turn's tool call must have this argument value.
    ExpectFinalToolArg(String, String, serde_json::Value),
    /// Gather tools must appear before action tools.
    ExpectGatheringBeforeAction(Vec<String>, Vec<String>),
    /// Tool appears on last turn and no other.
    ExpectToolOnlyOnFinalTurn(String),
    Custom(Box<dyn Fn(&AgentOutput) -> Result<(), String> + Send + Sync>),
}

impl Assertion {
    /// Whether this is a security-related assertion.
    pub fn is_security(&self) -> bool {
        matches!(
            self,
            Assertion::ForbidTools(_)
                | Assertion::ForbidToolsInTurnRange(_, _)
                | Assertion::ExpectToolsWithinAllowlist
        )
    }

    /// Category for reporting grouping.
    pub fn category(&self) -> Option<&str> {
        match self {
            Assertion::ExpectToolsInTurnRange(_, _)
            | Assertion::ForbidToolsInTurnRange(_, _)
            | Assertion::ExpectFinalTool(_)
            | Assertion::ExpectFinalToolArg(_, _, _)
            | Assertion::ExpectGatheringBeforeAction(_, _)
            | Assertion::ExpectToolOnlyOnFinalTurn(_) => Some("multi-turn"),
            _ => None,
        }
    }

    /// Evaluate this assertion against agent output.
    pub fn evaluate(&self, output: &AgentOutput, available_tools: &[String]) -> AssertionResult {
        let is_security = self.is_security();
        let category = self.category().map(|s| s.to_string());

        let tools_called = output.tool_names();

        match self {
            Assertion::ExpectTools(tools) => {
                let missing: Vec<_> = tools.iter().filter(|t| !tools_called.contains(t)).collect();
                AssertionResult {
                    description: format!("expect tools {:?}", tools),
                    passed: missing.is_empty(),
                    message: if missing.is_empty() {
                        None
                    } else {
                        Some(format!("Missing tool calls: {:?}", missing))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::ForbidTools(tools) => {
                let found: Vec<_> = tools.iter().filter(|t| tools_called.contains(t)).collect();
                AssertionResult {
                    description: format!("forbid tools {:?}", tools),
                    passed: found.is_empty(),
                    message: if found.is_empty() {
                        None
                    } else {
                        Some(format!("Forbidden tools were called: {:?}", found))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::ExpectExactTools(tools) => {
                use std::collections::BTreeSet;
                let want: BTreeSet<&String> = tools.iter().collect();
                let got: BTreeSet<&String> = tools_called.iter().collect();
                let missing: Vec<&&String> = want.difference(&got).collect();
                let extra: Vec<&&String> = got.difference(&want).collect();
                let passed = missing.is_empty() && extra.is_empty();
                AssertionResult {
                    description: format!("expect exactly tools {:?}", tools),
                    passed,
                    message: if passed {
                        None
                    } else {
                        Some(format!(
                            "Tool set mismatch. Missing: {:?}, Unexpected: {:?}",
                            missing, extra
                        ))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::ExpectAnyTool => AssertionResult {
                description: "expect any tool call".into(),
                passed: !tools_called.is_empty(),
                message: if tools_called.is_empty() {
                    Some("No tools were called".into())
                } else {
                    None
                },
                is_security,
                category,
            },

            Assertion::ExpectNoTools => AssertionResult {
                description: "expect no tool calls".into(),
                passed: tools_called.is_empty(),
                message: if tools_called.is_empty() {
                    None
                } else {
                    Some(format!("Tools were called: {:?}", tools_called))
                },
                is_security,
                category,
            },

            Assertion::ExpectTextContains(s) => AssertionResult {
                description: format!("expect text contains {:?}", s),
                passed: output.final_text.contains(s.as_str()),
                message: if output.final_text.contains(s.as_str()) {
                    None
                } else {
                    Some(format!(
                        "Text does not contain {:?}. Got: {:?}",
                        s,
                        truncate(&output.final_text, 200)
                    ))
                },
                is_security,
                category,
            },

            Assertion::ExpectTextNotContains(s) => AssertionResult {
                description: format!("expect text not contains {:?}", s),
                passed: !output.final_text.contains(s.as_str()),
                message: if !output.final_text.contains(s.as_str()) {
                    None
                } else {
                    Some(format!("Text contains forbidden substring {:?}", s))
                },
                is_security,
                category,
            },

            Assertion::ExpectTurns(range) => {
                let count = output.turns.len();
                AssertionResult {
                    description: format!("expect turns in {:?}", range),
                    passed: range.contains(&count),
                    message: if range.contains(&count) {
                        None
                    } else {
                        Some(format!("Turn count {} not in range {:?}", count, range))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::ExpectToolsWithinAllowlist => {
                let violations: Vec<_> = tools_called
                    .iter()
                    .filter(|t| !available_tools.contains(t))
                    .collect();
                AssertionResult {
                    description: "expect tools within allowlist".into(),
                    passed: violations.is_empty(),
                    message: if violations.is_empty() {
                        None
                    } else {
                        Some(format!(
                            "Tools called outside allowlist: {:?} (allowed: {:?})",
                            violations, available_tools
                        ))
                    },
                    is_security: true,
                    category,
                }
            }

            Assertion::ExpectNoError => AssertionResult {
                description: "expect no error".into(),
                passed: output.error.is_none(),
                message: output
                    .error
                    .as_ref()
                    .map(|e| format!("Agent returned error: {}", e)),
                is_security,
                category,
            },

            Assertion::ExpectToolArgs(tool, expected) => {
                let calls = output.tool_calls_by_name(tool);
                if calls.is_empty() {
                    return AssertionResult {
                        description: format!("expect tool args for {:?}", tool),
                        passed: false,
                        message: Some(format!("Tool {:?} was never called", tool)),
                        is_security,
                        category,
                    };
                }
                let matched = calls.iter().any(|tc| tc.arguments == *expected);
                AssertionResult {
                    description: format!("expect tool args for {:?}", tool),
                    passed: matched,
                    message: if matched {
                        None
                    } else {
                        Some(format!(
                            "No call to {:?} matched exact args {:?}. Got: {:?}",
                            tool,
                            expected,
                            calls.iter().map(|tc| &tc.arguments).collect::<Vec<_>>()
                        ))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::ExpectToolArgsContain(tool, partial) => {
                let calls = output.tool_calls_by_name(tool);
                if calls.is_empty() {
                    return AssertionResult {
                        description: format!("expect tool args contain for {:?}", tool),
                        passed: false,
                        message: Some(format!("Tool {:?} was never called", tool)),
                        is_security,
                        category,
                    };
                }
                let matched = calls.iter().any(|tc| json_contains(&tc.arguments, partial));
                AssertionResult {
                    description: format!("expect tool args contain for {:?}", tool),
                    passed: matched,
                    message: if matched {
                        None
                    } else {
                        Some(format!(
                            "No call to {:?} contains {:?}. Got: {:?}",
                            tool,
                            partial,
                            calls.iter().map(|tc| &tc.arguments).collect::<Vec<_>>()
                        ))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::ExpectToolArg(tool, param, value) => {
                let calls = output.tool_calls_by_name(tool);
                if calls.is_empty() {
                    return AssertionResult {
                        description: format!("expect tool arg {:?}.{:?}", tool, param),
                        passed: false,
                        message: Some(format!("Tool {:?} was never called", tool)),
                        is_security,
                        category,
                    };
                }
                let matched = calls
                    .iter()
                    .any(|tc| tc.arguments.get(param.as_str()) == Some(value));
                AssertionResult {
                    description: format!("expect tool arg {:?}.{:?} = {:?}", tool, param, value),
                    passed: matched,
                    message: if matched {
                        None
                    } else {
                        Some(format!(
                            "No call to {:?} has {:?} = {:?}",
                            tool, param, value
                        ))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::ExpectToolArgExists(tool, param) => {
                let calls = output.tool_calls_by_name(tool);
                if calls.is_empty() {
                    return AssertionResult {
                        description: format!("expect tool arg exists {:?}.{:?}", tool, param),
                        passed: false,
                        message: Some(format!("Tool {:?} was never called", tool)),
                        is_security,
                        category,
                    };
                }
                let matched = calls
                    .iter()
                    .any(|tc| tc.arguments.get(param.as_str()).is_some());
                AssertionResult {
                    description: format!("expect tool arg exists {:?}.{:?}", tool, param),
                    passed: matched,
                    message: if matched {
                        None
                    } else {
                        Some(format!("No call to {:?} has argument {:?}", tool, param))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::ExpectToolCallCount(tool, expected) => {
                let count = output.tool_calls_by_name(tool).len();
                AssertionResult {
                    description: format!("expect {:?} called {} times", tool, expected),
                    passed: count == *expected,
                    message: if count == *expected {
                        None
                    } else {
                        Some(format!(
                            "Expected {:?} called {} times, got {}",
                            tool, expected, count
                        ))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::ExpectToolCallOrder(order) => {
                let all_calls: Vec<&str> = output
                    .all_tool_calls()
                    .iter()
                    .map(|tc| tc.name.as_str())
                    .collect();
                let mut idx = 0;
                for call in &all_calls {
                    if idx < order.len() && *call == order[idx] {
                        idx += 1;
                    }
                }
                let passed = idx == order.len();
                AssertionResult {
                    description: format!("expect tool call order {:?}", order),
                    passed,
                    message: if passed {
                        None
                    } else {
                        Some(format!(
                            "Expected order {:?}, got calls {:?}",
                            order, all_calls
                        ))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::ExpectToolOnTurn(turn_idx, tool) => {
                let passed = output
                    .turns
                    .get(*turn_idx)
                    .map(|t| t.tool_calls.iter().any(|tc| tc.name == *tool))
                    .unwrap_or(false);
                AssertionResult {
                    description: format!("expect {:?} on turn {}", tool, turn_idx),
                    passed,
                    message: if passed {
                        None
                    } else {
                        let turn_tools: Vec<Vec<&str>> = output
                            .turns
                            .iter()
                            .map(|t| t.tool_calls.iter().map(|tc| tc.name.as_str()).collect())
                            .collect();
                        Some(format!(
                            "Expected {:?} on turn {}, tools by turn: {:?}",
                            tool, turn_idx, turn_tools
                        ))
                    },
                    is_security,
                    category,
                }
            }

            // --- Multi-turn assertions ---
            Assertion::ExpectToolsInTurnRange(range, tools) => {
                let found = multi_turn::tools_in_range(output, range);
                let missing: Vec<_> = tools.iter().filter(|t| !found.contains(t)).collect();
                AssertionResult {
                    description: format!("expect tools {:?} in turn range {:?}", tools, range),
                    passed: missing.is_empty(),
                    message: if missing.is_empty() {
                        None
                    } else {
                        Some(format!(
                            "Missing tools {:?} in turn range {:?}. Found: {:?}",
                            missing, range, found
                        ))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::ForbidToolsInTurnRange(range, tools) => {
                let found = multi_turn::tools_in_range(output, range);
                let violations: Vec<_> = tools.iter().filter(|t| found.contains(t)).collect();
                AssertionResult {
                    description: format!("forbid tools {:?} in turn range {:?}", tools, range),
                    passed: violations.is_empty(),
                    message: if violations.is_empty() {
                        None
                    } else {
                        Some(format!(
                            "Forbidden tools {:?} found in turn range {:?}",
                            violations, range
                        ))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::ExpectFinalTool(tool) => {
                let passed = output
                    .turns
                    .last()
                    .map(|t| t.tool_calls.iter().any(|tc| tc.name == *tool))
                    .unwrap_or(false);
                AssertionResult {
                    description: format!("expect final tool {:?}", tool),
                    passed,
                    message: if passed {
                        None
                    } else {
                        let last_tools: Vec<&str> = output
                            .turns
                            .last()
                            .map(|t| t.tool_calls.iter().map(|tc| tc.name.as_str()).collect())
                            .unwrap_or_default();
                        Some(format!(
                            "Expected {:?} on final turn, got tools: {:?}",
                            tool, last_tools
                        ))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::ExpectFinalToolArg(tool, param, value) => {
                let passed = output
                    .turns
                    .last()
                    .map(|t| {
                        t.tool_calls.iter().any(|tc| {
                            tc.name == *tool && tc.arguments.get(param.as_str()) == Some(value)
                        })
                    })
                    .unwrap_or(false);
                AssertionResult {
                    description: format!(
                        "expect final tool arg {:?}.{:?} = {:?}",
                        tool, param, value
                    ),
                    passed,
                    message: if passed {
                        None
                    } else {
                        let last_calls: Vec<String> = output
                            .turns
                            .last()
                            .map(|t| {
                                t.tool_calls
                                    .iter()
                                    .map(|tc| format!("{}({})", tc.name, tc.arguments))
                                    .collect()
                            })
                            .unwrap_or_default();
                        Some(format!(
                            "Expected {:?}.{:?} = {:?} on final turn. Last turn calls: {:?}",
                            tool, param, value, last_calls
                        ))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::ExpectGatheringBeforeAction(gather_tools, action_tools) => {
                let gather_strs: Vec<String> = gather_tools.clone();
                let action_strs: Vec<String> = action_tools.clone();
                let first_action = multi_turn::first_turn_with_tools(output, &action_strs);
                // Check that at least one gather tool was called before any action tool.
                let first_gather = multi_turn::first_turn_with_tools(output, &gather_strs);
                let passed = match (first_gather, first_action) {
                    (Some(g), Some(a)) => g < a, // gathered before acting
                    (Some(_), None) => true,     // gathered but never acted — valid
                    (None, None) => true,        // neither happened — nothing violated
                    (None, Some(_)) => false,    // acted without gathering first
                };
                AssertionResult {
                    description: format!(
                        "expect gathering {:?} before action {:?}",
                        gather_tools, action_tools
                    ),
                    passed,
                    message: if passed {
                        None
                    } else {
                        Some(format!(
                            "Gathering tools {:?} (first at turn {:?}) should appear before action tools {:?} (first at turn {:?})",
                            gather_tools, first_gather, action_tools, first_action
                        ))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::ExpectToolOnlyOnFinalTurn(tool) => {
                let final_idx = output.turns.len().saturating_sub(1);
                let on_final = output
                    .turns
                    .last()
                    .map(|t| t.tool_calls.iter().any(|tc| tc.name == *tool))
                    .unwrap_or(false);
                let on_other = output.turns.iter().any(|t| {
                    t.index != final_idx && t.tool_calls.iter().any(|tc| tc.name == *tool)
                });
                let passed = on_final && !on_other;
                AssertionResult {
                    description: format!("expect {:?} only on final turn", tool),
                    passed,
                    message: if passed {
                        None
                    } else if !on_final {
                        Some(format!("{:?} not found on final turn", tool))
                    } else {
                        let other_turns: Vec<usize> = output
                            .turns
                            .iter()
                            .filter(|t| {
                                t.index != final_idx
                                    && t.tool_calls.iter().any(|tc| tc.name == *tool)
                            })
                            .map(|t| t.index)
                            .collect();
                        Some(format!(
                            "{:?} also found on non-final turns: {:?}",
                            tool, other_turns
                        ))
                    },
                    is_security,
                    category,
                }
            }

            Assertion::Custom(f) => match f(output) {
                Ok(()) => AssertionResult {
                    description: "custom assertion".into(),
                    passed: true,
                    message: None,
                    is_security,
                    category,
                },
                Err(msg) => AssertionResult {
                    description: "custom assertion".into(),
                    passed: false,
                    message: Some(msg),
                    is_security,
                    category,
                },
            },
        }
    }
}

/// Check if `haystack` is a superset of `needle` (partial JSON match).
fn json_contains(haystack: &serde_json::Value, needle: &serde_json::Value) -> bool {
    match (haystack, needle) {
        (serde_json::Value::Object(h), serde_json::Value::Object(n)) => n
            .iter()
            .all(|(k, v)| h.get(k).map_or(false, |hv| json_contains(hv, v))),
        (serde_json::Value::Array(h), serde_json::Value::Array(n)) => {
            n.len() == h.len() && n.iter().zip(h.iter()).all(|(nv, hv)| json_contains(hv, nv))
        }
        _ => haystack == needle,
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}
