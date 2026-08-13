use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A parameter definition for a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamDef {
    pub name: String,
    #[serde(rename = "type")]
    pub param_type: String,
    pub description: String,
    pub required: bool,
}

/// A tool definition parsed from a markdown file with YAML frontmatter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ParamDef>,
}

impl ToolDef {
    /// Parse a tool definition from markdown content with YAML-like frontmatter.
    ///
    /// Expected format:
    /// ```text
    /// ---
    /// name: toolName
    /// description: What the tool does
    /// parameters:
    ///   - name: paramName
    ///     type: string
    ///     description: What the param is
    ///     required: true
    /// ---
    /// Optional body (ignored for now)
    /// ```
    pub fn from_markdown(content: &str) -> Result<Self, String> {
        let content = content.trim();
        if !content.starts_with("---") {
            return Err("Tool markdown must start with --- frontmatter delimiter".into());
        }

        let after_first = &content[3..];
        let end_idx = after_first
            .find("---")
            .ok_or("Missing closing --- frontmatter delimiter")?;
        let frontmatter = after_first[..end_idx].trim();

        parse_tool_frontmatter(frontmatter)
    }

    /// Load a tool definition from a markdown file.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        Self::from_markdown(&content)
    }

    /// Convert to OpenAI-compatible tool JSON.
    pub fn to_openai_json(&self) -> serde_json::Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &self.parameters {
            properties.insert(
                param.name.clone(),
                serde_json::json!({
                    "type": param.param_type,
                    "description": param.description
                }),
            );
            if param.required {
                required.push(serde_json::Value::String(param.name.clone()));
            }
        }

        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        })
    }
}

/// A collection of tool definitions.
#[derive(Debug, Clone)]
pub struct Toolkit {
    pub tools: Vec<ToolDef>,
    tools_by_name: HashMap<String, usize>,
}

impl Toolkit {
    /// Create a toolkit from a list of tool definitions.
    pub fn new(tools: Vec<ToolDef>) -> Self {
        let tools_by_name = tools
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        Self {
            tools,
            tools_by_name,
        }
    }

    /// Load all `.md` files from a directory as tool definitions.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let mut tools = Vec::new();
        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("Failed to read directory {}: {}", dir.display(), e))?;

        let mut paths: Vec<_> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "md"))
            .collect();
        paths.sort();

        for path in paths {
            tools.push(ToolDef::from_file(&path)?);
        }

        Ok(Self::new(tools))
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<&ToolDef> {
        self.tools_by_name.get(name).map(|&i| &self.tools[i])
    }

    /// Get all tool names.
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name.clone()).collect()
    }

    /// Convert all tools to OpenAI-compatible JSON array.
    pub fn to_openai_json(&self) -> Vec<serde_json::Value> {
        self.tools.iter().map(|t| t.to_openai_json()).collect()
    }

    /// Generate a human-readable tool listing for embedding in prompts.
    pub fn to_prompt_listing(&self) -> String {
        let mut out = String::new();
        for tool in &self.tools {
            out.push_str(&format!("### {}\n", tool.name));
            out.push_str(&format!("{}\n", tool.description));
            if !tool.parameters.is_empty() {
                out.push_str("Parameters:\n");
                for p in &tool.parameters {
                    let req = if p.required { " (required)" } else { "" };
                    out.push_str(&format!(
                        "  - `{}` ({}): {}{}\n",
                        p.name, p.param_type, p.description, req
                    ));
                }
            }
            out.push('\n');
        }
        out
    }
}

/// A prompt template that supports `{{tools}}` placeholder substitution.
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    pub template: String,
}

impl PromptTemplate {
    /// Create from a template string.
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
        }
    }

    /// Load from a file.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        Ok(Self::new(content))
    }

    /// Render the template, replacing `{{tools}}` with the toolkit listing.
    pub fn render(&self, toolkit: &Toolkit) -> String {
        self.template
            .replace("{{tools}}", &toolkit.to_prompt_listing())
    }

    /// Render with a custom set of variable replacements.
    pub fn render_with(&self, vars: &HashMap<String, String>) -> String {
        let mut result = self.template.clone();
        for (key, value) in vars {
            result = result.replace(&format!("{{{{{}}}}}", key), value);
        }
        result
    }
}

/// Simple YAML-like frontmatter parser (no external YAML dependency).
fn parse_tool_frontmatter(frontmatter: &str) -> Result<ToolDef, String> {
    let mut name = String::new();
    let mut description = String::new();
    let mut parameters = Vec::new();

    let mut in_parameters = false;
    let mut current_param: Option<ParamBuilder> = None;

    for line in frontmatter.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Top-level keys
        if !line.starts_with(' ') && !line.starts_with('\t') {
            // Flush any pending param
            if let Some(pb) = current_param.take() {
                parameters.push(pb.build()?);
            }

            if let Some(val) = trimmed.strip_prefix("name:") {
                name = val.trim().to_string();
                in_parameters = false;
            } else if let Some(val) = trimmed.strip_prefix("description:") {
                description = val.trim().to_string();
                in_parameters = false;
            } else if trimmed == "parameters:" {
                in_parameters = true;
            }
            continue;
        }

        if !in_parameters {
            continue;
        }

        // Parameter list items
        let stripped = trimmed.trim_start_matches('-').trim();
        if trimmed.starts_with('-') {
            // New parameter entry
            if let Some(pb) = current_param.take() {
                parameters.push(pb.build()?);
            }
            let mut pb = ParamBuilder::default();
            if let Some(val) = stripped.strip_prefix("name:") {
                pb.name = Some(val.trim().to_string());
            }
            current_param = Some(pb);
        } else if let Some(ref mut pb) = current_param {
            // Continuation of current parameter
            if let Some(val) = stripped.strip_prefix("name:") {
                pb.name = Some(val.trim().to_string());
            } else if let Some(val) = stripped.strip_prefix("type:") {
                pb.param_type = Some(val.trim().to_string());
            } else if let Some(val) = stripped.strip_prefix("description:") {
                pb.description = Some(val.trim().to_string());
            } else if let Some(val) = stripped.strip_prefix("required:") {
                pb.required = Some(val.trim() == "true");
            }
        }
    }

    // Flush last param
    if let Some(pb) = current_param.take() {
        parameters.push(pb.build()?);
    }

    if name.is_empty() {
        return Err("Tool frontmatter missing 'name' field".into());
    }
    if description.is_empty() {
        return Err("Tool frontmatter missing 'description' field".into());
    }

    Ok(ToolDef {
        name,
        description,
        parameters,
    })
}

#[derive(Default)]
struct ParamBuilder {
    name: Option<String>,
    param_type: Option<String>,
    description: Option<String>,
    required: Option<bool>,
}

impl ParamBuilder {
    fn build(self) -> Result<ParamDef, String> {
        Ok(ParamDef {
            name: self.name.ok_or("Parameter missing 'name'")?,
            param_type: self.param_type.unwrap_or_else(|| "string".into()),
            description: self.description.unwrap_or_default(),
            required: self.required.unwrap_or(false),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tool_from_markdown() {
        let md = r#"---
name: getWeather
description: Get current weather for a location
parameters:
  - name: location
    type: string
    description: The city name
    required: true
---
# getWeather
Extra docs here.
"#;
        let tool = ToolDef::from_markdown(md).unwrap();
        assert_eq!(tool.name, "getWeather");
        assert_eq!(tool.parameters.len(), 1);
        assert_eq!(tool.parameters[0].name, "location");
        assert!(tool.parameters[0].required);
    }

    #[test]
    fn prompt_template_renders_tools() {
        let toolkit = Toolkit::new(vec![ToolDef {
            name: "myTool".into(),
            description: "Does stuff".into(),
            parameters: vec![],
        }]);
        let tpl = PromptTemplate::new("You have these tools:\n{{tools}}\nUse them wisely.");
        let rendered = tpl.render(&toolkit);
        assert!(rendered.contains("myTool"));
        assert!(rendered.contains("Does stuff"));
        assert!(!rendered.contains("{{tools}}"));
    }
}
