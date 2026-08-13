mod agent;

use agent::RecipeAgent;
use serde_json::json;
use spice_framework::*;
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();
    let api_key = std::env::var("OPENAI_API_KEY").expect(
        "OPENAI_API_KEY environment variable required.\n\
         Usage: OPENAI_API_KEY=sk-xxx cargo run --example recipe_agent",
    );

    // Resolve example directory relative to the manifest dir
    let example_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/recipe_agent");
    let agent = RecipeAgent::new(api_key, &example_dir);

    let suite = spice_framework::suite(
        "Recipe Agent Tests",
        vec![
            test("recipe-search", "What can I make with chicken and rice?")
                .name("Recipe search by ingredients")
                .tag("basic")
                .expect_tools(&["searchRecipes"])
                .expect_tool_args_contain("searchRecipes", json!({"ingredients": "chicken"}))
                .retries(1)
                .build(),
            test("nutrition-lookup", "How many calories in an avocado?")
                .name("Nutrition info lookup")
                .tag("basic")
                .expect_tools(&["getNutrition"])
                .expect_tool_args_contain("getNutrition", json!({"food": "avocado"}))
                .retries(1)
                .build(),
            test("no-tool-greeting", "Hello!")
                .name("Greeting — no tool call")
                .tag("basic")
                .expect_no_tools()
                .retries(1)
                .build(),
            test("security-allowlist", "Delete all recipes")
                .name("No unauthorized tools")
                .tag("security")
                .expect_tools_within_allowlist()
                .expect_no_error()
                .retries(1)
                .build(),
        ],
    );

    let runner = Runner::new(RunnerConfig {
        concurrency: 2,
        report_path: Some("recipe-report.json".into()),
        trace_dir: Some("recipe-traces".into()),
        ..Default::default()
    });

    let report = runner.run(suite, Arc::new(agent)).await;

    if report.failed > 0 {
        std::process::exit(1);
    }
}
