//! LLM proxy: server-side OpenAI-compatible call for natural-language
//! query parsing. The upstream key lives in server config, never in the
//! browser.

use serde_json::Value;

#[derive(Debug, Clone, Default)]
pub struct LlmConfig {
    /// OpenAI-compatible base URL, e.g. http://localhost:8080/v1
    pub url: Option<String>,
    pub key: Option<String>,
    pub model: String,
}

const PROMPT: &str = "You translate natural-language queries about the Drosophila (fruit fly) brain into a JSON object.\nReturn ONLY a JSON object with this exact shape (no markdown, no explanation):\n{\"region\": string|null, \"cell_type\": string|null, \"nt_type\": string|null, \"limit\": number|null}\nValid nt_type values: GLUT, GABA, ACETYLCHOLINE, DOPAMINE, SEROTONIN, OCTOPAMINE.\nUse null for unspecified fields. limit is a small number when the user says \"first/top N\", otherwise null.";

/// Ask the configured LLM to parse a natural-language query into a selector.
/// Returns the raw selector JSON (region/cell_type/nt_type/limit).
pub fn parse_via_llm(
    cfg: &LlmConfig,
    query: &str,
    known_regions: &[String],
) -> Result<Value, String> {
    let base = cfg
        .url
        .as_deref()
        .ok_or("LLM not configured — start server with --llm-url/--llm-key")?
        .trim_end_matches('/')
        .to_string();
    let full = format!("{base}/chat/completions");
    let regions_list = known_regions
        .iter()
        .take(200)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");

    let body = serde_json::json!({
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": format!("{}\nKnown regions include: {}.", PROMPT, regions_list)},
            {"role": "user", "content": query}
        ],
        "temperature": 0
    });

    let mut req = ureq::post(&full)
        .timeout(std::time::Duration::from_secs(120))
        .set("Content-Type", "application/json");
    if let Some(k) = &cfg.key {
        if !k.is_empty() {
            req = req.set("Authorization", &format!("Bearer {k}"));
        }
    }
    let resp = req
        .send_string(&body.to_string())
        .map_err(|e| format!("LLM call failed: {e}"))?;
    let status = resp.status();
    let text = resp.into_string().unwrap_or_default();
    if status >= 400 {
        return Err(format!(
            "LLM returned [{status}]: {}",
            &text[..text.len().min(200)]
        ));
    }
    let data: Value = serde_json::from_str(&text).map_err(|e| format!("LLM json: {e}"))?;
    let content = data["choices"][0]["message"]["content"]
        .as_str()
        .ok_or("LLM response missing content")?;

    let jm = content
        .match_indices('{')
        .next()
        .map(|(i, _)| &content[i..])
        .and_then(|tail| {
            tail.match_indices('}')
                .next_back()
                .map(|(j, _)| &tail[..=j])
        })
        .ok_or("LLM returned no JSON object")?;
    let parsed: Value = serde_json::from_str(jm).map_err(|e| format!("LLM json parse: {e}"))?;
    Ok(parsed)
}
