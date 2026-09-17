//! API client: token persistence + fetch helpers.

use gloo_net::http::Request;
use serde_json::Value;
use wasm_bindgen::JsValue;

pub fn token() -> Option<String> {
    web_sys::window()?
        .local_storage()
        .ok()
        .flatten()?
        .get_item("fly_token")
        .ok()
        .flatten()
}

pub fn set_token(t: &str) {
    if let Some(ls) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        let _ = ls.set_item("fly_token", t);
    }
}

pub fn clear_token() {
    if let Some(ls) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        let _ = ls.remove_item("fly_token");
    }
}

pub fn base() -> String {
    web_sys::window()
        .and_then(|w| w.location().origin().ok())
        .unwrap_or_else(|| "http://127.0.0.1:8321".into())
}

async fn send(method: &str, path: &str, body: Option<String>) -> Result<Value, String> {
    let url = format!("{}{}", base(), path);
    let token = token().unwrap_or_default();
    let auth = format!("Bearer {token}");
    let req = match method {
        "GET" => Request::get(&url),
        "POST" => Request::post(&url),
        "DELETE" => Request::delete(&url),
        "PATCH" => Request::new(&gloo_net::http::Method::PATCH, &url),
        _ => Request::get(&url),
    }
    .header("Authorization", &auth);
    let req = match body {
        Some(b) => req.header("Content-Type", "application/json").body(b),
        None => req,
    };
    let resp = req.send().await.map_err(|e| format!("network: {e:?}"))?;
    let status = resp.status();
    let text = resp.text().await.map_err(|e| format!("read: {e:?}"))?;
    let v: Value = serde_json::from_str(&text).unwrap_or(Value::String(text.clone()));
    if status >= 400 {
        let msg = v["error"]["message"].as_str().unwrap_or(&text).to_string();
        if status == 401 {
            clear_token();
        }
        return Err(format!("[{status}] {msg}"));
    }
    Ok(v)
}

pub async fn get(path: &str) -> Result<Value, String> {
    send("GET", path, None).await
}

pub async fn post(path: &str, body: Value) -> Result<Value, String> {
    send("POST", path, Some(serde_json::to_string(&body).unwrap_or_default())).await
}

pub async fn del(path: &str) -> Result<Value, String> {
    send("DELETE", path, None).await
}

pub fn js_str(s: &str) -> JsValue {
    JsValue::from_str(s)
}
