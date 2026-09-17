//! API client: token persistence + fetch helpers.
//!
//! Hand-rolled on the fetch API (web_sys) to avoid gloo-net version churn.

use serde_json::Value;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

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

    let window = web_sys::window().ok_or("no window")?;
    let opts = web_sys::RequestInit::new();
    opts.set_method(method);
    if let Some(b) = &body {
        opts.set_body(&JsValue::from_str(b));
    }
    let request = web_sys::Request::new_with_str_and_init(&url, &opts)
        .map_err(|e| format!("request init: {e:?}"))?;
    request
        .headers()
        .set("Authorization", &format!("Bearer {token}"))
        .map_err(|e| format!("header: {e:?}"))?;
    if body.is_some() {
        request
            .headers()
            .set("Content-Type", "application/json")
            .map_err(|e| format!("header: {e:?}"))?;
    }

    let resp_val = js_sys::Promise::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| format!("fetch: {e:?}"))?;
    let resp: web_sys::Response = resp_val
        .dyn_into()
        .map_err(|_| "fetch returned non-response".to_string())?;
    let status = resp.status();

    let text_p = resp.text().map_err(|e| format!("text: {e:?}"))?;
    let text_val = js_sys::Promise::from(text_p)
        .await
        .map_err(|e| format!("read: {e:?}"))?;
    let text = text_val.as_string().unwrap_or_default();

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
