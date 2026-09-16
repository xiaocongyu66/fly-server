//! REST API: OpenAI-style routes over the session manager.

pub mod http;

use std::collections::HashMap;
use std::sync::{Arc, mpsc};

use crate::error::ApiError;
use crate::session::SessionManager;
use crate::substrate::Substrate;
use crate::types::*;
use http::{Body, HttpRequest, HttpResponse};

pub fn run_server(
    substrate: Arc<Substrate>,
    substrate_id: String,
    port: u16,
    engine_cfg: crate::engine::EngineConfig,
) -> std::io::Result<()> {
    let mgr = Arc::new(SessionManager::new(substrate, substrate_id, engine_cfg));
    http::serve(port, move |req| route(&mgr, req))
}

fn route(mgr: &SessionManager, req: &HttpRequest) -> HttpResponse {
    let segs: Vec<&str> = req.path.trim_matches('/').split('/').filter(|s| !s.is_empty()).collect();
    match (req.method.as_str(), segs.as_slice()) {
        ("GET", ["health"]) => HttpResponse::json(200, format!(r#"{{"ok":true,"version":"{}"}}"#, crate::VERSION)),

        ("GET", ["v1", "models"]) => json_ok(mgr.models()),

        ("POST", ["v1", "sessions"]) => with_body(req, |b: CreateSessionRequest| mgr.create(b)),
        ("POST", ["v1", "sessions", "fork"]) => with_body(req, |b: ForkRequest| mgr.fork(b)),

        ("GET", ["v1", "sessions", id]) => match mgr.get(id) {
            Ok(v) => json_ok(v),
            Err(e) => json_err(&e),
        },
        ("PATCH", ["v1", "sessions", id]) => with_body(req, |b: UpdateSessionRequest| mgr.update(id, b)),
        ("DELETE", ["v1", "sessions", id]) => match mgr.delete(id) {
            Ok(()) => json_ok(serde_json::json!({"deleted": true, "id": id})),
            Err(e) => json_err(&e),
        },
        ("POST", ["v1", "sessions", id, "observe"]) => with_body(req, |b: ObserveRequest| mgr.observe(id, b)),
        ("POST", ["v1", "sessions", id, "step"]) => {
            let parsed: Result<StepRequest, HttpResponse> = if req.body.is_empty() {
                Ok(StepRequest { steps: 1 })
            } else {
                parse_body(req)
            };
            match parsed {
                Ok(b) => match mgr.step(id, b) {
                    Ok(v) => json_ok(v),
                    Err(e) => json_err(&e),
                },
                Err(resp) => resp,
            }
        }
        ("GET", ["v1", "sessions", id, "items"]) => {
            let q = ListItemsQuery {
                limit: req.query.get("limit").and_then(|v| v.parse().ok()),
                after: req.query.get("after").cloned(),
                order: req.query.get("order").cloned(),
            };
            match mgr.list_items(id, q) {
                Ok(v) => json_ok(v),
                Err(e) => json_err(&e),
            }
        }
        ("GET", ["v1", "sessions", id, "activity"]) => match mgr.subscribe(id) {
            Ok(rx) => sse_bridge(rx),
            Err(e) => json_err(&e),
        },

        _ => HttpResponse::not_found_json(),
    }
}

fn json_ok<T: serde::Serialize>(v: T) -> HttpResponse {
    match serde_json::to_string(&v) {
        Ok(s) => HttpResponse::json(200, s),
        Err(e) => json_err(&ApiError::internal(e.to_string())),
    }
}

fn json_err(e: &ApiError) -> HttpResponse {
    HttpResponse::json(e.http_status(), e.to_json())
}

fn parse_body<T: serde::de::DeserializeOwned>(req: &HttpRequest) -> Result<T, HttpResponse> {
    serde_json::from_slice(&req.body)
        .map_err(|e| json_err(&ApiError::invalid_request("invalid_json", e.to_string(), None)))
}

fn with_body<T, R>(req: &HttpRequest, f: impl FnOnce(T) -> Result<R, ApiError>) -> HttpResponse
where
    T: serde::de::DeserializeOwned,
    R: serde::Serialize,
{
    match parse_body::<T>(req) {
        Ok(b) => match f(b) {
            Ok(r) => json_ok(r),
            Err(e) => json_err(&e),
        },
        Err(resp) => resp,
    }
}

fn sse_bridge(rx: mpsc::Receiver<ActivityEvent>) -> HttpResponse {
    let (tx, out_rx) = mpsc::channel();
    std::thread::spawn(move || {
        for ev in rx {
            if let Ok(s) = serde_json::to_string(&ev) {
                if tx.send(format!("data: {s}\n\n")).is_err() {
                    break;
                }
            }
        }
    });
    HttpResponse::sse(out_rx)
}

#[allow(dead_code)]
fn unused(_: HashMap<String, String>, _: Body) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::{FlybinHeader, StringTables};

    fn mini() -> Arc<Substrate> {
        Arc::new(Substrate {
            header: FlybinHeader {
                format_version: 1,
                n_neurons: 2,
                n_edges: 1,
                source: "test".into(),
                string_tables: StringTables {
                    regions: vec!["".into(), "brain_R".into(), "VNC_T1_R".into()],
                    cell_types: vec!["".into()],
                    nt_types: vec!["".into(), "GLUT".into()],
                },
            },
            indptr: vec![0, 1, 1],
            indices: vec![1],
            weights: crate::substrate::flybin::Weights::F32(vec![50.0]),
            root_ids: vec![100, 200],
            region: vec![1, 2],
            cell_type: vec![0, 0],
            nt_type: vec![1, 1],
        })
    }

    fn req(method: &str, path: &str, body: &str) -> HttpRequest {
        HttpRequest {
            method: method.into(),
            path: path.into(),
            query: HashMap::new(),
            headers: Vec::new(),
            body: body.as_bytes().to_vec(),
        }
    }

    #[test]
    fn health_and_models() {
        let mgr = SessionManager::new(mini(), "test-substrate".into(), crate::engine::EngineConfig::default());
        let r = route(&mgr, &req("GET", "/health", ""));
        assert_eq!(r.status, 200);
        let r = route(&mgr, &req("GET", "/v1/models", ""));
        let Body::Bytes(b) = r.body else { panic!() };
        let s = String::from_utf8(b).unwrap();
        assert!(s.contains("test-substrate"));
    }

    #[test]
    fn create_and_step() {
        let mgr = SessionManager::new(mini(), "test-substrate".into(), crate::engine::EngineConfig::default());
        let r = route(&mgr, &req("POST", "/v1/sessions", r#"{"substrate":"test-substrate"}"#));
        assert_eq!(r.status, 200);
        let Body::Bytes(b) = r.body else { panic!() };
        let obj: SessionObject = serde_json::from_slice(&b).unwrap();
        assert!(obj.id.starts_with("sess_"));

        let r = route(&mgr, &req("POST", &format!("/v1/sessions/{}/step", obj.id), ""));
        assert_eq!(r.status, 200);

        let r = route(&mgr, &req("GET", "/nope", ""));
        assert_eq!(r.status, 404);
    }
}
