//! REST API: OpenAI-style routes over the session manager.
//!
//! Auth model: `/health`, the admin SPA, and `POST /v1/admin/login` are
//! public; everything under `/v1/*` requires a Bearer token — either the
//! admin token (login) or an API key (`fly_sk_...`). Per-key usage is
//! recorded for billing. SSE activity also accepts `?token=` because
//! EventSource cannot set headers.

pub mod http;

use std::collections::HashMap;
use std::sync::{mpsc, Arc};

use crate::admin::AdminStore;
use crate::error::ApiError;
use crate::session::SessionManager;
use crate::substrate::Substrate;
use crate::types::*;
use http::{Body, HttpRequest, HttpResponse};

#[derive(serde::Deserialize)]
struct LoginRequest {
    username: String,
    password: String,
}

pub fn run_server(
    substrate: Arc<Substrate>,
    substrate_id: String,
    port: u16,
    engine_cfg: crate::engine::EngineConfig,
    admin_dist: Option<std::path::PathBuf>,
    admin: Arc<AdminStore>,
    datasets: Arc<crate::datasets::DatasetStore>,
    host: &str,
) -> std::io::Result<()> {
    let mgr = Arc::new(SessionManager::new(substrate, substrate_id, engine_cfg));
    http::serve(host, port, move |req| {
        if let Some(dist) = &admin_dist {
            if req.method == "GET" && !req.path.starts_with("/v1/") && req.path != "/health" {
                if let Some(resp) = http::serve_static(dist, &req.path) {
                    return resp;
                }
            }
        }
        route(&mgr, &admin, &datasets, req)
    })
}

fn bearer(req: &HttpRequest) -> Option<&str> {
    // header first, then query token (SSE EventSource cannot set headers)
    if let Some((_, v)) = req.headers.iter().find(|(k, _)| k == "authorization") {
        if let Some(t) = v
            .strip_prefix("Bearer ")
            .or(Some(v.as_str()).filter(|s| !s.is_empty()))
        {
            return Some(t.trim());
        }
    }
    req.query.get("token").map(String::as_str)
}

fn route(
    mgr: &SessionManager,
    admin: &AdminStore,
    datasets: &Arc<crate::datasets::DatasetStore>,
    req: &HttpRequest,
) -> HttpResponse {
    let segs: Vec<&str> = req
        .path
        .trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();

    // public endpoints
    match (req.method.as_str(), segs.as_slice()) {
        ("GET", ["health"]) => {
            return HttpResponse::json(
                200,
                format!(r#"{{"ok":true,"version":"{}"}}"#, crate::VERSION),
            );
        }
        ("POST", ["v1", "admin", "login"]) => {
            return with_body(req, |b: LoginRequest| {
                admin
                    .login(&b.username, &b.password)
                    .map(|token| serde_json::json!({"token": token, "role": "admin"}))
                    .ok_or_else(|| {
                        ApiError::invalid_request(
                            "invalid_credentials",
                            "wrong username or password",
                            None,
                        )
                    })
            });
        }
        _ => {}
    }

    // authenticated endpoints
    let authed = bearer(req)
        .and_then(|t| admin.authorize(t))
        .ok_or_else(|| HttpResponse::json(
            401,
            r#"{"error":{"type":"authentication_error","code":"invalid_api_key","message":"Missing or invalid bearer token"}}"#.to_string(),
        ));
    let (role, key_id) = match authed {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let resp = route_authed(mgr, admin, datasets, role.as_str(), &key_id, req, &segs);
    resp
}

fn route_authed(
    mgr: &SessionManager,
    admin: &AdminStore,
    datasets: &Arc<crate::datasets::DatasetStore>,
    role: &str,
    key_id: &Option<String>,
    req: &HttpRequest,
    segs: &[&str],
) -> HttpResponse {
    let is_admin = role == "admin";

    // admin-only management endpoints
    if segs.first() == Some(&"v1") && segs.get(1) == Some(&"admin") {
        if !is_admin {
            return json_err(&ApiError::invalid_request(
                "admin_required",
                "admin token required",
                None,
            ));
        }
        let (m2, a2, a3, a4) = (
            req.method.as_str(),
            segs.get(2).copied(),
            segs.get(3).copied(),
            segs.get(4).copied(),
        );
        return match (m2, a2, a3, a4) {
            ("GET", Some("keys"), None, _) => json_ok(admin.list_keys()),
            ("POST", Some("keys"), None, _) => match parse_body::<serde_json::Value>(req) {
                Ok(b) => {
                    let name = b
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("default")
                        .to_string();
                    json_ok(admin.create_key(&name))
                }
                Err(resp) => resp,
            },
            ("POST", Some("keys"), Some(id), Some("enable" | "disable")) => {
                let enabled = a3 == Some("enable");
                match admin.set_key_enabled(id, enabled) {
                    true => json_ok(serde_json::json!({"id": id, "enabled": enabled})),
                    false => json_err(&ApiError::not_found(format!("key `{id}` not found"))),
                }
            }
            ("GET", Some("usage"), None, _) => json_ok(serde_json::json!({
                "total": admin.total_usage(),
                "keys": admin.list_keys(),
            })),
            ("GET", Some("datasets"), None, _) => json_ok(datasets.list()),
            ("POST", Some("datasets"), Some(tier), Some("download")) => {
                match datasets.start_download(tier) {
                    Ok(()) => json_ok(serde_json::json!({"tier": tier, "started": true})),
                    Err(e) => json_err(&ApiError::invalid_request("download_error", e, None)),
                }
            }
            ("GET", Some("datasets"), Some("status"), None) => {
                json_ok(serde_json::json!({"tiers": datasets.list()}))
            }
            _ => HttpResponse::not_found_json(),
        };
    }

    // billing: count the request for API keys
    admin.record_usage(key_id.as_deref(), 1, 0, 0);

    match (req.method.as_str(), segs) {
        ("GET", ["v1", "models"]) => json_ok(mgr.models()),

        ("GET", ["v1", "sessions"]) => json_ok(mgr.list()),
        ("POST", ["v1", "sessions"]) => {
            admin.record_usage(key_id.as_deref(), 0, 0, 1);
            with_body(req, |b: CreateSessionRequest| mgr.create(b))
        }
        ("POST", ["v1", "sessions", "fork"]) => with_body(req, |b: ForkRequest| mgr.fork(b)),

        ("GET", ["v1", "sessions", id]) => match mgr.get(id) {
            Ok(v) => json_ok(v),
            Err(e) => json_err(&e),
        },
        ("PATCH", ["v1", "sessions", id]) => {
            with_body(req, |b: UpdateSessionRequest| mgr.update(id, b))
        }
        ("DELETE", ["v1", "sessions", id]) => match mgr.delete(id) {
            Ok(()) => json_ok(serde_json::json!({"deleted": true, "id": id})),
            Err(e) => json_err(&e),
        },
        ("POST", ["v1", "sessions", id, "observe"]) => {
            with_body(req, |b: ObserveRequest| mgr.observe(id, b))
        }
        ("POST", ["v1", "sessions", id, "step"]) => {
            let parsed: Result<StepRequest, HttpResponse> = if req.body.is_empty() {
                Ok(StepRequest { steps: 1 })
            } else {
                parse_body(req)
            };
            match parsed {
                Ok(b) => {
                    let n_steps = b.steps;
                    match mgr.step(id, b) {
                        Ok(v) => {
                            admin.record_usage(key_id.as_deref(), 0, u64::from(n_steps), 0);
                            json_ok(v)
                        }
                        Err(e) => json_err(&e),
                    }
                }
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
    serde_json::from_slice(&req.body).map_err(|e| {
        json_err(&ApiError::invalid_request(
            "invalid_json",
            e.to_string(),
            None,
        ))
    })
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
        let mgr = SessionManager::new(
            mini(),
            "test-substrate".into(),
            crate::engine::EngineConfig::default(),
        );
        let datasets = std::sync::Arc::new(crate::datasets::DatasetStore::new(
            std::path::PathBuf::from("/tmp/agent-datasets"),
        ));
        let admin = crate::admin::AdminStore::new(
            "admin",
            "pw",
            std::path::PathBuf::from("/tmp/test_keys1.json"),
        );
        let r = route(&mgr, &admin, &datasets, &req("GET", "/health", ""));
        assert_eq!(r.status, 200);
        let key = admin.create_key("t");
        let mut req = req("GET", "/v1/models", "");
        req.headers
            .push(("authorization".into(), format!("Bearer {}", key.secret)));
        let r = route(&mgr, &admin, &datasets, &req);
        let Body::Bytes(b) = r.body else { panic!() };
        let s = String::from_utf8(b).unwrap();
        assert!(s.contains("test-substrate"));
    }

    #[test]
    fn create_and_step() {
        let mgr = SessionManager::new(
            mini(),
            "test-substrate".into(),
            crate::engine::EngineConfig::default(),
        );
        let datasets = std::sync::Arc::new(crate::datasets::DatasetStore::new(
            std::path::PathBuf::from("/tmp/agent-datasets"),
        ));
        let admin = crate::admin::AdminStore::new(
            "admin",
            "pw",
            std::path::PathBuf::from("/tmp/test_keys2.json"),
        );
        let key = admin.create_key("t");
        let authed = |method: &str, path: &str, body: &str| {
            let mut rq = req(method, path, body);
            rq.headers
                .push(("authorization".into(), format!("Bearer {}", key.secret)));
            route(&mgr, &admin, &datasets, &rq)
        };
        let r = authed("POST", "/v1/sessions", r#"{"substrate":"test-substrate"}"#);
        assert_eq!(r.status, 200);
        let Body::Bytes(b) = r.body else { panic!() };
        let obj: SessionObject = serde_json::from_slice(&b).unwrap();
        assert!(obj.id.starts_with("sess_"));

        let r = authed("POST", &format!("/v1/sessions/{}/step", obj.id), "");
        assert_eq!(r.status, 200);

        // unauthenticated: 401 (does not leak route existence)
        assert_eq!(
            route(&mgr, &admin, &datasets, &req("GET", "/nope", "")).status,
            401
        );
        let r = authed("GET", "/nope", "");
        assert_eq!(r.status, 404);
    }
}
