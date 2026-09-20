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

/// POST /v1/admin/datasets/{tier}/activate
///
/// Hot-swap to a compiled `.flybin` when one exists; otherwise compile the
/// tier's feather in a background thread and auto-activate when done. The
/// compile slot is claimed atomically, so repeated clicks are idempotent —
/// status flips `downloading → compiling → compiled` in the 1s status poll.
fn activate_tier(
    mgr: &Arc<SessionManager>,
    datasets: &Arc<crate::datasets::DatasetStore>,
    tier: &str,
    substrates_dir: &std::path::Path,
) -> HttpResponse {
    let flybin = substrates_dir.join(format!("{tier}.flybin"));
    if flybin.exists() {
        return match crate::substrate::load(&flybin) {
            Ok(sub) => {
                let cleared = mgr.reload_substrate(Arc::new(sub), tier.to_string());
                json_ok(serde_json::json!({
                    "activated": tier,
                    "sessions_cleared": cleared,
                }))
            }
            Err(e) => json_err(&ApiError::invalid_request(
                "substrate_load_error",
                e.to_string(),
                None,
            )),
        };
    }

    // no compiled substrate yet — locate the feather input. The full tier
    // is a per-synapse partner list (no weight column); both tiers share
    // the same MaleCNS body ids, so standard's annotation files enrich both.
    let (weights_name, pre_col, post_col, weight_col) = match tier {
        "standard" => (
            "connectome_weights.feather",
            "body_pre",
            "body_post",
            Some("weight"),
        ),
        "full" => ("syn_partners.feather", "body_pre", "body_post", None),
        _ => {
            return json_err(&ApiError::invalid_request(
                "unknown_tier",
                format!(
                    "tier `{tier}` cannot be activated (lite uses substrate/lite.flybin/activate)"
                ),
                None,
            ))
        }
    };
    let weights_path = datasets.tier_dir(tier).join(weights_name);
    if !crate::substrate::feather::feather_complete(&weights_path) {
        return json_err(&ApiError::invalid_request(
            "feather_incomplete",
            format!(
                "{} is missing or truncated — download it first",
                weights_path.display()
            ),
            None,
        ));
    }
    if let Err(e) = datasets.claim_compile(tier) {
        return json_ok(serde_json::json!({"tier": tier, "status": e}));
    }

    // Both MaleCNS tiers share the same body ids, so standard's annotation
    // files enrich the full tier too (full's own download has none).
    let anno_path = datasets
        .tier_dir("standard")
        .join("body_annotations.feather");
    let nt_path = datasets.tier_dir("standard").join("body_nt.feather");
    let annotations = crate::substrate::feather::feather_complete(&anno_path).then_some(anno_path);
    let nt = crate::substrate::feather::feather_complete(&nt_path).then_some(nt_path);
    let out_path = substrates_dir.join(format!("{tier}.flybin"));
    let mgr = Arc::clone(mgr);
    let datasets = Arc::clone(datasets);
    let tier_s = tier.to_string();
    datasets.set_hint(tier, "compiling: pass 1/2 (indexing neurons)");
    std::thread::spawn(move || {
        let spec = crate::substrate::MaleCnsSpec {
            weights_path: &weights_path,
            pre_col,
            post_col,
            weight_col,
            annotations_path: annotations.as_deref(),
            nt_path: nt.as_deref(),
        };
        match crate::substrate::compile_malecns(&spec, &out_path, crate::substrate::Quant::F32) {
            Ok(report) => {
                datasets.set_hint(&tier_s, "compiling: loading substrate");
                match crate::substrate::load(&out_path) {
                    Ok(sub) => {
                        // publish the artifact hash: `<tier>.flybin.md5`
                        // sidecar (md5sum format) + status field, so a
                        // model download service can verify the compiled
                        // substrate without recompiling
                        let md5 = crate::datasets::file_md5(&out_path)
                            .unwrap_or_else(|_| "unavailable".into());
                        let sidecar = out_path.with_extension("flybin.md5");
                        if let Some(name) = out_path.file_name() {
                            let _ = std::fs::write(
                                &sidecar,
                                format!("{md5}  {}\n", name.to_string_lossy()),
                            );
                        }
                        mgr.reload_substrate(Arc::new(sub), tier_s.clone());
                        datasets.mark_compiled(&tier_s, &report, md5);
                    }
                    Err(e) => datasets.mark_compile_error(&tier_s, &e.to_string()),
                }
            }
            Err(e) => {
                eprintln!("malecns compile failed for {tier_s}: {e}");
                datasets.mark_compile_error(&tier_s, &e.to_string());
            }
        }
    });
    json_ok(serde_json::json!({"tier": tier, "compiling": true}))
}

#[allow(clippy::too_many_arguments)]
pub fn run_server(
    substrate: Option<Arc<Substrate>>,
    substrate_id: String,
    port: u16,
    mut engine_cfg: crate::engine::EngineConfig,
    admin_dist: Option<std::path::PathBuf>,
    admin: Arc<AdminStore>,
    datasets: Arc<crate::datasets::DatasetStore>,
    llm: Arc<crate::llm::LlmConfig>,
    host: &str,
    data_dir: std::path::PathBuf,
) -> std::io::Result<()> {
    // persist engine settings so restarts keep user choices
    let settings_path = data_dir.join("config.json");
    if settings_path.exists() {
        if let Ok(saved) = std::fs::read_to_string(&settings_path) {
            if let Ok(cfg) = serde_json::from_str::<crate::engine::EngineConfig>(&saved) {
                engine_cfg = cfg;
            }
        }
    }
    let substrates_dir = data_dir.join("substrates");
    let mgr = Arc::new(SessionManager::new(substrate, substrate_id, engine_cfg));
    mgr.start_gc_thread(std::time::Duration::from_secs(5));
    http::serve(host, port, move |req| {
        if let Some(dist) = &admin_dist {
            if req.method == "GET" && !req.path.starts_with("/v1/") && req.path != "/health" {
                if let Some(resp) = http::serve_static(dist, &req.path) {
                    return resp;
                }
            }
        }
        route(
            &mgr,
            &admin,
            &datasets,
            &llm,
            &settings_path,
            &substrates_dir,
            req,
        )
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
    mgr: &Arc<SessionManager>,
    admin: &AdminStore,
    datasets: &Arc<crate::datasets::DatasetStore>,
    llm: &crate::llm::LlmConfig,
    settings_path: &std::path::Path,
    substrates_dir: &std::path::Path,
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

    let resp = route_authed(
        mgr,
        admin,
        datasets,
        llm,
        settings_path,
        substrates_dir,
        role.as_str(),
        &key_id,
        req,
        &segs,
    );
    resp
}

#[allow(clippy::too_many_arguments)]
fn route_authed(
    mgr: &Arc<SessionManager>,
    admin: &AdminStore,
    datasets: &Arc<crate::datasets::DatasetStore>,
    llm: &crate::llm::LlmConfig,
    settings_path: &std::path::Path,
    substrates_dir: &std::path::Path,
    role: &str,
    key_id: &Option<String>,
    req: &HttpRequest,
    segs: &[&str],
) -> HttpResponse {
    let is_admin = role == "admin";
    // settings live at <data_dir>/config.json — derive the data dir from it
    let data_dir = settings_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));

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
                    let rpm = b.get("rpm_limit").and_then(|v| v.as_u64()).unwrap_or(0);
                    let tpm = b.get("tpm_limit").and_then(|v| v.as_u64()).unwrap_or(0);
                    let budget = b.get("tick_budget").and_then(|v| v.as_u64()).unwrap_or(0);
                    json_ok(admin.create_key(&name, rpm, tpm, budget))
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
            ("DELETE", Some("keys"), Some(id), None) => match admin.delete_key(id) {
                true => json_ok(serde_json::json!({"deleted": true, "id": id})),
                false => json_err(&ApiError::not_found(format!("key `{id}` not found"))),
            },
            ("GET", Some("usage"), None, _) => json_ok(serde_json::json!({
                "total": admin.total_usage(),
                "keys": admin.list_keys(),
            })),
            ("GET", Some("datasets"), None, _) => json_ok(datasets.list()),
            ("GET", Some("settings"), None, _) => json_ok(mgr.get_engine_config()),
            ("POST", Some("settings"), None, _) => {
                with_body(req, |b: crate::engine::EngineConfig| {
                    mgr.update_engine_config(&b);
                    // persist for restarts
                    if let Ok(json) = serde_json::to_string_pretty(&b) {
                        let _ = std::fs::write(settings_path, json);
                    }
                    Ok(mgr.get_engine_config())
                })
            }
            ("POST", Some("substrate"), Some(rel_path), Some("activate")) => {
                // hot-swap to a .flybin under data_dir/substrates/
                let rel = rel_path.replace("..", "");
                let full = substrates_dir.join(&rel);
                match crate::substrate::load(&full) {
                    Ok(sub) => {
                        let id = full
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        let cleared = mgr.reload_substrate(std::sync::Arc::new(sub), id.clone());
                        json_ok(serde_json::json!({"activated": id, "sessions_cleared": cleared}))
                    }
                    Err(e) => json_err(&ApiError::not_found(format!("substrate load failed: {e}"))),
                }
            }
            // training plug-ins (.flydelta files): list / enable / delete
            ("GET", Some("training"), None, _) => {
                let at = mgr.active_training.lock().unwrap();
                json_ok(serde_json::json!({
                    "files": crate::train::list_training(data_dir),
                    "enabled": at.files,
                }))
            }
            ("POST", Some("training"), Some(name), Some("enable" | "disable")) => {
                let on = a3 == Some("enable");
                match crate::train::set_training_enabled(mgr, data_dir, name, on) {
                    Ok(()) => {
                        mgr.refresh_overlays();
                        json_ok(serde_json::json!({"name": name, "enabled": on}))
                    }
                    Err(e) => json_err(&e),
                }
            }
            ("DELETE", Some("training"), Some(name), None) => {
                match crate::train::delete_training(mgr, data_dir, name) {
                    Ok(()) => {
                        mgr.refresh_overlays();
                        json_ok(serde_json::json!({"deleted": true, "name": name}))
                    }
                    Err(e) => json_err(&e),
                }
            }
            ("POST", Some("train"), Some("save"), None) => {
                match parse_body::<serde_json::Value>(req) {
                    Ok(b) => {
                        let name = b.get("name").and_then(|v| v.as_str()).unwrap_or("trained");
                        let cfg = mgr.train_job.last_cfg.lock().unwrap().clone();
                        match crate::train::save_training(mgr, data_dir, name, &cfg) {
                            Ok(meta) => json_ok(serde_json::json!({
                                "saved": true,
                                "name": meta.name,
                                "deltas": meta.deltas,
                                "bytes": meta.bytes,
                            })),
                            Err(e) => json_err(&e),
                        }
                    }
                    Err(resp) => resp,
                }
            }
            ("GET", Some("gpu"), None, _) => json_ok(serde_json::json!({
                "available": crate::engine::gpu::probe(),
                "mode": mgr.get_engine_config().use_gpu,
            })),
            ("GET", Some("memory"), None, _) => json_ok(mgr.memory_stats()),
            ("POST", Some("memory"), Some("gc"), None) => json_ok(mgr.force_gc()),
            ("POST", Some("datasets"), Some(tier), Some("download")) => {
                match datasets.start_download(tier) {
                    Ok(()) => json_ok(serde_json::json!({"tier": tier, "started": true})),
                    Err(e) if e.contains("already downloaded") => {
                        // files exist on disk — just refresh status
                        json_ok(serde_json::json!({"tier": tier, "status": "already_downloaded"}))
                    }
                    Err(e) if e.contains("already in flight") => {
                        json_ok(serde_json::json!({"tier": tier, "status": "downloading"}))
                    }
                    Err(e) => json_err(&ApiError::invalid_request("download_error", e, None)),
                }
            }
            ("POST", Some("datasets"), Some(tier), Some("activate")) => {
                self::activate_tier(mgr, datasets, tier, substrates_dir)
            }
            // post-training slot: rollout episodes against the live substrate
            ("POST", Some("train"), None, _) => {
                match parse_body::<crate::train::TrainConfig>(req) {
                    Ok(cfg) => match mgr.start_train(cfg) {
                        Ok(()) => json_ok(serde_json::json!({"started": true})),
                        Err(e) => json_err(&e),
                    },
                    Err(resp) => resp,
                }
            }
            ("GET", Some("train"), Some("status"), None) => json_ok(mgr.train_status()),
            ("POST", Some("train"), Some("stop"), None) => {
                json_ok(serde_json::json!({"stop_requested": mgr.stop_train()}))
            }
            ("GET", Some("datasets"), Some("status"), None) => {
                json_ok(serde_json::json!({"tiers": datasets.list()}))
            }
            _ => HttpResponse::not_found_json(),
        };
    }

    // billing + rate limiting: check RPM/TPM/budget for API keys
    if let Some(kid) = key_id {
        if let Some(limit_err) = admin.check_rate_limit(kid) {
            return json_err(&limit_err);
        }
    }
    admin.record_usage(key_id.as_deref(), 1, 0, 0);

    match (req.method.as_str(), segs) {
        ("GET", ["v1", "models"]) => json_ok(mgr.models()),

        ("POST", ["v1", "query", "llm"]) => match parse_body::<serde_json::Value>(req) {
            Ok(b) => {
                let q = b
                    .get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if q.is_empty() {
                    return json_err(&ApiError::invalid_request(
                        "empty_query",
                        "query is required",
                        Some("query"),
                    ));
                }
                let regions: Vec<String> = mgr.regions().unwrap_or_default();
                match crate::llm::parse_via_llm(llm, &q, &regions) {
                    Ok(sel_json) => json_ok(sel_json),
                    Err(e) => json_err(&ApiError::invalid_request("llm_error", e, None)),
                }
            }
            Err(resp) => resp,
        },

        ("POST", ["v1", "chat", "simulate"]) => match parse_body::<serde_json::Value>(req) {
            Ok(b) => {
                let target = match b
                    .get("target")
                    .cloned()
                    .map(serde_json::from_value::<crate::types::NeuronSelector>)
                    .transpose()
                {
                    Ok(t) => t.unwrap_or_default(),
                    Err(e) => {
                        return json_err(&ApiError::invalid_request(
                            "bad_target",
                            e.to_string(),
                            Some("target"),
                        ))
                    }
                };
                let current = b.get("current").and_then(|v| v.as_f64()).unwrap_or(30.0) as f32;
                let steps = b.get("steps").and_then(|v| v.as_u64()).unwrap_or(100) as u32;
                eprintln!(
                    "[access] POST /v1/chat_simulate target={} current={} steps={}",
                    serde_json::to_string(&target).unwrap_or_default(),
                    current,
                    steps
                );
                match mgr.chat_simulate(&target, current, steps) {
                    Ok(v) => json_ok(v),
                    Err(e) => json_err(&e),
                }
            }
            Err(resp) => resp,
        },

        ("POST", ["v1", "query"]) => match parse_body::<crate::types::NeuronSelector>(req) {
            Ok(b) => match mgr.query(&b) {
                Ok(v) => json_ok(v),
                Err(e) => json_err(&e),
            },
            Err(resp) => resp,
        },
        ("GET", ["v1", "substrate", "graph"]) => {
            let nodes = req
                .query
                .get("nodes")
                .and_then(|v| v.parse().ok())
                .unwrap_or(500)
                .min(5000);
            let edges = req
                .query
                .get("edges")
                .and_then(|v| v.parse().ok())
                .unwrap_or(5000)
                .min(50000);
            match mgr.subgraph(nodes, edges) {
                Ok(v) => json_ok(v),
                Err(e) => json_err(&e),
            }
        }
        ("GET", ["v1", "substrate", "edges"]) => {
            let limit = req
                .query
                .get("limit")
                .and_then(|v| v.parse().ok())
                .unwrap_or(2000);
            // no cap: full connectome (2.7M edges ≈ 64MB GPU line buffer)
            match mgr.sample_edges(limit) {
                Ok(v) => json_ok(v),
                Err(e) => json_err(&e),
            }
        }

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
        ("GET", ["v1", "sessions", id, "ws"]) => {
            // WebSocket upgrade: first HTTP handshake, then a live frame
            // stream of per-tick activity for the session.
            let key = req
                .headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("sec-websocket-key"))
                .map(|(_, v)| v.clone());
            let upgrade = req.headers.iter().any(|(k, v)| {
                k.eq_ignore_ascii_case("upgrade") && v.eq_ignore_ascii_case("websocket")
            });
            match (key, mgr.subscribe(id)) {
                (Some(k), Ok(rx)) if upgrade => {
                    const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
                    let digest = crate::api::http::sha1(format!("{k}{GUID}").as_bytes());
                    let accept = crate::api::http::b64(&digest);
                    let (tx, out) = mpsc::channel();
                    std::thread::spawn(move || {
                        for ev in rx {
                            if serde_json::to_string(&ev)
                                .map(|s| tx.send(s).is_err())
                                .unwrap_or(true)
                            {
                                break;
                            }
                        }
                    });
                    HttpResponse::ws(out, accept)
                }
                (_, Ok(_)) => json_err(&ApiError::invalid_request(
                    "not_upgrade",
                    "expected WebSocket upgrade headers",
                    None,
                )),
                (_, Err(e)) => json_err(&e),
            }
        }

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
        let mgr = std::sync::Arc::new(SessionManager::new(
            Some(mini()),
            "test-substrate".into(),
            crate::engine::EngineConfig::default(),
        ));
        let datasets = std::sync::Arc::new(crate::datasets::DatasetStore::new(
            std::path::PathBuf::from("/tmp/agent-datasets"),
        ));
        let admin = crate::admin::AdminStore::new(
            "admin",
            "pw",
            std::path::PathBuf::from("/tmp/test_keys1.json"),
        );
        let llm = crate::llm::LlmConfig::default();
        let r = route(
            &mgr,
            &admin,
            &datasets,
            &llm,
            std::path::Path::new("/tmp"),
            std::path::Path::new("/tmp/subs"),
            &req("GET", "/health", ""),
        );
        assert_eq!(r.status, 200);
        let key = admin.create_key("t", 0, 0, 0);
        let mut req = req("GET", "/v1/models", "");
        req.headers
            .push(("authorization".into(), format!("Bearer {}", key.secret)));
        let llm = crate::llm::LlmConfig::default();
        let r = route(
            &mgr,
            &admin,
            &datasets,
            &llm,
            std::path::Path::new("/tmp"),
            std::path::Path::new("/tmp/subs"),
            &req,
        );
        let Body::Bytes(b) = r.body else { panic!() };
        let s = String::from_utf8(b).unwrap();
        assert!(s.contains("test-substrate"));
    }

    #[test]
    fn create_and_step() {
        let mgr = std::sync::Arc::new(SessionManager::new(
            Some(mini()),
            "test-substrate".into(),
            crate::engine::EngineConfig::default(),
        ));
        let datasets = std::sync::Arc::new(crate::datasets::DatasetStore::new(
            std::path::PathBuf::from("/tmp/agent-datasets"),
        ));
        let admin = crate::admin::AdminStore::new(
            "admin",
            "pw",
            std::path::PathBuf::from("/tmp/test_keys2.json"),
        );
        let key = admin.create_key("t", 0, 0, 0);
        let llm = crate::llm::LlmConfig::default();
        let authed = |method: &str, path: &str, body: &str| {
            let mut rq = req(method, path, body);
            rq.headers
                .push(("authorization".into(), format!("Bearer {}", key.secret)));
            route(
                &mgr,
                &admin,
                &datasets,
                &llm,
                std::path::Path::new("/tmp"),
                std::path::Path::new("/tmp/subs"),
                &rq,
            )
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
            route(
                &mgr,
                &admin,
                &datasets,
                &llm,
                std::path::Path::new("/tmp"),
                std::path::Path::new("/tmp/subs"),
                &req("GET", "/nope", "")
            )
            .status,
            401
        );
        let r = authed("GET", "/nope", "");
        assert_eq!(r.status, 404);
    }
}
