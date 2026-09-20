//! Session manager — OpenAI Conversations semantics over a live fly brain.
//!
//! A session owns an Engine (the simulated brain instance), an item log
//! (experiment history with cursor pagination), a usage ledger
//! (ticks_simulated / ticks_reused / snapshot_writes), and an SSE
//! subscriber list. Forking restores a snapshot: reused ticks are the
//! cache-hit analog of prompt caching.

pub mod items;
pub mod snapshot;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::engine::{Engine, EngineConfig, TickReport};
use crate::error::{ApiError, ApiResult};
use crate::substrate::Substrate;
use crate::types::*;

/// Map an unknown selector name to a 400 the client (or the LLM driving
/// tool calls) can act on instead of silently matching the whole universe.
fn selector_unknown_error(e: crate::substrate::UnknownSelector) -> ApiError {
    let param: &'static str = match e.field {
        "region" => "region",
        "cell_type" => "cell_type",
        "nt_type" => "nt_type",
        _ => "target",
    };
    ApiError::invalid_request("unknown_selector", e.to_string(), Some(param))
}

static SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn unix_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

pub struct SessionState {
    pub id: String,
    pub created_at: u64,
    pub last_activity: std::time::Instant,
    pub substrate_id: String,
    pub dt_ms: f32,
    pub metadata: HashMap<String, String>,
    pub engine: Engine,
    pub log: items::ItemLog,
    pub usage: Usage,
    pub subscribers: Vec<std::sync::mpsc::Sender<ActivityEvent>>,
}

pub struct SessionManager {
    pub(crate) substrate: RwLock<Option<Arc<Substrate>>>,
    substrate_id: RwLock<String>,
    engine_cfg: RwLock<EngineConfig>,
    pub(crate) sessions: Mutex<HashMap<String, SessionState>>,
    /// Session carrying the in-progress training overlay, if any.
    pub train_session: RwLock<Option<String>>,
    /// Enabled training files (plug-ins) and their merged root-id deltas.
    pub active_training: Mutex<crate::train::ActiveTraining>,
    snapshots: Mutex<snapshot::SnapshotStore>,
    pub train_job: crate::train::TrainJob,
    snapshot_every: u64,
    pub session_ttl: std::time::Duration,
    pub max_sessions: usize,
    pub max_items_per_session: usize,
}

impl SessionManager {
    /// Access the loaded substrate, or a guidance error if none loaded.
    fn substrate(&self) -> ApiResult<Arc<Substrate>> {
        self.substrate.read().unwrap().clone().ok_or_else(|| {
            ApiError {
                err_type: "invalid_request_error",
                code: "no_model_loaded",
                message: "no brain model loaded — download one on the Models page (admin UI) and restart, or pass --substrate".into(),
                param: None,
            }
        })
    }

    pub fn new(
        substrate: Option<Arc<Substrate>>,
        substrate_id: String,
        engine_cfg: EngineConfig,
    ) -> Self {
        Self {
            substrate: RwLock::new(substrate),
            substrate_id: RwLock::new(substrate_id),
            engine_cfg: RwLock::new(engine_cfg),
            sessions: Mutex::new(HashMap::new()),
            train_session: RwLock::new(None),
            active_training: Mutex::new(crate::train::ActiveTraining::default()),
            snapshots: Mutex::new(snapshot::SnapshotStore::new(8)),
            train_job: crate::train::TrainJob::default(),
            snapshot_every: 100,
            session_ttl: std::time::Duration::from_secs(5 * 60),
            max_sessions: 50,
            max_items_per_session: 10_000,
        }
    }

    /// Remove sessions idle > TTL, then LRU-evict beyond max_sessions.
    /// Returns number of sessions removed.
    pub fn gc(&self) -> usize {
        let mut sessions = self.sessions.lock().unwrap();
        let now = std::time::Instant::now();
        let ttl = self.session_ttl;

        // pass 1: TTL-based removal
        let expired: Vec<String> = sessions
            .iter()
            .filter(|(_, s)| now.duration_since(s.last_activity) > ttl)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &expired {
            sessions.remove(id);
        }

        // pass 2: LRU eviction beyond max_sessions
        let mut removed = expired.len();
        while sessions.len() > self.max_sessions {
            // find least-recently-active
            let lru = sessions
                .iter()
                .min_by_key(|(_, s)| s.last_activity)
                .map(|(id, _)| id.clone());
            match lru {
                Some(id) => {
                    sessions.remove(&id);
                    removed += 1;
                }
                None => break,
            }
        }
        removed
    }

    /// Memory statistics for the /memory endpoint.
    pub fn memory_stats(&self) -> serde_json::Value {
        let sessions = self.sessions.lock().unwrap();
        let mut total_items = 0usize;
        for s in sessions.values() {
            total_items += s.log.len();
        }
        let snapshots = self.snapshots.lock().unwrap();
        serde_json::json!({
            "active_sessions": sessions.len(),
            "max_sessions": self.max_sessions,
            "session_ttl_secs": self.session_ttl.as_secs(),
            "total_log_items": total_items,
            "total_snapshots": snapshots.total_len(),
        })
    }

    /// Force full GC regardless of TTL (admin endpoint).
    pub fn force_gc(&self) -> serde_json::Value {
        let before = self.sessions.lock().unwrap().len();
        let removed = self.gc();
        let after = self.sessions.lock().unwrap().len();
        serde_json::json!({"sessions_before": before, "removed": removed, "sessions_after": after})
    }

    /// Background GC thread — call once at server startup.
    pub fn start_gc_thread(self: &Arc<Self>, interval: std::time::Duration) {
        let store = Arc::clone(self);
        std::thread::spawn(move || loop {
            std::thread::sleep(interval);
            let removed = store.gc();
            if removed > 0 {
                eprintln!("gc: removed {} expired sessions", removed);
            }
        });
    }

    pub fn models(&self) -> ModelsResponse {
        ModelsResponse {
            object: "list".into(),
            data: match self.substrate.read().unwrap().as_ref() {
                Some(sub) => vec![serde_json::json!({
                    "id": self.substrate_id.read().unwrap().clone(),
                    "object": "substrate",
                    "n_neurons": sub.header.n_neurons,
                    "n_edges": sub.header.n_edges,
                    "source": &sub.header.source,
                    "regions": &sub.header.string_tables.regions,
                    "cell_types": &sub.header.string_tables.cell_types,
                    "nt_types": &sub.header.string_tables.nt_types,
                })],
                None => vec![],
            },
        }
    }

    /// Session summaries for the admin UI list view.
    pub fn regions(&self) -> ApiResult<Vec<String>> {
        Ok(self.substrate()?.header.string_tables.regions.clone())
    }

    /// Resolve a selector directly against the substrate (no session).
    pub fn query(&self, sel: &crate::types::NeuronSelector) -> ApiResult<serde_json::Value> {
        let matched = self
            .substrate()?
            .select(sel)
            .map_err(selector_unknown_error)?;
        let count = matched.len() as u64;
        let limit = (sel.limit.unwrap_or(200000) as usize).min(matched.len());
        // even stride sampling when the match set exceeds the limit
        let stride = matched.len().checked_div(limit).unwrap_or(1);
        let picked: Vec<u32> = (0..limit).map(|i| matched[i * stride]).collect();
        let neurons = self.substrate()?.selected_details(&picked);
        Ok(serde_json::json!({"count": count, "returned": neurons.len(), "neurons": neurons}))
    }

    /// Deterministically sample edges from the local connectome.
    /// Return a connected subgraph for 3D rendering.
    pub fn subgraph(&self, node_limit: usize, edge_limit: usize) -> ApiResult<serde_json::Value> {
        let (neurons, edges) = self.substrate()?.subgraph(node_limit, edge_limit);
        Ok(serde_json::json!({"neurons": neurons, "edges": edges}))
    }

    pub fn sample_edges(&self, limit: usize) -> ApiResult<serde_json::Value> {
        let edges = self.substrate()?.sample_edges(limit);
        Ok(serde_json::json!({"count": edges.len(), "edges": edges}))
    }

    /// Update engine config for all future ticks on all sessions.
    /// Only runtime-adjustable fields are modified.
    pub fn update_engine_config(&self, new_cfg: &crate::engine::EngineConfig) {
        {
            let mut cfg = self.engine_cfg.write().unwrap();
            cfg.dt_ms = new_cfg.dt_ms;
            cfg.use_simd = new_cfg.use_simd;
            cfg.n_threads = new_cfg.n_threads.max(1);
            cfg.weight_scale = new_cfg.weight_scale;
            cfg.input_gain = new_cfg.input_gain;
            cfg.v_thresh = new_cfg.v_thresh;
        }
        let mut sessions = self.sessions.lock().unwrap();
        for s in sessions.values_mut() {
            s.engine.cfg.dt_ms = new_cfg.dt_ms;
            s.engine.cfg.use_simd = new_cfg.use_simd;
            s.engine.cfg.n_threads = new_cfg.n_threads.max(1);
            s.engine.cfg.weight_scale = new_cfg.weight_scale;
            s.engine.cfg.input_gain = new_cfg.input_gain;
            s.engine.cfg.v_thresh = new_cfg.v_thresh;
        }
    }

    pub fn get_engine_config(&self) -> crate::engine::EngineConfig {
        self.engine_cfg.read().unwrap().clone()
    }

    /// OpenAI-style stateless call: create session -> observe input ->
    /// step -> read actions -> delete session. User never sees session ids.
    /// Returns actions (the "assistant response" of the fly brain).
    pub fn chat_simulate(
        &self,
        target: &crate::types::NeuronSelector,
        current: f32,
        steps: u32,
    ) -> ApiResult<serde_json::Value> {
        let sess = self.create(crate::types::CreateSessionRequest {
            substrate: String::new(),
            adapters: Vec::new(),
            dt_ms: None,
            metadata: [("kind".to_string(), "chat_simulate".to_string())]
                .into_iter()
                .collect(),
        })?;
        let sid = sess.id;
        let result = (|| -> ApiResult<serde_json::Value> {
            self.observe(
                &sid,
                crate::types::ObserveRequest {
                    modality: "visual".into(),
                    target: target.clone(),
                    current,
                    duration_ticks: 1,
                },
            )?;
            let resp = self.step(
                &sid,
                crate::types::StepRequest {
                    steps: steps.clamp(1, 10_000),
                },
            )?;
            Ok(serde_json::json!({
                "object": "simulation",
                "spikes_last_tick": resp.n_spikes,
                "ticks": resp.tick,
                "actions": resp.actions,
                "usage": resp.usage,
            }))
        })();
        // always clean up the internal session
        let _ = self.delete(&sid);
        result
    }

    /// Hot-swap the loaded substrate: replace the model atomically and
    /// clear all sessions (their engine state belongs to the old model).
    pub fn reload_substrate(&self, new_substrate: Arc<Substrate>, new_id: String) -> usize {
        {
            let mut sub = self.substrate.write().unwrap();
            *sub = Some(new_substrate);
        }
        {
            let mut id = self.substrate_id.write().unwrap();
            *id = new_id;
        }
        let mut sessions = self.sessions.lock().unwrap();
        let cleared = sessions.len();
        sessions.clear();
        cleared
    }

    pub fn session_count(&self) -> usize {
        self.sessions.lock().unwrap().len()
    }

    pub fn list(&self) -> Vec<SessionObject> {
        self.sessions
            .lock()
            .unwrap()
            .values()
            .map(|s| Self::to_object(s, &[]))
            .collect()
    }

    pub fn create(&self, req: CreateSessionRequest) -> ApiResult<SessionObject> {
        if !req.substrate.is_empty() && req.substrate != *self.substrate_id.read().unwrap() {
            return Err(ApiError::invalid_request(
                "substrate_not_found",
                format!(
                    "unknown substrate `{}` (available: `{}`)",
                    req.substrate,
                    self.substrate_id.read().unwrap()
                ),
                Some("substrate"),
            ));
        }
        let id = format!(
            "sess_{}_{:x}",
            unix_nanos(),
            SESSION_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let dt_ms = req.dt_ms.unwrap_or(self.engine_cfg.read().unwrap().dt_ms);
        let mut cfg = self.engine_cfg.read().unwrap().clone();
        cfg.dt_ms = dt_ms;
        let engine = Engine::new(self.substrate()?.clone(), cfg);
        let state = SessionState {
            id: id.clone(),
            created_at: unix_secs(),
            last_activity: std::time::Instant::now(),
            substrate_id: self.substrate_id.read().unwrap().clone(),
            dt_ms,
            metadata: req.metadata,
            engine,
            log: items::ItemLog::new(),
            usage: Usage::default(),
            subscribers: Vec::new(),
        };
        let obj = Self::to_object(&state, &req.adapters);
        self.sessions.lock().unwrap().insert(id, state);
        // carry enabled training plug-ins into the fresh engine
        if !self.active_training.lock().unwrap().merged.is_empty() {
            self.refresh_overlays();
        }
        Ok(obj)
    }

    fn to_object(state: &SessionState, adapters: &[String]) -> SessionObject {
        SessionObject {
            id: state.id.clone(),
            object: "session".into(),
            created_at: state.created_at,
            substrate: state.substrate_id.clone(),
            adapters: adapters.to_vec(),
            dt_ms: state.dt_ms,
            metadata: state.metadata.clone(),
            current_tick: state.engine.tick_count(),
        }
    }

    pub fn get(&self, id: &str) -> ApiResult<SessionObject> {
        let sessions = self.sessions.lock().unwrap();
        let s = sessions
            .get(id)
            .ok_or_else(|| ApiError::not_found(format!("session `{id}` not found")))?;
        Ok(Self::to_object(s, &[]))
    }

    pub fn update(&self, id: &str, req: UpdateSessionRequest) -> ApiResult<SessionObject> {
        let mut sessions = self.sessions.lock().unwrap();
        let s = sessions
            .get_mut(id)
            .ok_or_else(|| ApiError::not_found(format!("session `{id}` not found")))?;
        if let Some(meta) = req.metadata {
            s.metadata = meta;
        }
        Ok(Self::to_object(s, &[]))
    }

    pub fn delete(&self, id: &str) -> ApiResult<()> {
        let removed = self.sessions.lock().unwrap().remove(id);
        if removed.is_none() {
            return Err(ApiError::not_found(format!("session `{id}` not found")));
        }
        Ok(())
    }

    /// Resolve a NeuronSelector into dense neuron indices. Unknown selector
    /// names are an error; an empty substrate resolves to nothing.
    fn resolve_selector(&self, sel: &NeuronSelector) -> ApiResult<Vec<u32>> {
        let sub = match self.substrate.read().unwrap().as_ref() {
            Some(s) => s.clone(),
            None => return Ok(Vec::new()),
        };
        let mut out = sub.select(sel).map_err(selector_unknown_error)?;
        // true retinotopy: azimuth filter on soma positions (v3 substrates)
        if let (Some(az), false) = (sel.retina, sub.positions.is_empty()) {
            if sel.ids.is_empty() {
                let az = az.clamp(0.0, 1.0);
                let n = sub.positions.len() / 3;
                let (mut cx, mut cz) = (0.0f32, 0.0f32);
                for i in 0..n {
                    cx += sub.positions[i * 3];
                    cz += sub.positions[i * 3 + 2];
                }
                cx /= n as f32;
                cz /= n as f32;
                let half = std::f32::consts::PI / 16.0; // ±11.25° window
                out.retain(|&i| {
                    let (x, z) = (
                        sub.positions[i as usize * 3],
                        sub.positions[i as usize * 3 + 2],
                    );
                    let dx = x - cx;
                    let dz = z - cz;
                    if dx.abs() < 1e-6 && dz.abs() < 1e-6 {
                        return false;
                    }
                    let a = dz.atan2(dx).rem_euclid(2.0 * std::f32::consts::PI);
                    let a01 = a / (2.0 * std::f32::consts::PI);
                    let d = (a01 - az).abs();
                    d.min(1.0 - d) <= half / (2.0 * std::f32::consts::PI)
                });
            }
        }
        if sel.ids.is_empty() {
            // explicit ids are never truncated; name filters are
            let cap = sel.limit.map(|l| l as usize).unwrap_or(256);
            if let Some(f) = sel.offset {
                // retinotopy: a contiguous slice of the matched set, starting
                // at the given fraction — e.g. offset 0.5 = "straight ahead"
                let f = f.clamp(0.0, 1.0);
                let start = ((f * out.len() as f32) as usize).min(out.len());
                let end = (start + cap).min(out.len());
                out = out[start..end].to_vec();
            } else {
                out.truncate(cap);
            }
        }
        Ok(out)
    }

    pub fn observe(&self, id: &str, req: ObserveRequest) -> ApiResult<SessionItem> {
        let mut sessions = self.sessions.lock().unwrap();
        let s = sessions
            .get_mut(id)
            .ok_or_else(|| ApiError::not_found(format!("session `{id}` not found")))?;
        let idx = self.resolve_selector(&req.target)?;
        if idx.is_empty() {
            return Err(ApiError::invalid_request(
                "empty_selection",
                "selector matched no neurons",
                Some("target"),
            ));
        }
        s.engine.inject(&idx, req.current);
        let tick = s.engine.tick_count();
        let body = serde_json::json!({
            "modality": req.modality,
            "target": req.target,
            "current": req.current,
            "duration_ticks": req.duration_ticks,
            "n_neurons_stimulated": idx.len(),
        });
        let item = s.log.append(id, "observe", tick, body);
        s.log.cap(self.max_items_per_session);
        Ok(item)
    }

    pub fn step(&self, id: &str, req: StepRequest) -> ApiResult<StepResponse> {
        let steps = req.steps.clamp(1, 10_000);
        let snapshot_every = self.snapshot_every;
        let substrate = self.substrate()?.clone();
        let mut sessions = self.sessions.lock().unwrap();
        let s = sessions
            .get_mut(id)
            .ok_or_else(|| ApiError::not_found(format!("session `{id}` not found")))?;

        // split borrows: engine goes into run_ticks, the rest stay callable
        let SessionState {
            engine,
            log,
            usage,
            subscribers,
            ..
        } = s;

        // Motor readout: VNC neurons' firing rate over the whole window.
        // (Membrane-potential sampling read zero — spiking neurons are
        // membrane-reset before the view is taken; counts are the signal.)
        let mut spike_counts: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::new();
        let mut last_report: Option<TickReport> = None;
        engine.run_ticks(steps, &mut |report, view| {
            let mut tick_vnc: Vec<u64> = Vec::new();
            for &i in view.spikes() {
                let region = substrate.header.string_tables.regions
                    [substrate.region[i as usize] as usize]
                    .clone();
                if region.to_lowercase().contains("vnc") {
                    *spike_counts.entry(i).or_insert(0) += 1;
                    tick_vnc.push(substrate.root_ids[i as usize]);
                }
            }

            if snapshot_every > 0 && report.tick % snapshot_every == 0 {
                let blob = view.state_bytes();
                usage.snapshot_writes += 1;
                self.snapshots.lock().unwrap().put(id, report.tick, blob);
            }
            let event = ActivityEvent {
                session_id: id.to_string(),
                tick: report.tick,
                t_ms: report.t_ms,
                n_spikes: report.n_spikes,
                spike_sample: tick_vnc,
            };
            subscribers.retain(|tx| tx.send(event.clone()).is_ok());
            last_report = Some(report.clone());
        });
        let report = last_report.ok_or_else(|| ApiError::internal("no tick executed"))?;
        usage.ticks_simulated += steps as u64;

        let mut actions: Vec<Action> = spike_counts
            .into_iter()
            .map(|(i, c)| Action {
                neuron_id: substrate.root_ids[i as usize],
                rate: ((c as f32) / (steps as f32)).min(1.0),
            })
            .collect();
        actions.sort_by(|a, b| {
            b.rate
                .partial_cmp(&a.rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        actions.truncate(32);

        let body = serde_json::json!({"tick": report.tick, "n_spikes": report.n_spikes, "actions": actions});
        log.append(id, "step", report.tick, body);

        Ok(StepResponse {
            object: "session.step".into(),
            session_id: id.to_string(),
            tick: report.tick,
            t_ms: report.t_ms,
            n_spikes: report.n_spikes,
            actions,
            previous_tick_id: None,
            usage: usage.clone(),
        })
    }

    pub fn list_items(&self, id: &str, q: ListItemsQuery) -> ApiResult<ListItemsResponse> {
        let sessions = self.sessions.lock().unwrap();
        let s = sessions
            .get(id)
            .ok_or_else(|| ApiError::not_found(format!("session `{id}` not found")))?;
        Ok(s.log.list(&q))
    }

    pub fn fork(&self, req: ForkRequest) -> ApiResult<SessionObject> {
        let blob = self
            .snapshots
            .lock()
            .unwrap()
            .get(&req.source_session, req.at_tick)
            .cloned()
            .ok_or_else(|| {
                ApiError::invalid_request(
                    "snapshot_not_found",
                    format!(
                        "no snapshot of session `{}` at tick {} (snapshots are taken every {} ticks)",
                        req.source_session, req.at_tick, self.snapshot_every
                    ),
                    Some("at_tick"),
                )
            })?;
        let new_req = CreateSessionRequest {
            substrate: self.substrate_id.read().unwrap().clone(),
            adapters: Vec::new(),
            dt_ms: None,
            metadata: req.metadata,
        };
        let mut obj = self.create(new_req)?;
        {
            let mut sessions = self.sessions.lock().unwrap();
            let s = sessions.get_mut(&obj.id).unwrap();
            s.engine.restore_state(&blob).map_err(ApiError::internal)?;
            s.usage.ticks_reused = req.at_tick;
            let body =
                serde_json::json!({"source_session": req.source_session, "at_tick": req.at_tick});
            s.log.append(&s.id, "fork", req.at_tick, body);
        }
        obj.current_tick = req.at_tick;
        Ok(obj)
    }

    pub fn subscribe(&self, id: &str) -> ApiResult<std::sync::mpsc::Receiver<ActivityEvent>> {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut sessions = self.sessions.lock().unwrap();
        let s = sessions
            .get_mut(id)
            .ok_or_else(|| ApiError::not_found(format!("session `{id}` not found")))?;
        s.subscribers.push(tx);
        Ok(rx)
    }

    pub fn usage_of(&self, id: &str) -> ApiResult<Usage> {
        let sessions = self.sessions.lock().unwrap();
        let s = sessions
            .get(id)
            .ok_or_else(|| ApiError::not_found(format!("session `{id}` not found")))?;
        Ok(s.usage.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::{FlybinHeader, StringTables};

    fn mini_substrate() -> Arc<Substrate> {
        // 2 neurons: 0 (brain) -> 1 (VNC)
        let header = FlybinHeader {
            format_version: 1,
            n_neurons: 2,
            n_edges: 1,
            source: "test".into(),
            string_tables: StringTables {
                regions: vec!["".into(), "brain_R".into(), "VNC_T1_R".into()],
                cell_types: vec!["".into()],
                nt_types: vec!["".into(), "GLUT".into()],
            },
            weights_f32: false,
        };
        Substrate {
            header,
            indptr: vec![0, 1, 1],
            indices: vec![1],
            weights: crate::substrate::flybin::Weights::F32(vec![50.0]),
            root_ids: vec![100, 200],
            region: vec![1, 2],
            cell_type: vec![0, 0],
            nt_type: vec![1, 1],
            positions: Vec::new(),
        }
        .into()
    }

    fn mgr() -> SessionManager {
        let mut m = SessionManager::new(
            Some(mini_substrate()),
            "test-substrate".into(),
            crate::engine::EngineConfig::default(),
        );
        m.snapshot_every = 1;
        m
    }

    #[test]
    fn full_lifecycle() {
        let m = mgr();
        let s = m
            .create(CreateSessionRequest {
                substrate: "test-substrate".into(),
                adapters: vec!["builtin-vnc-readout".into()],
                dt_ms: None,
                metadata: HashMap::new(),
            })
            .unwrap();
        let sid = s.id.clone();
        assert!(sid.starts_with("sess_"));

        let item = m
            .observe(
                &sid,
                ObserveRequest {
                    modality: "current".into(),
                    target: NeuronSelector {
                        ids: vec![100],
                        ..Default::default()
                    },
                    current: 30.0,
                    duration_ticks: 1,
                },
            )
            .unwrap();
        assert_eq!(item.kind, "observe");

        let step = m.step(&sid, StepRequest { steps: 2 }).unwrap();
        assert_eq!(step.tick, 2);
        assert_eq!(step.usage.ticks_simulated, 2);

        let items = m.list_items(&sid, ListItemsQuery::default()).unwrap();
        assert_eq!(items.data.len(), 2);

        let forked = m
            .fork(ForkRequest {
                source_session: sid.clone(),
                at_tick: 1,
                metadata: HashMap::new(),
            })
            .unwrap();
        assert_eq!(forked.current_tick, 1);

        let usage = m.usage_of(&sid).unwrap();
        assert!(usage.snapshot_writes >= 1);
        let fu = m.usage_of(&forked.id).unwrap();
        assert_eq!(fu.ticks_reused, 1);

        m.delete(&sid).unwrap();
        assert!(m.get(&sid).is_err());
    }

    #[test]
    fn fork_missing_snapshot_is_error() {
        let m = mgr();
        let err = m
            .fork(ForkRequest {
                source_session: "nope".into(),
                at_tick: 5,
                metadata: HashMap::new(),
            })
            .unwrap_err();
        assert_eq!(err.code, "snapshot_not_found");
    }
}
