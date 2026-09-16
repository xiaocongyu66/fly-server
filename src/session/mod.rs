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
use std::sync::{Arc, Mutex};

use crate::engine::{Engine, EngineConfig};
use crate::error::{ApiError, ApiResult};
use crate::substrate::Substrate;
use crate::types::*;

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
    pub substrate_id: String,
    pub dt_ms: f32,
    pub metadata: HashMap<String, String>,
    pub engine: Engine,
    pub log: items::ItemLog,
    pub usage: Usage,
    pub subscribers: Vec<std::sync::mpsc::Sender<ActivityEvent>>,
}

pub struct SessionManager {
    substrate: Arc<Substrate>,
    substrate_id: String,
    engine_cfg: EngineConfig,
    sessions: Mutex<HashMap<String, SessionState>>,
    snapshots: Mutex<snapshot::SnapshotStore>,
    snapshot_every: u64,
}

impl SessionManager {
    pub fn new(substrate: Arc<Substrate>, substrate_id: String) -> Self {
        Self {
            substrate,
            substrate_id,
            engine_cfg: EngineConfig::default(),
            sessions: Mutex::new(HashMap::new()),
            snapshots: Mutex::new(snapshot::SnapshotStore::new(8)),
            snapshot_every: 100,
        }
    }

    pub fn models(&self) -> ModelsResponse {
        ModelsResponse {
            object: "list".into(),
            data: vec![serde_json::to_value(SubstrateObject {
                id: self.substrate_id.clone(),
                object: "substrate".into(),
                n_neurons: self.substrate.header.n_neurons,
                n_edges: self.substrate.header.n_edges,
                source: self.substrate.header.source.clone(),
            })
            .unwrap_or_default()],
        }
    }

    pub fn create(&self, req: CreateSessionRequest) -> ApiResult<SessionObject> {
        if !req.substrate.is_empty() && req.substrate != self.substrate_id {
            return Err(ApiError::invalid_request(
                "substrate_not_found",
                format!("unknown substrate `{}` (available: `{}`)", req.substrate, self.substrate_id),
                Some("substrate"),
            ));
        }
        let id = format!("sess_{}_{:x}", unix_nanos(), SESSION_COUNTER.fetch_add(1, Ordering::Relaxed));
        let dt_ms = req.dt_ms.unwrap_or(self.engine_cfg.dt_ms);
        let mut cfg = self.engine_cfg.clone();
        cfg.dt_ms = dt_ms;
        let engine = Engine::new(self.substrate.clone(), cfg);
        let state = SessionState {
            id: id.clone(),
            created_at: unix_secs(),
            substrate_id: self.substrate_id.clone(),
            dt_ms,
            metadata: req.metadata,
            engine,
            log: items::ItemLog::new(),
            usage: Usage::default(),
            subscribers: Vec::new(),
        };
        let obj = Self::to_object(&state, &req.adapters);
        self.sessions.lock().unwrap().insert(id, state);
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
        let s = sessions.get(id).ok_or_else(|| ApiError::not_found(format!("session `{id}` not found")))?;
        Ok(Self::to_object(s, &[]))
    }

    pub fn update(&self, id: &str, req: UpdateSessionRequest) -> ApiResult<SessionObject> {
        let mut sessions = self.sessions.lock().unwrap();
        let s = sessions.get_mut(id).ok_or_else(|| ApiError::not_found(format!("session `{id}` not found")))?;
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

    /// Resolve a NeuronSelector into dense neuron indices.
    fn resolve_selector(&self, sel: &NeuronSelector) -> Vec<u32> {
        let sub = &self.substrate;
        let n = sub.n_neurons();
        let mut out: Vec<u32> = Vec::new();
        if !sel.ids.is_empty() {
            for root in &sel.ids {
                if let Ok(i) = sub.root_ids.binary_search(root) {
                    out.push(i as u32);
                }
            }
        } else {
            let region_idx = sel
                .region
                .as_ref()
                .and_then(|r| sub.header.string_tables.regions.iter().position(|t| t.eq_ignore_ascii_case(r)));
            let ct_idx = sel
                .cell_type
                .as_ref()
                .and_then(|c| sub.header.string_tables.cell_types.iter().position(|t| t.eq_ignore_ascii_case(c)));
            let nt_idx = sel
                .nt_type
                .as_ref()
                .and_then(|t| sub.header.string_tables.nt_types.iter().position(|x| x.eq_ignore_ascii_case(t)));
            if region_idx.is_some() || ct_idx.is_some() || nt_idx.is_some() {
                for i in 0..n {
                    if region_idx.is_some() && sub.region[i] as usize != region_idx.unwrap() {
                        continue;
                    }
                    if ct_idx.is_some() && sub.cell_type[i] as usize != ct_idx.unwrap() {
                        continue;
                    }
                    if nt_idx.is_some() && sub.nt_type[i] as usize != nt_idx.unwrap() {
                        continue;
                    }
                    out.push(i as u32);
                }
            }
        }
        if let Some(limit) = sel.limit {
            out.truncate(limit as usize);
        } else {
            out.truncate(256);
        }
        out
    }

    pub fn observe(&self, id: &str, req: ObserveRequest) -> ApiResult<SessionItem> {
        let mut sessions = self.sessions.lock().unwrap();
        let s = sessions.get_mut(id).ok_or_else(|| ApiError::not_found(format!("session `{id}` not found")))?;
        let idx = self.resolve_selector(&req.target);
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
        Ok(s.log.append(id, "observe", tick, body))
    }

    pub fn step(&self, id: &str, req: StepRequest) -> ApiResult<StepResponse> {
        let steps = req.steps.clamp(1, 10_000);
        let v_thresh = self.engine_cfg.v_thresh;
        let mut sessions = self.sessions.lock().unwrap();
        let s = sessions.get_mut(id).ok_or_else(|| ApiError::not_found(format!("session `{id}` not found")))?;

        let mut last_report = None;
        let mut actions: Vec<Action> = Vec::new();
        for _ in 0..steps {
            let report = s.engine.tick();
            // Motor readout: spiking VNC neurons, rate from membrane potential.
            actions.clear();
            let sub = &self.substrate;
            for &i in s.engine.last_spikes() {
                let region = sub.header.string_tables.regions[sub.region[i as usize] as usize].clone();
                if region.to_lowercase().contains("vnc") {
                    let rate = (s.engine.membrane(i as usize) / v_thresh).clamp(0.0, 1.0);
                    actions.push(Action { neuron_id: sub.root_ids[i as usize], rate });
                }
            }
            actions.sort_by(|a, b| b.rate.partial_cmp(&a.rate).unwrap_or(std::cmp::Ordering::Equal));
            actions.truncate(32);
            let tick = report.tick;
            let event = ActivityEvent {
                session_id: id.to_string(),
                tick: report.tick,
                t_ms: report.t_ms,
                n_spikes: report.n_spikes,
                spike_sample: actions.iter().map(|a| a.neuron_id).collect(),
            };
            last_report = Some(report);

            if self.snapshot_every > 0 && tick % self.snapshot_every == 0 {
                let blob = s.engine.state_bytes();
                s.usage.snapshot_writes += 1;
                self.snapshots.lock().unwrap().put(id, tick, blob);
            }
            s.subscribers.retain(|tx| tx.send(event.clone()).is_ok());
        }
        let report = last_report.ok_or_else(|| ApiError::internal("no tick executed"))?;
        s.usage.ticks_simulated += steps as u64;

        let body = serde_json::json!({"tick": report.tick, "n_spikes": report.n_spikes, "actions": actions});
        s.log.append(id, "step", report.tick, body);

        Ok(StepResponse {
            object: "session.step".into(),
            session_id: id.to_string(),
            tick: report.tick,
            t_ms: report.t_ms,
            n_spikes: report.n_spikes,
            actions: actions.clone(),
            previous_tick_id: None,
            usage: s.usage.clone(),
        })
    }

    pub fn list_items(&self, id: &str, q: ListItemsQuery) -> ApiResult<ListItemsResponse> {
        let sessions = self.sessions.lock().unwrap();
        let s = sessions.get(id).ok_or_else(|| ApiError::not_found(format!("session `{id}` not found")))?;
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
            substrate: self.substrate_id.clone(),
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
            let body = serde_json::json!({"source_session": req.source_session, "at_tick": req.at_tick});
            s.log.append(&s.id, "fork", req.at_tick, body);
        }
        obj.current_tick = req.at_tick;
        Ok(obj)
    }

    pub fn subscribe(&self, id: &str) -> ApiResult<std::sync::mpsc::Receiver<ActivityEvent>> {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut sessions = self.sessions.lock().unwrap();
        let s = sessions.get_mut(id).ok_or_else(|| ApiError::not_found(format!("session `{id}` not found")))?;
        s.subscribers.push(tx);
        Ok(rx)
    }

    pub fn usage_of(&self, id: &str) -> ApiResult<Usage> {
        let sessions = self.sessions.lock().unwrap();
        let s = sessions.get(id).ok_or_else(|| ApiError::not_found(format!("session `{id}` not found")))?;
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
        };
        Substrate {
            header,
            indptr: vec![0, 1, 1],
            indices: vec![1],
            weights: vec![50.0],
            root_ids: vec![100, 200],
            region: vec![1, 2],
            cell_type: vec![0, 0],
            nt_type: vec![1, 1],
        }
        .into()
    }

    fn mgr() -> SessionManager {
        let mut m = SessionManager::new(mini_substrate(), "test-substrate".into());
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
                    target: NeuronSelector { ids: vec![100], ..Default::default() },
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
            .fork(ForkRequest { source_session: "nope".into(), at_tick: 5, metadata: HashMap::new() })
            .unwrap_err();
        assert_eq!(err.code, "snapshot_not_found");
    }
}
