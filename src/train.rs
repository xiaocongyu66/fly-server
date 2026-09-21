//! Training loop: rollout episodes over the session API with a fixed
//! baseline policy (argmax readout rate); reward = mean spikes per step.

use crate::error::ApiError;
use crate::session::SessionManager;
use crate::types::{Action, CreateSessionRequest, NeuronSelector, ObserveRequest, StepRequest};

fn default_episodes() -> u32 {
    10
}
fn default_steps_per_episode() -> u32 {
    100
}
fn default_stim_current() -> f32 {
    30.0
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainConfig {
    #[serde(default = "default_episodes")]
    pub episodes: u32,
    #[serde(default = "default_steps_per_episode")]
    pub steps_per_episode: u32,
    #[serde(default)]
    pub stim_region: Option<String>,
    #[serde(default = "default_stim_current")]
    pub stim_current: f32,
    /// Learn while rolling out (R-STDP into the session's weight overlay).
    #[serde(default)]
    pub learn: bool,
    #[serde(default = "default_lr")]
    pub lr: f32,
    #[serde(default = "default_window")]
    pub window: u32,
    #[serde(default = "default_tau")]
    pub tau: f32,
}

fn default_lr() -> f32 {
    0.05
}
fn default_window() -> u32 {
    8
}
fn default_tau() -> f32 {
    6.0
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            episodes: default_episodes(),
            steps_per_episode: default_steps_per_episode(),
            stim_region: None,
            stim_current: default_stim_current(),
            learn: false,
            lr: default_lr(),
            window: default_window(),
            tau: default_tau(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TrainResult {
    pub episodes: u32,
    pub mean_reward: f32,
    pub best_reward: f32,
    pub total_ticks: u64,
}

/// One episode: create a session, inject the stimulus, then step
/// `steps_per_episode` times. Reward = total spikes / steps.
pub fn rollout(mgr: &SessionManager, cfg: &TrainConfig) -> Result<f32, ApiError> {
    run_episode(mgr, cfg, 0)
}

fn run_episode(mgr: &SessionManager, cfg: &TrainConfig, episode: u32) -> Result<f32, ApiError> {
    let region = match &cfg.stim_region {
        Some(r) => r.clone(),
        None => {
            return Err(ApiError::invalid_request(
                "stim_region_required",
                "train requires stim_region to select the stimulus target",
                Some("stim_region"),
            ))
        }
    };

    let mut metadata = std::collections::HashMap::new();
    metadata.insert("kind".to_string(), "train".to_string());
    metadata.insert("episode".to_string(), episode.to_string());
    let req = CreateSessionRequest {
        substrate: String::new(),
        adapters: Vec::new(),
        dt_ms: None,
        metadata,
    };
    let session = mgr.create(req)?;
    let sid = session.id;

    let observe = ObserveRequest {
        modality: "current".to_string(),
        frame: Vec::new(),
        target: NeuronSelector {
            region: Some(region),
            ..Default::default()
        },
        current: cfg.stim_current,
        duration_ticks: 1,
    };
    mgr.observe(&sid, observe)?;

    let mut total_spikes = 0u64;
    let steps = cfg.steps_per_episode.max(1);
    for _ in 0..cfg.steps_per_episode {
        let resp = mgr.step(&sid, StepRequest { steps: 1 })?;
        total_spikes += resp.n_spikes as u64;
    }
    Ok(total_spikes as f32 / steps as f32)
}

/// Run `episodes` rollouts and aggregate reward statistics.
pub fn train(mgr: &SessionManager, cfg: &TrainConfig) -> Result<TrainResult, ApiError> {
    let episodes = cfg.episodes;
    let mut rewards = Vec::with_capacity(episodes as usize);
    for ep in 0..episodes {
        rewards.push(run_episode(mgr, cfg, ep)?);
    }
    let n = rewards.len() as f32;
    let mean = if n > 0.0 {
        rewards.iter().sum::<f32>() / n
    } else {
        0.0
    };
    let best = rewards.iter().copied().fold(0.0f32, f32::max);
    Ok(TrainResult {
        episodes,
        mean_reward: mean,
        best_reward: best,
        total_ticks: episodes as u64 * cfg.steps_per_episode as u64,
    })
}

/// Baseline policy: pick the readout neuron with the highest firing rate;
/// on ties the earliest action wins.
pub fn baseline_policy(actions: &[Action]) -> Option<u64> {
    let mut best: Option<(f32, u64)> = None;
    for a in actions {
        match best {
            Some((r, _)) if a.rate <= r => {}
            _ => best = Some((a.rate, a.neuron_id)),
        }
    }
    best.map(|(_, id)| id)
}

/// Lifecycle of the background training job, reported by
/// `GET /v1/admin/train/status`.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum TrainPhase {
    Idle,
    Running {
        episode: u32,
        total_episodes: u32,
        steps_done: u64,
        session_id: Option<String>,
    },
    Done {
        #[serde(flatten)]
        result: TrainResult,
        stopped: bool,
        session_id: Option<String>,
    },
    Failed {
        error: String,
    },
}

impl crate::session::SessionManager {
    /// Claim the training slot and spawn the background episode loop
    /// (409-conflict when a job is already running).
    pub fn start_train(self: &std::sync::Arc<Self>, cfg: TrainConfig) -> Result<(), ApiError> {
        {
            let mut st = self.train_job.state.lock().unwrap();
            if matches!(*st, TrainPhase::Running { .. }) {
                return Err(ApiError::conflict(
                    "train_in_flight",
                    "a training job is already running — poll /v1/admin/train/status",
                ));
            }
            *st = TrainPhase::Running {
                episode: 0,
                total_episodes: cfg.episodes.max(1),
                steps_done: 0,
                session_id: None,
            };
        }
        self.train_job
            .stop
            .store(false, std::sync::atomic::Ordering::Relaxed);
        *self.train_job.last_cfg.lock().unwrap() = cfg.clone();
        let mgr = std::sync::Arc::clone(self);
        std::thread::spawn(move || {
            mgr.run_train_loop(cfg);
        });
        Ok(())
    }

    /// Episode loop: rollout → record reward → publish progress. Exits
    /// early on `stop` or episode failure.
    fn run_train_loop(self: &std::sync::Arc<Self>, cfg: TrainConfig) {
        if cfg.learn {
            return self.run_learn_loop(cfg);
        }
        let episodes = cfg.episodes.max(1);
        let steps = cfg.steps_per_episode.max(1) as u64;
        let mut rewards: Vec<f32> = Vec::with_capacity(episodes as usize);
        let mut stopped = false;
        for ep in 0..episodes {
            if self
                .train_job
                .stop
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                stopped = true;
                break;
            }
            match run_episode(self, &cfg, ep) {
                Ok(r) => rewards.push(r),
                Err(e) => {
                    *self.train_job.state.lock().unwrap() = TrainPhase::Failed {
                        error: e.to_string(),
                    };
                    return;
                }
            }
            *self.train_job.state.lock().unwrap() = TrainPhase::Running {
                episode: rewards.len() as u32,
                total_episodes: episodes,
                steps_done: rewards.len() as u64 * steps,
                session_id: self.train_session.read().unwrap().clone(),
            };
        }
        let n = rewards.len() as f32;
        let mean = if n > 0.0 {
            rewards.iter().sum::<f32>() / n
        } else {
            0.0
        };
        let best = rewards.iter().copied().fold(0.0f32, f32::max);
        *self.train_job.state.lock().unwrap() = TrainPhase::Done {
            result: TrainResult {
                episodes: rewards.len() as u32,
                mean_reward: mean,
                best_reward: best,
                total_ticks: rewards.len() as u64 * steps,
            },
            stopped,
            session_id: self.train_session.read().unwrap().clone(),
        };
    }

    pub fn train_status(&self) -> TrainPhase {
        self.train_job.state.lock().unwrap().clone()
    }

    /// Rebuild every engine's overlay from the enabled training plug-ins
    /// (root-id keyed deltas translated through the loaded substrate).
    pub fn refresh_overlays(&self) {
        let Some(sub) = self.substrate.read().unwrap().clone() else {
            return;
        };
        let at = self.active_training.lock().unwrap();
        let mut idx: std::collections::HashMap<u32, std::collections::HashMap<u32, f32>> =
            std::collections::HashMap::new();
        for (&(pre_root, post_root), &dw) in at.merged.iter() {
            if let (Ok(pi), Ok(qi)) = (
                sub.root_ids.binary_search(&pre_root),
                sub.root_ids.binary_search(&post_root),
            ) {
                *idx.entry(pi as u32)
                    .or_default()
                    .entry(qi as u32)
                    .or_insert(0.0) += dw;
            }
        }
        drop(at);
        let mut sessions = self.sessions.lock().unwrap();
        for s in sessions.values_mut() {
            s.engine.overlay = idx.clone();
        }
    }

    /// Request an early stop; returns whether a job was running.
    pub fn stop_train(&self) -> bool {
        let running = matches!(
            *self.train_job.state.lock().unwrap(),
            TrainPhase::Running { .. }
        );
        if running {
            self.train_job
                .stop
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        running
    }
}

impl crate::session::SessionManager {
    /// Learning loop: ONE persistent session so the weight overlay survives
    /// across episodes; reward per episode modulates STDP via the engine's
    /// LearningState. The overlay stays in the session's engine until
    /// exported as a `.flydelta` plug-in.
    fn run_learn_loop(self: &std::sync::Arc<Self>, cfg: TrainConfig) {
        use crate::types::{ObserveRequest, StepRequest};
        let episodes = cfg.episodes.max(1);
        let steps = cfg.steps_per_episode.max(1) as u64;
        let region = match &cfg.stim_region {
            Some(r) => r.clone(),
            None => {
                *self.train_job.state.lock().unwrap() = TrainPhase::Failed {
                    error: "train requires stim_region".into(),
                };
                return;
            }
        };
        let session = match self.create(crate::types::CreateSessionRequest {
            substrate: String::new(),
            adapters: Vec::new(),
            dt_ms: None,
            metadata: [("kind".to_string(), "train".to_string())]
                .into_iter()
                .collect(),
        }) {
            Ok(s) => s,
            Err(e) => {
                *self.train_job.state.lock().unwrap() = TrainPhase::Failed {
                    error: e.to_string(),
                };
                return;
            }
        };
        let sid = session.id.clone();
        *self.train_session.write().unwrap() = Some(sid.clone());
        {
            let mut sessions = self.sessions.lock().unwrap();
            if let Some(s) = sessions.get_mut(&sid) {
                s.engine.learning = Some(crate::engine::LearningState {
                    lr: cfg.lr,
                    window: cfg.window,
                    tau: cfg.tau,
                    reward: 1.0,
                });
            }
        }
        let mut rewards: Vec<f32> = Vec::with_capacity(episodes as usize);
        let mut stopped = false;
        'outer: for _ep in 0..episodes {
            if self
                .train_job
                .stop
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                stopped = true;
                break;
            }
            let mut total = 0u64;
            for _ in 0..steps {
                if self
                    .train_job
                    .stop
                    .load(std::sync::atomic::Ordering::Relaxed)
                {
                    stopped = true;
                    break 'outer;
                }
                // persistent stimulus every step: the stimulated set re-spikes
                // each tick while downstream neurons lag one tick behind —
                // the timing structure STDP learns from
                if let Err(e) = self.observe(
                    &sid,
                    ObserveRequest {
                        modality: "current".into(),
                        frame: Vec::new(),
                        target: crate::types::NeuronSelector {
                            region: Some(region.clone()),
                            ..Default::default()
                        },
                        current: cfg.stim_current,
                        duration_ticks: 1,
                    },
                ) {
                    let _ = self.delete(&sid);
                    *self.train_job.state.lock().unwrap() = TrainPhase::Failed {
                        error: e.to_string(),
                    };
                    return;
                }
                match self.step(&sid, StepRequest { steps: 1 }) {
                    Ok(r) => total += r.n_spikes as u64,
                    Err(e) => {
                        let _ = self.delete(&sid);
                        *self.train_job.state.lock().unwrap() = TrainPhase::Failed {
                            error: e.to_string(),
                        };
                        return;
                    }
                }
            }
            let r = total as f32 / steps as f32;
            rewards.push(r);
            let baseline = rewards.iter().sum::<f32>() / rewards.len() as f32;
            {
                let mut sessions = self.sessions.lock().unwrap();
                if let Some(s) = sessions.get_mut(&sid) {
                    if let Some(l) = s.engine.learning.as_mut() {
                        l.reward = (r / baseline.max(1.0)).clamp(0.2, 3.0);
                    }
                }
            }
            *self.train_job.state.lock().unwrap() = TrainPhase::Running {
                episode: rewards.len() as u32,
                total_episodes: episodes,
                steps_done: rewards.len() as u64 * steps,
                session_id: Some(sid.clone()),
            };
        }
        {
            let mut sessions = self.sessions.lock().unwrap();
            if let Some(s) = sessions.get_mut(&sid) {
                s.engine.learning = None;
            }
        }
        let n = rewards.len() as f32;
        let mean = if n > 0.0 {
            rewards.iter().sum::<f32>() / n
        } else {
            0.0
        };
        let best = rewards.iter().copied().fold(0.0f32, f32::max);
        *self.train_job.state.lock().unwrap() = TrainPhase::Done {
            result: TrainResult {
                episodes: rewards.len() as u32,
                mean_reward: mean,
                best_reward: best,
                total_ticks: rewards.len() as u64 * steps,
            },
            stopped,
            session_id: Some(sid),
        };
    }
}

/// Shared slot for at most one background training job.
pub struct TrainJob {
    state: std::sync::Mutex<TrainPhase>,
    stop: std::sync::atomic::AtomicBool,
    /// Config of the most recent run (hyperparams for the save step).
    pub last_cfg: std::sync::Mutex<TrainConfig>,
}

impl Default for TrainJob {
    fn default() -> Self {
        Self {
            state: std::sync::Mutex::new(TrainPhase::Idle),
            stop: std::sync::atomic::AtomicBool::new(false),
            last_cfg: std::sync::Mutex::new(TrainConfig::default()),
        }
    }
}

/// Enabled training plug-ins and their merged root-id delta set. The base
/// substrate is never modified; deltas translate to per-engine overlays.
#[derive(Default)]
pub struct ActiveTraining {
    pub files: Vec<String>,
    pub merged: std::collections::HashMap<(u64, u64), f32>,
}

/// A persisted training plug-in (`.flydelta`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrainingFileMeta {
    pub name: String,
    pub base_model: String,
    pub deltas: usize,
    pub bytes: u64,
    pub created_at: u64,
}

pub fn training_dir(data_dir: &std::path::Path) -> std::path::PathBuf {
    data_dir.join("training")
}

fn safe_name(name: &str) -> Result<String, ApiError> {
    let cleaned: String = name
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .take(64)
        .collect();
    if cleaned.is_empty() {
        return Err(ApiError::invalid_request(
            "bad_name",
            "training file name needs at least one alphanumeric character",
            Some("name"),
        ));
    }
    Ok(cleaned)
}

/// Export the training session's overlay as a root-id-keyed `.flydelta`
/// plug-in. The base model file is never touched.
pub fn save_training(
    mgr: &crate::session::SessionManager,
    data_dir: &std::path::Path,
    name: &str,
    cfg: &TrainConfig,
) -> Result<TrainingFileMeta, ApiError> {
    let sid = mgr.train_session.read().unwrap().clone().ok_or_else(|| {
        ApiError::invalid_request("no_training_session", "no training session to export", None)
    })?;
    let sub_id = {
        let sessions = mgr.sessions.lock().unwrap();
        let s = sessions
            .get(&sid)
            .ok_or_else(|| ApiError::not_found(format!("training session `{sid}` is gone")))?;
        let sub = s.engine.substrate.clone();
        let base = s.substrate_id.clone();
        // root-id keyed deltas: portable across substrates sharing body ids
        let roots = &sub.root_ids;
        let mut deltas: Vec<(u64, u64, f32)> = Vec::new();
        for (&pre_idx, entries) in s.engine.overlay.iter() {
            let pre_root = roots[pre_idx as usize];
            for (&post_idx, &dw) in entries {
                deltas.push((pre_root, roots[post_idx as usize], dw));
            }
        }
        (base, deltas, sub)
    };
    let (base, deltas, _sub) = sub_id;
    let name = safe_name(name)?;
    let dir = training_dir(data_dir);
    std::fs::create_dir_all(&dir).map_err(|e| ApiError::internal(e.to_string()))?;
    let doc = serde_json::json!({
        "format": "flydelta-v1",
        "base_model": base,
        "name": name,
        "created_at": now_secs(),
        "hyperparams": {
            "lr": cfg.lr, "window": cfg.window, "tau": cfg.tau,
            "episodes": cfg.episodes, "steps_per_episode": cfg.steps_per_episode,
            "stim_region": cfg.stim_region,
        },
        "deltas": deltas,
    });
    let json = serde_json::to_vec_pretty(&doc).map_err(|e| ApiError::internal(e.to_string()))?;
    let path = dir.join(format!("{name}.flydelta"));
    std::fs::write(&path, &json).map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(TrainingFileMeta {
        name,
        base_model: base,
        deltas: deltas.len(),
        bytes: json.len() as u64,
        created_at: now_secs(),
    })
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Enable/disable a plug-in: (re)merge deltas from the enabled set, then
/// refresh every engine's overlay. Additive merge — a later enable of the
/// same file is a no-op.
pub fn set_training_enabled(
    mgr: &crate::session::SessionManager,
    data_dir: &std::path::Path,
    name: &str,
    on: bool,
) -> Result<(), ApiError> {
    let name = safe_name(name)?;
    let mut at = mgr.active_training.lock().unwrap();
    if on {
        if at.files.iter().any(|f| f == &name) {
            return Ok(());
        }
        let path = training_dir(data_dir).join(format!("{name}.flydelta"));
        let doc: serde_json::Value = std::fs::read(&path)
            .map_err(|_| ApiError::not_found(format!("training file `{name}` not found")))
            .and_then(|b| {
                serde_json::from_slice(&b).map_err(|e| ApiError::internal(e.to_string()))
            })?;
        let deltas = doc
            .get("deltas")
            .and_then(|d| d.as_array())
            .ok_or_else(|| ApiError::internal("training file missing deltas"))?;
        for row in deltas {
            let (p, q, dw) = (
                row.get(0).and_then(|v| v.as_u64()),
                row.get(1).and_then(|v| v.as_u64()),
                row.get(2).and_then(|v| v.as_f64()),
            );
            if let (Some(p), Some(q), Some(dw)) = (p, q, dw) {
                *at.merged.entry((p, q)).or_insert(0.0) += dw as f32;
            }
        }
        at.files.push(name);
    } else {
        at.files.retain(|f| f != &name);
        // rebuild from the remaining enabled files
        at.merged.clear();
        for f in at.files.clone() {
            let path = training_dir(data_dir).join(format!("{f}.flydelta"));
            if let Ok(b) = std::fs::read(&path) {
                if let Ok(doc) = serde_json::from_slice::<serde_json::Value>(&b) {
                    if let Some(rows) = doc.get("deltas").and_then(|d| d.as_array()) {
                        for row in rows {
                            if let (Some(p), Some(q), Some(dw)) = (
                                row.get(0).and_then(|v| v.as_u64()),
                                row.get(1).and_then(|v| v.as_u64()),
                                row.get(2).and_then(|v| v.as_f64()),
                            ) {
                                *at.merged.entry((p, q)).or_insert(0.0) += dw as f32;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Delete a plug-in file; if it was enabled, disable first.
pub fn delete_training(
    mgr: &crate::session::SessionManager,
    data_dir: &std::path::Path,
    name: &str,
) -> Result<(), ApiError> {
    let name = safe_name(name)?;
    let was_enabled = mgr.active_training.lock().unwrap().files.contains(&name);
    if was_enabled {
        set_training_enabled(mgr, data_dir, &name, false)?;
    }
    let path = training_dir(data_dir).join(format!("{name}.flydelta"));
    std::fs::remove_file(&path)
        .map_err(|_| ApiError::not_found(format!("training file `{name}` not found")))?;
    Ok(())
}

/// List every `.flydelta` in the training dir.
pub fn list_training(data_dir: &std::path::Path) -> Vec<TrainingFileMeta> {
    let dir = training_dir(data_dir);
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("flydelta") {
                continue;
            }
            if let Ok(doc) = std::fs::read(&p)
                .map_err(|_| ())
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).map_err(|_| ()))
            {
                let deltas = doc
                    .get("deltas")
                    .and_then(|d| d.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);
                out.push(TrainingFileMeta {
                    name: p
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    base_model: doc
                        .get("base_model")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .into(),
                    deltas,
                    bytes: doc.to_string().len() as u64,
                    created_at: doc.get("created_at").and_then(|v| v.as_u64()).unwrap_or(0),
                });
            }
        }
    }
    out.sort_by_key(|f| std::cmp::Reverse(f.created_at));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn act(id: u64, rate: f32) -> Action {
        Action {
            neuron_id: id,
            rate,
        }
    }

    #[test]
    fn baseline_empty() {
        assert_eq!(baseline_policy(&[]), None);
    }

    #[test]
    fn baseline_picks_max() {
        let actions = [act(1, 0.2), act(2, 0.9), act(3, 0.5)];
        assert_eq!(baseline_policy(&actions), Some(2));
    }

    #[test]
    fn baseline_tie_takes_first() {
        let actions = [act(7, 0.9), act(8, 0.9), act(9, 0.1)];
        assert_eq!(baseline_policy(&actions), Some(7));
    }
}

#[cfg(test)]
mod learning_tests {
    use crate::engine::{Engine, EngineConfig};
    use crate::substrate::Substrate;
    use std::sync::Arc;

    fn tiny_substrate() -> Substrate {
        // two neurons, one edge 0 -> 1 with weight 10
        let mut tables = crate::substrate::StringTables::default();
        tables.regions.push("r".into());
        tables.cell_types.push("c".into());
        tables.nt_types.push("acetylcholine".into());
        let header = crate::substrate::FlybinHeader {
            format_version: 1,
            weights_f32: true,
            n_neurons: 2,
            n_edges: 1,
            source: "test".into(),
            string_tables: tables,
        };
        Substrate {
            header,
            indptr: vec![0, 1, 1],
            indices: vec![1],
            weights: crate::substrate::Weights::F32(vec![10.0]),
            root_ids: vec![100, 200],
            region: vec![0, 0],
            cell_type: vec![0, 0],
            nt_type: vec![0, 0],
            positions: Vec::new(),
        }
    }

    #[test]
    fn stdp_potentiates_pre_before_post() {
        let sub = Arc::new(tiny_substrate()); // ok
        let mut eng = Engine::new(sub, EngineConfig::default());
        eng.learning = Some(crate::engine::LearningState {
            lr: 0.1,
            window: 8,
            tau: 6.0,
            reward: 1.0,
        });
        // tick 1: neuron 0 spikes
        eng.inject(&[0], 100.0);
        eng.run_ticks(1, &mut |_, _| {});
        // tick 2: neuron 1 spikes (neuron 0 stays quiet)
        eng.inject(&[1], 100.0);
        eng.run_ticks(1, &mut |_, _| {});
        let dw = eng
            .overlay
            .get(&0)
            .and_then(|m| m.get(&1))
            .copied()
            .unwrap_or(0.0);
        assert!(dw > 0.0, "expected LTP delta, got {dw}");
    }
}
