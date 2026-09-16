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
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            episodes: default_episodes(),
            steps_per_episode: default_steps_per_episode(),
            stim_region: None,
            stim_current: default_stim_current(),
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
