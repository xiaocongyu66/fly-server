//! Shared API wire types (serde). Single source of truth for all modules.
//! Field names follow OpenAI conventions; snake_case JSON.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------- sessions ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionObject {
    pub id: String,      // "sess_..."
    pub object: String,  // "session"
    pub created_at: u64, // unix seconds
    pub substrate: String,
    pub adapters: Vec<String>,
    pub dt_ms: f32,
    pub metadata: HashMap<String, String>,
    pub current_tick: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CreateSessionRequest {
    pub substrate: String,
    #[serde(default)]
    pub adapters: Vec<String>,
    #[serde(default)]
    pub dt_ms: Option<f32>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpdateSessionRequest {
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

// ---------- neuron selectors ----------

/// Select neurons by root id list or by metadata class. Root ids win if both given.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct NeuronSelector {
    #[serde(default)]
    pub ids: Vec<u64>,
    #[serde(default)]
    pub region: Option<String>,
    #[serde(default)]
    pub cell_type: Option<String>,
    #[serde(default)]
    pub nt_type: Option<String>,
    /// Position 0.0-1.0 within the matched set (retinotopic slice start).
    #[serde(default)]
    pub offset: Option<f32>,
    /// True retinotopy: azimuth 0.0-1.0 around the head center; selects
    /// neurons whose soma sits at that bearing (needs v3 positions).
    #[serde(default)]
    pub retina: Option<f32>,
    #[serde(default)]
    pub limit: Option<u32>,
}

// ---------- observe / step ----------

/// One azimuth slice of a visual frame (llama.cpp mtmd-style: frame →
/// patches → per-patch injection, here into optic-lobe columns).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RetinaColumn {
    /// Azimuth 0.0-1.0 (0 = +x axis, clockwise seen from +y).
    pub az: f32,
    /// Injected current for this column's neurons.
    pub current: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ObserveRequest {
    /// Visual frame: multiple azimuth columns injected in one call.
    #[serde(default)]
    pub frame: Vec<RetinaColumn>,
    /// "current" | "pain" | "reward" — semantic tag recorded in the item log;
    /// current source is the same, pain/reward may later map to modulatory neurons.
    pub modality: String,
    pub target: NeuronSelector,
    /// Injected current per selected neuron (mV-ish units).
    pub current: f32,
    /// Repeat injection for this many ticks (default 1).
    #[serde(default)]
    pub duration_ticks: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct Action {
    /// Root id of the readout (descending/motor) neuron.
    pub neuron_id: u64,
    /// Normalized firing rate 0..1 from the readout adapter.
    pub rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    pub ticks_simulated: u64,
    /// Ticks restored from snapshot on fork (the "cache hit" of simulation).
    pub ticks_reused: u64,
    pub snapshot_writes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepResponse {
    pub object: String, // "session.step"
    pub session_id: String,
    pub tick: u64,
    pub t_ms: f32,
    pub n_spikes: u32,
    pub actions: Vec<Action>,
    pub previous_tick_id: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StepRequest {
    #[serde(default = "default_steps")]
    pub steps: u32,
}
fn default_steps() -> u32 {
    1
}

// ---------- items (experiment log, paginated) ----------

#[derive(Debug, Clone, Serialize)]
pub struct SessionItem {
    pub id: String,     // "item_..."
    pub object: String, // "session.item"
    pub session_id: String,
    pub created_at: u64,
    pub tick: u64,
    pub kind: String, // "observe" | "step" | "fork" | "adapter_load"
    pub body: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct ListItemsResponse {
    pub object: String, // "list"
    pub data: Vec<SessionItem>,
    pub has_more: bool,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ListItemsQuery {
    pub limit: Option<u32>,
    pub after: Option<String>,
    pub order: Option<String>, // "asc" | "desc" (default desc)
}

// ---------- fork (snapshot reuse = cache hit) ----------

#[derive(Debug, Clone, Deserialize)]
pub struct ForkRequest {
    pub source_session: String,
    /// Fork state as of this tick (must be a snapshotted tick).
    pub at_tick: u64,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

// ---------- models ----------

#[derive(Debug, Clone, Serialize)]
pub struct SubstrateObject {
    pub id: String,
    pub object: String, // "substrate"
    pub n_neurons: u32,
    pub n_edges: u64,
    pub source: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdapterObject {
    pub id: String,     // "adp_..."
    pub object: String, // "adapter"
    pub kind: String,   // "readout" | "gains" | "sensory"
    pub substrate: String,
    pub trained_ticks: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelsResponse {
    pub object: String, // "list"
    pub data: Vec<serde_json::Value>,
}

// ---------- activity stream ----------

#[derive(Debug, Clone, Serialize)]
pub struct ActivityEvent {
    pub session_id: String,
    pub tick: u64,
    pub t_ms: f32,
    pub n_spikes: u32,
    /// Sampled spiking neuron root ids (capped).
    pub spike_sample: Vec<u64>,
}
