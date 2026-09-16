//! LIF simulation engine over a compiled substrate.
//!
//! Classic synchronous discrete-time LIF: every `dt`, decay synapse
//! conductances, scatter current from last tick's spikes through the CSR
//! weights (excitatory/inhibitory split by neurotransmitter type), integrate
//! membrane potential, threshold, spike. Simple, cache-friendly, and fast
//! enough: ~2.7M synapses per tick is single-digit milliseconds in release.

use std::sync::Arc;

use crate::substrate::Substrate;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EngineConfig {
    /// Simulation timestep in milliseconds.
    pub dt_ms: f32,
    /// Membrane time constant (ms).
    pub tau_mem_ms: f32,
    /// Excitatory conductance decay (ms).
    pub tau_exc_ms: f32,
    /// Inhibitory conductance decay (ms).
    pub tau_inh_ms: f32,
    /// Spike threshold (mV, relative to rest).
    pub v_thresh: f32,
    /// Post-spike reset potential (mV, relative to rest).
    pub v_reset: f32,
    /// Resting potential (mV, relative to rest; usually 0).
    pub v_rest: f32,
    /// Multiplier applied to raw syn_count weights.
    pub weight_scale: f32,
    /// Multiplier for excitatory current into membrane integration.
    pub input_gain: f32,
    /// External stimulus current per active sensory neuron.
    pub stim_current: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            dt_ms: 0.5,
            tau_mem_ms: 20.0,
            tau_exc_ms: 5.0,
            tau_inh_ms: 10.0,
            v_thresh: 15.0,
            v_reset: 0.0,
            v_rest: 0.0,
            weight_scale: 0.01,
            input_gain: 1.0,
            stim_current: 30.0,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TickReport {
    pub tick: u64,
    pub t_ms: f32,
    pub n_spikes: u32,
    /// Mean subthreshold potential, for quick diagnostics.
    pub mean_v: f32,
}

/// Neurotransmitter polarity per neuron (aggregated from its predicted NT).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    Excitatory,
    Inhibitory,
    Unknown,
}

/// Map a FlyWire predicted NT string to polarity.
pub fn nt_polarity(nt: &str) -> Polarity {
    match nt.to_uppercase().as_str() {
        "GLUT" | "ACETYLCHOLINE" | "ACH" | "DOPAMINE" | "SEROTONINE" | "SEROTONIN" | "OCTOPAMINE" | "HISTAMINE" | "TYRAMINE" | "TYRAMINE-OCTOPAMINE" => Polarity::Excitatory,
        "GABA" | "GLYCINE" | "GLUTINHIB" => Polarity::Inhibitory,
        _ => Polarity::Unknown,
    }
}

pub struct Engine {
    pub substrate: Arc<Substrate>,
    pub cfg: EngineConfig,
    polarity: Vec<Polarity>,
    tick: u64,
    t_ms: f32,
    /// Postsynaptic targets spiked last tick (indices into CSR rows of those pres).
    last_spiked: Vec<u32>,
    // SoA state
    v: Vec<f32>,
    g_exc: Vec<f32>,
    g_inh: Vec<f32>,
    spikes_last: u32,
}

impl Engine {
    pub fn new(substrate: Arc<Substrate>, cfg: EngineConfig) -> Self {
        let n = substrate.n_neurons();
        // Neuron polarity: the presynaptic neuron's NT decides whether its
        // outgoing synapses excite or inhibit. Unknown NTs default excitatory
        // but can be overridden per-adapter later.
        let polarity: Vec<Polarity> = substrate
            .header
            .string_tables
            .nt_types
            .iter()
            .enumerate()
            .map(|(i, name)| {
                // per-neuron lookup happens at scatter time; store per neuron
                let _ = i;
                nt_polarity(name)
            })
            .collect();
        let _ = polarity.len();
        Self {
            substrate,
            cfg,
            polarity: Vec::new(), // replaced below (per-neuron, not per-nt)
            tick: 0,
            t_ms: 0.0,
            last_spiked: Vec::with_capacity(4096),
            v: vec![0.0; n],
            g_exc: vec![0.0; n],
            g_inh: vec![0.0; n],
            spikes_last: 0,
        }
        .init_polarity()
    }

    fn init_polarity(mut self) -> Self {
        let nts = &self.substrate.nt_type;
        let table: Vec<Polarity> = self
            .substrate
            .header
            .string_tables
            .nt_types
            .iter()
            .map(|s| nt_polarity(s))
            .collect();
        self.polarity = nts.iter().map(|&t| table[t as usize]).collect();
        self
    }

    pub fn tick(&mut self) -> TickReport {
        let cfg = &self.cfg;
        let decay_exc = (-cfg.dt_ms / cfg.tau_exc_ms).exp();
        let decay_inh = (-cfg.dt_ms / cfg.tau_inh_ms).exp();
        let leak = (-cfg.dt_ms / cfg.tau_mem_ms).exp();

        let n = self.substrate.n_neurons();

        // 1. Conductance decay.
        for g in self.g_exc[..].iter_mut() {
            *g *= decay_exc;
        }
        for g in self.g_inh[..].iter_mut() {
            *g *= decay_inh;
        }

        // 2. Scatter currents from last tick's spikes through CSR rows.
        let spiked = std::mem::take(&mut self.last_spiked);
        let w_scale = cfg.weight_scale;
        for &pre in &spiked {
            for (post, w) in self.substrate.outgoing(pre as usize) {
                match self.polarity[pre as usize] {
                    Polarity::Inhibitory => self.g_inh[post as usize] += w * w_scale,
                    _ => self.g_exc[post as usize] += w * w_scale,
                }
            }
        }

        // 3. Integrate membrane + threshold.
        let (v, g_exc, g_inh) = (&mut self.v, &self.g_exc, &self.g_inh);
        let mut spike_count = 0u32;
        let mut v_sum = 0f32;
        for i in 0..n {
            v[i] += (cfg.v_rest - v[i]) * (1.0 - leak) + cfg.input_gain * (g_exc[i] - g_inh[i]);
            v_sum += v[i];
            if v[i] >= cfg.v_thresh {
                v[i] = cfg.v_reset;
                self.last_spiked.push(i as u32);
                spike_count += 1;
            }
        }

        self.tick += 1;
        self.t_ms += cfg.dt_ms;
        self.spikes_last = spike_count;
        TickReport {
            tick: self.tick,
            t_ms: self.t_ms,
            n_spikes: spike_count,
            mean_v: if n > 0 { v_sum / n as f32 } else { 0.0 },
        }
    }

    /// Inject current into specific neurons (sensory stimulus, pain, reward).
    pub fn inject(&mut self, neurons: &[u32], current: f32) {
        for &i in neurons {
            if (i as usize) < self.v.len() {
                self.v[i] += current;
            }
        }
    }

    /// Neurons that spiked on the most recent tick.
    pub fn last_spikes(&self) -> &[u32] {
        &self.last_spiked
    }

    pub fn tick_count(&self) -> u64 {
        self.tick
    }
    pub fn sim_time_ms(&self) -> f32 {
        self.t_ms
    }
}
