//! LIF simulation engine over a compiled substrate.
//!
//! State lives in `n_threads` contiguous chunks, each behind a Mutex, so a
//! single unified worker loop runs serial (one chunk, zero contention) or
//! parallel (`--threads N`, scoped workers with barrier phases):
//!
//!   decay (own chunk) -> scatter (thread-private dense buffers + touched
//!   list) -> drain touched items into global g chunks (per-chunk Mutex)
//!   -> integrate (own chunk).
//!
//! Determinism contract: `n_threads == 1` replays are bit-identical. With
//! `n_threads > 1` the drain order changes float accumulation order, so
//! outputs may drift by rounding (statistically equivalent).

use std::sync::{Arc, Barrier, Mutex};

use crate::substrate::Substrate;

pub mod gpu;
pub mod simd;

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
    /// Use NEON f32x4 kernels (aarch64) for decay/integrate. Bit-identical
    /// to scalar; disable for A/B comparison.
    pub use_simd: bool,
    /// Worker threads for the tick loop (1 = serial, bit-exact replays).
    pub n_threads: usize,
    /// GPU backend selection (auto probes wgpu/cuda, falls back to CPU).
    #[serde(default)]
    pub use_gpu: gpu::GpuMode,
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
            use_simd: true,
            n_threads: 1,
            use_gpu: gpu::GpuMode::Off,
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
        "GLUT"
        | "ACETYLCHOLINE"
        | "ACH"
        | "DOPAMINE"
        | "SEROTONINE"
        | "SEROTONIN"
        | "OCTOPAMINE"
        | "HISTAMINE"
        | "TYRAMINE"
        | "TYRAMINE-OCTOPAMINE" => Polarity::Excitatory,
        "GABA" | "GLYCINE" | "GLUTINHIB" => Polarity::Inhibitory,
        _ => Polarity::Unknown,
    }
}

/// Read-only view handed to per-tick callbacks (actions, SSE, snapshots).
pub trait EngineView {
    fn membrane(&self, i: usize) -> f32;
    fn state_bytes(&self) -> Vec<u8>;
    fn spikes(&self) -> &[u32];
}

impl EngineView for Engine {
    fn membrane(&self, i: usize) -> f32 {
        Engine::membrane(self, i)
    }
    fn state_bytes(&self) -> Vec<u8> {
        Engine::state_bytes(self)
    }
    fn spikes(&self) -> &[u32] {
        Engine::last_spikes(self)
    }
}

pub struct Engine {
    pub substrate: Arc<Substrate>,
    pub cfg: EngineConfig,
    polarity: Vec<Polarity>,
    tick: u64,
    t_ms: f32,
    /// Postsynaptic targets spiked last tick (global neuron indices).
    last_spiked: Vec<u32>,
    n_chunks: usize,
    chunk_len: usize,
    // SoA state, chunked: g_exc[chunk][local]
    v: Vec<Mutex<Vec<f32>>>,
    g_exc: Vec<Mutex<Vec<f32>>>,
    g_inh: Vec<Mutex<Vec<f32>>>,
    spikes_last: u32,
    gpu: Option<Box<dyn gpu::GpuBackend>>,
    gpu_built: gpu::GpuMode,
    is_inhibitory: Vec<bool>,
    v_mirror: Vec<f32>,
}

/// Per-thread scratch for the parallel scatter phase (owned by the worker).
struct Scratch {
    lex: Vec<f32>,
    lin: Vec<f32>,
    touched: Vec<u32>,
    bitmap: Vec<u64>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Self {
            lex: vec![0.0; n],
            lin: vec![0.0; n],
            touched: Vec::with_capacity(4096),
            bitmap: vec![0u64; n.div_ceil(64)],
        }
    }
    fn reset(&mut self) {
        for &p in &self.touched {
            self.lex[p as usize] = 0.0;
            self.lin[p as usize] = 0.0;
        }
        for &p in &self.touched {
            let p = p as usize;
            self.bitmap[p >> 6] &= !(1u64 << (p & 63));
        }
        self.touched.clear();
    }
    #[inline]
    fn touch(&mut self, p: usize) {
        if self.bitmap[p >> 6] & (1u64 << (p & 63)) == 0 {
            self.bitmap[p >> 6] |= 1u64 << (p & 63);
            self.touched.push(p as u32);
        }
    }
}

impl Engine {
    pub fn new(substrate: Arc<Substrate>, mut cfg: EngineConfig) -> Self {
        let n = substrate.n_neurons();
        let n_threads = cfg.n_threads.max(1).min(n.max(1));
        let chunk_len = n.div_ceil(n_threads);
        cfg.n_threads = n_threads;
        // Neuron polarity: the presynaptic neuron's NT decides whether its
        // outgoing synapses excite or inhibit. Unknown NTs default excitatory.
        let table: Vec<Polarity> = substrate
            .header
            .string_tables
            .nt_types
            .iter()
            .map(|s| nt_polarity(s))
            .collect();
        let polarity: Vec<Polarity> = substrate
            .nt_type
            .iter()
            .map(|&t| table[t as usize])
            .collect();
        let is_inhibitory: Vec<bool> = polarity
            .iter()
            .map(|p| matches!(p, Polarity::Inhibitory))
            .collect();
        let mk = || {
            (0..n_threads)
                .map(|_| Mutex::new(vec![0.0f32; chunk_len]))
                .collect()
        };
        Self {
            substrate,
            cfg,
            polarity,
            tick: 0,
            t_ms: 0.0,
            last_spiked: Vec::with_capacity(4096),
            n_chunks: n_threads,
            chunk_len,
            v: mk(),
            g_exc: mk(),
            g_inh: mk(),
            spikes_last: 0,
            gpu: None,
            gpu_built: gpu::GpuMode::Off,
            is_inhibitory,
            v_mirror: Vec::new(),
        }
    }

    fn locate(&self, i: usize) -> (usize, usize) {
        (i / self.chunk_len, i % self.chunk_len)
    }

    /// Inject current into specific neurons (sensory stimulus, pain, reward).
    pub fn inject(&mut self, neurons: &[u32], current: f32) {
        if let Some(b) = self.gpu.as_mut() {
            b.inject(neurons, current);
        }
        for &i in neurons {
            let (c, l) = self.locate(i as usize);
            if let Ok(mut ch) = self.v[c].try_lock() {
                if l < ch.len() {
                    ch[l] += current;
                }
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

    /// Membrane potential of neuron `i` (for readout rate computation).
    pub fn membrane(&self, i: usize) -> f32 {
        if self.gpu.is_some() {
            return self.v_mirror.get(i).copied().unwrap_or(0.0);
        }
        let (c, l) = self.locate(i);
        self.v[c]
            .lock()
            .map(|ch| ch.get(l).copied().unwrap_or(0.0))
            .unwrap_or(0.0)
    }

    /// Run `steps` ticks. `per_tick` is invoked after every tick with
    /// (report, read-only engine view). `steps == 0` is a no-op returning
    /// the current state as a report.
    pub fn run_ticks(
        &mut self,
        steps: u32,
        per_tick: &mut dyn FnMut(&TickReport, &dyn EngineView),
    ) -> TickReport {
        let current = TickReport {
            tick: self.tick,
            t_ms: self.t_ms,
            n_spikes: self.spikes_last,
            mean_v: 0.0,
        };
        let mut last = current;
        self.ensure_gpu();
        if self.gpu.is_some() {
            let params = self.gpu_params();
            let mut ran = 0u32;
            while ran < steps {
                // llvmpipe and flaky drivers can panic (or worse) inside a
                // dispatch; contain it so the server falls back to CPU
                // instead of dying with the session thread
                let step_res = match self.gpu.as_mut() {
                    Some(b) => {
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| b.tick(&params)))
                            .unwrap_or_else(|_| {
                                Err("gpu panic (driver crash) — falling back to CPU".into())
                            })
                    }
                    None => break,
                };
                match step_res {
                    Ok(t) => {
                        self.last_spiked = t.spikes;
                        self.spikes_last = self.last_spiked.len() as u32;
                        self.tick += 1;
                        self.t_ms += self.cfg.dt_ms;
                        let n = self.substrate.n_neurons();
                        if self.v_mirror.len() != n {
                            self.v_mirror.resize(n, 0.0);
                        }
                        let read_ok = match self.gpu.as_mut() {
                            Some(b) => {
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    b.read_v(&mut self.v_mirror)
                                }))
                                .map(|r| r.is_ok())
                                .unwrap_or(false)
                            }
                            None => false,
                        };
                        if !read_ok {
                            eprintln!("gpu: read_v failed, falling back to CPU");
                            self.gpu = None;
                            break;
                        }
                        let mean_v = if n > 0 {
                            self.v_mirror.iter().sum::<f32>() / n as f32
                        } else {
                            0.0
                        };
                        let r = TickReport {
                            tick: self.tick,
                            t_ms: self.t_ms,
                            n_spikes: self.spikes_last,
                            mean_v,
                        };
                        per_tick(&r, self);
                        last = r;
                        ran += 1;
                    }
                    Err(e) => {
                        eprintln!("gpu: tick failed, falling back to CPU: {e}");
                        self.gpu = None;
                        break;
                    }
                }
            }
            if self.gpu.is_some() && ran == steps {
                return last;
            }
            // GPU dropped mid-run (or partially ran): remaining steps take
            // the CPU paths below. Backend switches reset simulation state
            // by design.
        }
        if self.cfg.n_threads <= 1 {
            for _ in 0..steps {
                let r = self.tick_one_serial();
                per_tick(&r, self);
                last = r;
            }
        } else if steps > 0 {
            last = self.run_ticks_parallel(steps, per_tick);
        }
        last
    }

    fn gpu_params(&self) -> gpu::TickParams {
        let c = &self.cfg;
        gpu::TickParams {
            decay_exc: (-c.dt_ms / c.tau_exc_ms).exp(),
            decay_inh: (-c.dt_ms / c.tau_inh_ms).exp(),
            a: 1.0 - (-c.dt_ms / c.tau_mem_ms).exp(),
            v_rest: c.v_rest,
            input_gain: c.input_gain,
            v_thresh: c.v_thresh,
            v_reset: c.v_reset,
            weight_scale: c.weight_scale,
        }
    }

    /// (Re)build the GPU backend when `cfg.use_gpu` changed. On success the
    /// CPU-side membrane state (accumulated injections) is uploaded so the
    /// device continues from where the CPU left off.
    fn ensure_gpu(&mut self) {
        if self.cfg.use_gpu == self.gpu_built {
            return;
        }
        self.gpu = match gpu::select_backend(
            self.cfg.use_gpu,
            &self.substrate,
            &self.is_inhibitory,
            &self.cfg,
        ) {
            Ok(mut b) => {
                let n = self.substrate.n_neurons();
                let mut v = vec![0f32; n];
                for (c, chunk) in self.v.iter().enumerate() {
                    if let Ok(ch) = chunk.lock() {
                        let base = c * self.chunk_len;
                        for (l, &val) in ch.iter().enumerate() {
                            if base + l < n {
                                v[base + l] = val;
                            }
                        }
                    }
                }
                match b.write_state(&v) {
                    Ok(()) => {
                        eprintln!("gpu: backend {} active on {}", b.name(), b.device());
                        Some(b)
                    }
                    Err(e) => {
                        eprintln!("gpu: write_state failed, staying on CPU: {e}");
                        None
                    }
                }
            }
            Err(e) => {
                if self.cfg.use_gpu != gpu::GpuMode::Off {
                    eprintln!("gpu: no backend ({e}) — using CPU");
                }
                None
            }
        };
        self.gpu_built = self.cfg.use_gpu;
    }

    // ---- serial path (n_threads == 1): bit-exact replays ----

    fn tick_one_serial(&mut self) -> TickReport {
        let cfg = self.cfg.clone();
        let decay_exc = (-cfg.dt_ms / cfg.tau_exc_ms).exp();
        let decay_inh = (-cfg.dt_ms / cfg.tau_inh_ms).exp();
        let leak = (-cfg.dt_ms / cfg.tau_mem_ms).exp();
        let n = self.substrate.n_neurons();

        {
            let mut ge = self.g_exc[0].lock().unwrap();
            let mut gi = self.g_inh[0].lock().unwrap();
            if cfg.use_simd {
                simd::decay(&mut ge, decay_exc);
                simd::decay(&mut gi, decay_inh);
            } else {
                for x in ge.iter_mut() {
                    *x *= decay_exc;
                }
                for x in gi.iter_mut() {
                    *x *= decay_inh;
                }
            }
        }

        {
            let mut ge = self.g_exc[0].lock().unwrap();
            let mut gi = self.g_inh[0].lock().unwrap();
            let w_scale = cfg.weight_scale;
            for &pre in &self.last_spiked {
                match self.polarity[pre as usize] {
                    Polarity::Inhibitory => {
                        for (post, w) in self.substrate.outgoing(pre as usize) {
                            gi[post as usize] += w * w_scale;
                        }
                    }
                    _ => {
                        for (post, w) in self.substrate.outgoing(pre as usize) {
                            ge[post as usize] += w * w_scale;
                        }
                    }
                }
            }
        }

        let (new_spikes, spike_count, v_sum) = {
            let mut vv = self.v[0].lock().unwrap();
            let ge = self.g_exc[0].lock().unwrap();
            let gi = self.g_inh[0].lock().unwrap();
            if cfg.use_simd {
                let (spikes, v_sum) = simd::integrate(
                    &mut vv,
                    &ge,
                    &gi,
                    cfg.v_rest,
                    1.0 - leak,
                    cfg.input_gain,
                    cfg.v_thresh,
                    cfg.v_reset,
                );
                let c = spikes.len() as u32;
                (spikes, c, v_sum)
            } else {
                let mut spikes = Vec::new();
                let mut v_sum = 0f32;
                for i in 0..n {
                    vv[i] += (cfg.v_rest - vv[i]) * (1.0 - leak) + cfg.input_gain * (ge[i] - gi[i]);
                    v_sum += vv[i];
                    if vv[i] >= cfg.v_thresh {
                        vv[i] = cfg.v_reset;
                        spikes.push(i as u32);
                    }
                }
                let c = spikes.len() as u32;
                (spikes, c, v_sum)
            }
        };

        self.tick += 1;
        self.t_ms += cfg.dt_ms;
        self.spikes_last = spike_count;
        self.last_spiked = new_spikes;
        TickReport {
            tick: self.tick,
            t_ms: self.t_ms,
            n_spikes: spike_count,
            mean_v: if n > 0 { v_sum / n as f32 } else { 0.0 },
        }
    }

    // ---- parallel path ----

    fn run_ticks_parallel(
        &mut self,
        steps: u32,
        per_tick: &mut dyn FnMut(&TickReport, &dyn EngineView),
    ) -> TickReport {
        let n_threads = self.cfg.n_threads;
        let n = self.substrate.n_neurons();
        let chunk_len = self.chunk_len;
        let decay_exc = (-self.cfg.dt_ms / self.cfg.tau_exc_ms).exp();
        let decay_inh = (-self.cfg.dt_ms / self.cfg.tau_inh_ms).exp();
        let leak = (-self.cfg.dt_ms / self.cfg.tau_mem_ms).exp();
        let a = 1.0 - leak;
        let (rest, gain, thresh, reset, w_scale, use_simd) = (
            self.cfg.v_rest,
            self.cfg.input_gain,
            self.cfg.v_thresh,
            self.cfg.v_reset,
            self.cfg.weight_scale,
            self.cfg.use_simd,
        );

        let last_snap: Arc<Mutex<Vec<u32>>> =
            Arc::new(Mutex::new(std::mem::take(&mut self.last_spiked)));
        let barriers: Vec<Arc<Barrier>> =
            (0..4).map(|_| Arc::new(Barrier::new(n_threads))).collect();
        let spikes_pool: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
        let vsum_pool: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

        let mut last = TickReport {
            tick: self.tick,
            t_ms: self.t_ms,
            n_spikes: 0,
            mean_v: 0.0,
        };

        std::thread::scope(|s| {
            let substrate: &Substrate = &self.substrate;
            let polarity: &[Polarity] = &self.polarity;
            let g_exc: &[Mutex<Vec<f32>>] = &self.g_exc;
            let g_inh: &[Mutex<Vec<f32>>] = &self.g_inh;
            let v: &[Mutex<Vec<f32>>] = &self.v;
            let n_chunks = self.n_chunks;

            let mut handles = Vec::new();
            for tid in 1..n_threads {
                let (last_snap, barriers, spikes_pool, vsum_pool) = (
                    last_snap.clone(),
                    barriers.clone(),
                    spikes_pool.clone(),
                    vsum_pool.clone(),
                );
                let mut scratch = Scratch::new(n);
                handles.push(s.spawn(move || {
                    for _ in 0..steps {
                        tick_phases(
                            tid,
                            n_threads,
                            n,
                            chunk_len,
                            n_chunks,
                            substrate,
                            polarity,
                            g_exc,
                            g_inh,
                            v,
                            &last_snap,
                            &barriers,
                            &mut scratch,
                            decay_exc,
                            decay_inh,
                            a,
                            rest,
                            gain,
                            thresh,
                            reset,
                            use_simd,
                            w_scale,
                            &spikes_pool,
                            &vsum_pool,
                        );
                    }
                }));
            }

            // worker 0 owns per_tick and the engine's tick bookkeeping
            let mut scratch = Scratch::new(n);
            let mut local_tick = self.tick;
            let mut local_t_ms = self.t_ms;
            for _ in 0..steps {
                tick_phases(
                    0,
                    n_threads,
                    n,
                    chunk_len,
                    n_chunks,
                    substrate,
                    polarity,
                    g_exc,
                    g_inh,
                    v,
                    &last_snap,
                    &barriers,
                    &mut scratch,
                    decay_exc,
                    decay_inh,
                    a,
                    rest,
                    gain,
                    thresh,
                    reset,
                    use_simd,
                    w_scale,
                    &spikes_pool,
                    &vsum_pool,
                );
                let all_spikes = { std::mem::take(&mut *spikes_pool.lock().unwrap()) };
                let mut flat: Vec<u32> = Vec::new();
                for chunk in &all_spikes {
                    flat.extend_from_slice(chunk);
                }
                let v_sum: f32 = { vsum_pool.lock().unwrap().drain(..).sum() };
                local_tick += 1;
                local_t_ms += self.cfg.dt_ms;
                self.tick = local_tick;
                self.t_ms = local_t_ms;
                self.spikes_last = flat.len() as u32;
                self.last_spiked = flat.clone();
                // publish next tick's scatter input (workers read last_snap)
                *last_snap.lock().unwrap() = flat;
                let report = TickReport {
                    tick: self.tick,
                    t_ms: self.t_ms,
                    n_spikes: self.spikes_last,
                    mean_v: if n > 0 { v_sum / n as f32 } else { 0.0 },
                };
                per_tick(&report, self);
                last = report;
            }
            for h in handles {
                let _ = h.join();
            }
        });
        last
    }

    // ---- state snapshots ----

    /// Serialize simulation state for snapshotting (fork reuse).
    pub fn state_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        for field in [&self.v, &self.g_exc, &self.g_inh] {
            let total: usize = field.iter().map(|c| c.lock().unwrap().len()).sum();
            b.extend_from_slice(&(total as u64).to_le_bytes());
            for ch in field {
                for x in ch.lock().unwrap().iter() {
                    b.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        b.extend_from_slice(&(self.last_spiked.len() as u64).to_le_bytes());
        for x in &self.last_spiked {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b.extend_from_slice(&self.tick.to_le_bytes());
        b.extend_from_slice(&self.t_ms.to_le_bytes());
        b.extend_from_slice(&self.spikes_last.to_le_bytes());
        b
    }

    /// Restore state from `state_bytes`. Lengths must match the substrate.
    pub fn restore_state(&mut self, b: &[u8]) -> Result<(), String> {
        let mut off = 0usize;
        for field in [&self.v, &self.g_exc, &self.g_inh] {
            for ch in field {
                let mut dst = ch.lock().unwrap();
                for x in dst.iter_mut() {
                    if off + 4 > b.len() {
                        return Err("truncated state".into());
                    }
                    *x = f32::from_le_bytes(b[off..off + 4].try_into().unwrap());
                    off += 4;
                }
            }
        }
        if b.len() < off + 8 {
            return Err("truncated state".into());
        }
        let ns = u64::from_le_bytes(b[off..off + 8].try_into().unwrap()) as usize;
        off += 8;
        if b.len() < off + ns * 4 {
            return Err("truncated state".into());
        }
        self.last_spiked = b[off..off + ns * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        off += ns * 4;
        if b.len() < off + 20 {
            return Err("truncated state".into());
        }
        self.tick = u64::from_le_bytes(b[off..off + 8].try_into().unwrap());
        self.t_ms = f32::from_le_bytes(b[off + 8..off + 12].try_into().unwrap());
        self.spikes_last = u32::from_le_bytes(b[off + 12..off + 16].try_into().unwrap());
        Ok(())
    }
}

/// One tick's four barrier-synchronized phases, shared by every worker.
/// The driving thread (worker 0) calls this once per tick and handles
/// report construction + per_tick between calls.
#[allow(clippy::too_many_arguments)]
fn tick_phases(
    tid: usize,
    t_total: usize,
    _n: usize,
    chunk_len: usize,
    n_chunks: usize,
    substrate: &Substrate,
    polarity: &[Polarity],
    g_exc: &[Mutex<Vec<f32>>],
    g_inh: &[Mutex<Vec<f32>>],
    v: &[Mutex<Vec<f32>>],
    last_snap: &Mutex<Vec<u32>>,
    barriers: &[Arc<Barrier>],
    scratch: &mut Scratch,
    decay_exc: f32,
    decay_inh: f32,
    a: f32,
    rest: f32,
    gain: f32,
    thresh: f32,
    reset: f32,
    use_simd: bool,
    w_scale: f32,
    spikes_pool: &Mutex<Vec<Vec<u32>>>,
    vsum_pool: &Mutex<Vec<f32>>,
) {
    // ---- phase 0: decay (own chunk) ----
    let _ = barriers[0].wait();
    {
        let mut ge = g_exc[tid].lock().unwrap();
        let mut gi = g_inh[tid].lock().unwrap();
        if use_simd {
            simd::decay(&mut ge, decay_exc);
            simd::decay(&mut gi, decay_inh);
        } else {
            for x in ge.iter_mut() {
                *x *= decay_exc;
            }
            for x in gi.iter_mut() {
                *x *= decay_inh;
            }
        }
    }
    // ---- phase 1: scatter into private buffers ----
    let _ = barriers[1].wait();
    {
        let snap = last_snap.lock().unwrap();
        let total = snap.len();
        let blk = total.div_ceil(t_total);
        let begin = (tid * blk).min(total);
        let end = ((tid + 1) * blk).min(total);
        scratch.reset();
        for &pre in &snap[begin..end] {
            match polarity[pre as usize] {
                Polarity::Inhibitory => {
                    for (post, w) in substrate.outgoing(pre as usize) {
                        let p = post as usize;
                        scratch.touch(p);
                        scratch.lin[p] += w * w_scale;
                    }
                }
                _ => {
                    for (post, w) in substrate.outgoing(pre as usize) {
                        let p = post as usize;
                        scratch.touch(p);
                        scratch.lex[p] += w * w_scale;
                    }
                }
            }
        }
    }
    // ---- phase 2: drain touched items into global g chunks ----
    let _ = barriers[2].wait();
    {
        // bucket touched by chunk, then one lock per non-empty chunk
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n_chunks];
        for &p in &scratch.touched {
            buckets[(p as usize) / chunk_len].push(p);
        }
        for (c, bucket) in buckets.iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let mut ge = g_exc[c].lock().unwrap();
            let mut gi = g_inh[c].lock().unwrap();
            for &p in bucket {
                let l = p as usize - c * chunk_len;
                ge[l] += scratch.lex[p as usize];
                gi[l] += scratch.lin[p as usize];
            }
        }
    }
    // ---- phase 3: integrate (own chunk) ----
    let _ = barriers[3].wait();
    {
        let mut vv = v[tid].lock().unwrap();
        let ge = g_exc[tid].lock().unwrap();
        let gi = g_inh[tid].lock().unwrap();
        let (sp, _n_spikes_chunk, sum) = if use_simd {
            let (spikes, v_sum) = simd::integrate(&mut vv, &ge, &gi, rest, a, gain, thresh, reset);
            let c = spikes.len() as u32;
            (spikes, c, v_sum)
        } else {
            let mut spikes = Vec::new();
            let mut v_sum = 0f32;
            for i in 0..vv.len() {
                vv[i] += (rest - vv[i]) * a + gain * (ge[i] - gi[i]);
                v_sum += vv[i];
                if vv[i] >= thresh {
                    vv[i] = reset;
                    spikes.push(i as u32);
                }
            }
            let c = spikes.len() as u32;
            (spikes, c, v_sum)
        };
        let base = (tid * chunk_len) as u32;
        let global: Vec<u32> = sp.iter().map(|&i| base + i).collect();
        vsum_pool.lock().unwrap().push(sum);
        spikes_pool.lock().unwrap().push(global);
    }
}
