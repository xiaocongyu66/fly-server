//! Pluggable GPU backends for the LIF tick.
//!
//! Architecture (inspired by fly-brain's backend dispatch and Neurokernel):
//! neuron state persists in GPU buffers between dispatches, the engine owns
//! one `Box<dyn GpuBackend>` selected at runtime, and any init/tick failure
//! falls back to the CPU path. The CPU+NEON path remains the bit-exact
//! ground truth; GPU kernels share the same tick semantics — decay of the
//! cumulative conductances, scatter of the previous tick's spikes (1-tick
//! synaptic delay), then membrane integration.
//!
//! Backends:
//! - `wgpu` (feature `gpu`): Vulkan / Metal / DX12 / GL — from desktop
//!   GPUs down to Adreno phones via Mesa freedreno
//! - `cuda` (feature `cuda`): kernels compiled at runtime via NVRTC,
//!   driver API dlopen'd — present only on NVIDIA machines
//!
//! `GpuMode::Auto` tries wgpu then cuda; `Off` skips GPU entirely. The
//! admin Settings page lists what `probe()` finds on the current device.

#[cfg(feature = "cuda")]
pub mod cuda_backend;
#[cfg(feature = "gpu")]
pub mod wgpu_backend;

use crate::engine::EngineConfig;
use crate::substrate::Substrate;

/// Per-tick kernel parameters (the values the CPU path recomputes from
/// EngineConfig every tick).
#[derive(Debug, Clone, Copy)]
pub struct TickParams {
    pub decay_exc: f32,
    pub decay_inh: f32,
    /// 1 - leak factor.
    pub a: f32,
    pub v_rest: f32,
    pub input_gain: f32,
    pub v_thresh: f32,
    pub v_reset: f32,
    pub weight_scale: f32,
}

/// Result of one GPU tick.
pub struct GpuTick {
    /// Neurons that spiked this tick; scattered by the next tick
    /// (identical 1-tick delay to the CPU path).
    pub spikes: Vec<u32>,
}

/// One GPU backend instance. All neuron state lives inside the backend;
/// the engine mirrors membrane potentials out after every tick for the
/// readout layer.
pub trait GpuBackend: Send {
    /// Short backend id ("wgpu", "cuda") — reported by the settings API.
    fn name(&self) -> &'static str;
    /// Human-readable device description, e.g. "Adreno (TM) 750".
    fn device(&self) -> String;
    /// Queue `v[i] += current` injections; applied before the next tick's
    /// integrate. Bounded by the backend's pending-injection capacity.
    fn inject(&mut self, neurons: &[u32], current: f32);
    /// Push the current membrane state into the device (called once when
    /// the backend starts, so CPU-side injections carry over).
    fn write_state(&mut self, v: &[f32]) -> Result<(), String>;
    /// One full tick. On error the engine drops the backend and falls back
    /// to CPU; state continuity is lost either way.
    fn tick(&mut self, p: &TickParams) -> Result<GpuTick, String>;
    /// Mirror the membrane potentials into `out` (len == n_neurons).
    fn read_v(&mut self, out: &mut [f32]) -> Result<(), String>;
}

/// User-selectable GPU mode (`EngineConfig.use_gpu`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuMode {
    /// Try every compiled-in backend in turn, then CPU.
    #[default]
    Auto,
    Wgpu,
    Cuda,
    Off,
}

/// A GPU backend the current device could actually use.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuProbe {
    pub backend: &'static str,
    pub device: String,
}

/// Cheap device probe for the Settings page: enumerate what is reachable
/// without allocating simulation buffers. Cached after first call.
pub fn probe() -> Vec<GpuProbe> {
    static CACHE: std::sync::OnceLock<Vec<GpuProbe>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| {
            // `mut` is needed only when a backend feature is compiled in
            #[allow(unused_mut)]
            let mut out = Vec::new();
            #[cfg(feature = "gpu")]
            if let Some(p) = wgpu_backend::probe() {
                out.push(p);
            }
            #[cfg(feature = "cuda")]
            if let Some(p) = cuda_backend::probe() {
                out.push(p);
            }
            out
        })
        .clone()
}

/// Bring up the requested backend, or explain why none is available.
pub fn select_backend(
    mode: GpuMode,
    sub: &Substrate,
    is_inhibitory: &[bool],
    cfg: &EngineConfig,
) -> Result<Box<dyn GpuBackend>, String> {
    match mode {
        GpuMode::Off => Err("gpu disabled".into()),
        GpuMode::Wgpu => {
            #[cfg(feature = "gpu")]
            {
                wgpu_backend::init(sub, is_inhibitory, cfg)
                    .map(|b| Box::new(b) as Box<dyn GpuBackend>)
            }
            #[cfg(not(feature = "gpu"))]
            {
                let _ = (sub, is_inhibitory, cfg);
                Err("built without the `gpu` feature (no wgpu backend)".into())
            }
        }
        GpuMode::Cuda => Err("cuda backend kernel launches pending — use wgpu or off".into()),
        GpuMode::Auto => {
            #[cfg(feature = "gpu")]
            {
                if let Ok(b) = wgpu_backend::init(sub, is_inhibitory, cfg) {
                    return Ok(Box::new(b));
                }
            }
            Err(
                "no GPU backend available (wgpu: no adapter; cuda: not built in or no driver)"
                    .into(),
            )
        }
    }
}
