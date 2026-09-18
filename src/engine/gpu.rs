//! wgpu GPU backend for LIF tick computation.
//!
//! Architecture (following ggml/llama.cpp backend pattern):
//! - All neuron state persists as GPU buffers between dispatches
//! - Each tick dispatches compute shaders (decay + integrate)
//! - Scatter stays on CPU initially (GPU atomics = future optimization)
//! - Auto-selection: try wgpu, fall back to CPU on any error
//!
//! Enable with `--features gpu`. WGSL shader at shaders/lif_tick.wgsl.

/// Try to initialize GPU backend; always returns None when the `gpu`
/// feature is not enabled. When enabled, attempts wgpu init and falls
/// back to CPU on any error (driver missing, no Vulkan, etc).
pub fn try_init_gpu(_n: usize, _cfg: &crate::engine::EngineConfig) -> Option<()> {
    // TODO: implement when gpu feature is stable
    // 1. wgpu::Instance + request_adapter (HighPerformance)
    // 2. create compute pipelines from shaders/lif_tick.wgsl
    // 3. upload CSR buffers
    // 4. per-tick: dispatch_workgroups(decay) + dispatch(scatter) + dispatch(compute)
    // 5. read_back spike indices only
    None
}
