// LIF tick compute shaders (wgpu backend).
//
// State persists in GPU buffers between dispatches:
//   v        f32   — membrane potential
//   ge, gi   f32   — excitatory / inhibitory conductance (decaying, cumulative)
//   dge, dgi i32   — fixed-point (scale WQ) scatter deltas, zeroed after read
//   spiked   u32   — 0/1 flag per neuron, set by integrate, consumed by scatter
//   spikes   u32   — [0] = count, [1..] = spiked neuron indices (cap == n)
//
// One tick = dispatch(inject?) -> dispatch(scatter) -> dispatch(integrate),
// matching the CPU order: conductance decay + scatter of the previous
// tick's spikes (1-tick synaptic delay), then membrane integration.

struct Params {
    n: u32,
    nnz: u32,
    wg_x: u32,
    inj_count: u32,
    decay_exc: f32,
    decay_inh: f32,
    a: f32,            // 1 - leak
    v_rest: f32,
    input_gain: f32,
    v_thresh: f32,
    v_reset: f32,
    w_scale: f32,
    wq: f32,           // fixed-point scale for scatter deltas (1024)
    inv_wq: f32,       // 1 / wq
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> ge: array<f32>;
@group(0) @binding(3) var<storage, read_write> gi: array<f32>;
@group(0) @binding(4) var<storage, read_write> dge: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> dgi: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> spiked: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> spikes: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> src: array<u32>;
@group(0) @binding(9) var<storage, read> dst: array<u32>;
@group(0) @binding(10) var<storage, read> w: array<f32>;
@group(0) @binding(11) var<storage, read> epol: array<u32>;

@group(1) @binding(0) var<storage, read> inj_idx: array<u32>;
@group(1) @binding(1) var<storage, read> inj_val: array<f32>;

// ---- inject: one thread per queued injection --------------------------------
@compute @workgroup_size(64)
fn inject(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= P.inj_count) {
        return;
    }
    // indices are deduped+summed CPU-side, so plain RMW is race-free
    v[inj_idx[k]] = v[inj_idx[k]] + inj_val[k];
}

// ---- scatter: one thread per edge -------------------------------------------
// COO edges; current flows from neurons that spiked on the previous tick.
@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {
    // 2D dispatch: linear workgroup id across x×y, then the thread's edge
    let wg_lin = gid.y * P.wg_x + gid.x;
    let e = wg_lin * 64u + lid.x;
    if (e >= P.nnz) {
        return;
    }
    let pre = src[e];
    if (atomicLoad(&spiked[pre]) == 0u) {
        return;
    }
    let add = i32(round(w[e] * P.w_scale * P.wq));
    if (add == 0) {
        return;
    }
    let post = dst[e];
    if (epol[e] == 0u) {
        atomicAdd(&dge[post], add);
    } else {
        atomicAdd(&dgi[post], add);
    }
}

// ---- integrate: one thread per neuron ----------------------------------------
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.n) {
        return;
    }
    // clear the spike flag consumed by this tick's scatter; a spiking
    // neuron sets it again below for the next tick
    atomicStore(&spiked[i], 0u);
    let exc = f32(atomicLoad(&dge[i])) * P.inv_wq;
    let inh = f32(atomicLoad(&dgi[i])) * P.inv_wq;
    atomicStore(&dge[i], 0);
    atomicStore(&dgi[i], 0);
    var nv = v[i] + (P.v_rest - v[i]) * P.a + P.input_gain * (exc - inh);
    if (nv >= P.v_thresh) {
        nv = P.v_reset;
        let slot = atomicAdd(&spikes[0], 1u) + 1u;
        atomicStore(&spikes[slot], i);
    }
    v[i] = nv;
}
