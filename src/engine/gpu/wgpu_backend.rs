//! wgpu backend: Vulkan / Metal / DX12 / GL compute for the LIF tick.
//!
//! All state (v, ge, gi, spike flags) persists in GPU buffers; each tick
//! dispatches inject? → scatter → integrate and mirrors v + spikes back
//! through staging buffers. Works on desktop GPUs and phones (Adreno via
//! Mesa freedreno) — the same WGSL everywhere.

use super::{GpuBackend, GpuProbe, GpuTick, TickParams};
use crate::substrate::Substrate;
use pollster::block_on;
use wgpu::util::DeviceExt;

const WG: u32 = 64;
const WQ: f32 = 1024.0; // fixed-point scale for scatter deltas
const INJ_CAP: u32 = 65536;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    n: u32,
    nnz: u32,
    wg_x: u32,
    inj_count: u32,
    decay_exc: f32,
    decay_inh: f32,
    a: f32,
    v_rest: f32,
    input_gain: f32,
    v_thresh: f32,
    v_reset: f32,
    w_scale: f32,
    wq: f32,
    inv_wq: f32,
}

pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind0: wgpu::BindGroup,
    bind1: wgpu::BindGroup,
    pipe_inject: wgpu::ComputePipeline,
    pipe_scatter: wgpu::ComputePipeline,
    pipe_integrate: wgpu::ComputePipeline,
    v_buf: wgpu::Buffer,
    spikes_buf: wgpu::Buffer,
    inj_idx_buf: wgpu::Buffer,
    inj_val_buf: wgpu::Buffer,
    rb_spikes: wgpu::Buffer,
    rb_v: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    n: u32,
    nnz: u32,
    n_wg_int: u32,
    n_wg_sc: u32,
    device_name: String,
    pending_inj: Vec<(u32, f32)>,
}

/// Lightweight adapter probe for the Settings page (no buffers allocated).
/// Reports every Vulkan adapter, hardware first, software renderers last.
pub fn probe() -> Option<GpuProbe> {
    let probes = probe_all();
    probes.into_iter().next()
}

/// All Vulkan adapters on the device, hardware before software.
pub fn probe_all() -> Vec<GpuProbe> {
    let instance = wgpu::Instance::default();
    let mut out: Vec<GpuProbe> = instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .into_iter()
        .map(|a| {
            let info = a.get_info();
            GpuProbe {
                backend: "wgpu",
                device: format!(
                    "{} — {:?} / {:?}",
                    info.name,
                    info.backend,
                    type_label(info.device_type)
                ),
            }
        })
        .collect();
    // hardware adapters first so init prefers them over software Vulkan
    out.sort_by_key(|p| !is_hardware_label(&p.device));
    out
}

fn type_label(t: wgpu::DeviceType) -> &'static str {
    match t {
        wgpu::DeviceType::DiscreteGpu => "discrete GPU",
        wgpu::DeviceType::IntegratedGpu => "integrated GPU",
        wgpu::DeviceType::VirtualGpu => "virtual GPU",
        wgpu::DeviceType::Cpu => "software rendering",
        _ => "unknown",
    }
}

fn is_hardware_label(device: &str) -> bool {
    !device.contains("software rendering")
}

fn buf_u32(device: &wgpu::Device, usage: wgpu::BufferUsages, data: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

pub fn init(
    sub: &Substrate,
    is_inhibitory: &[bool],
    _cfg: &crate::engine::EngineConfig,
) -> Result<WgpuBackend, String> {
    let instance = wgpu::Instance::default();
    // hardware adapters first, software (llvmpipe) as last resort
    let mut adapters = instance.enumerate_adapters(wgpu::Backends::VULKAN);
    adapters.sort_by_key(|a| {
        matches!(
            a.get_info().device_type,
            wgpu::DeviceType::Cpu | wgpu::DeviceType::Other
        )
    });
    let adapter = adapters
        .into_iter()
        .next()
        .ok_or("no Vulkan adapter reachable")?;
    let info = adapter.get_info();
    let (device, queue) = block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("fly-lif"),
            // bgl0 binds 11 storage buffers — above the WebGPU-default 8
            required_limits: adapter.limits(),
            ..Default::default()
        },
        None,
    ))
    .map_err(|e| format!("gpu device request failed: {e}"))?;

    // ---- COO edge arrays from the substrate CSR --------------------------
    // polarity comes straight from the engine's table (single source of truth)
    let n = sub.n_neurons() as u32;
    let mut src: Vec<u32> = Vec::new();
    let mut dst: Vec<u32> = Vec::new();
    let mut w: Vec<f32> = Vec::new();
    let mut epol: Vec<u32> = Vec::new();
    for (pre, &inh) in is_inhibitory.iter().enumerate() {
        let p = u32::from(inh);
        for (post, weight) in sub.outgoing(pre) {
            src.push(pre as u32);
            dst.push(post);
            w.push(weight);
            epol.push(p);
        }
    }
    let nnz = src.len() as u32;
    if nnz == 0 {
        return Err("substrate has no edges — nothing to scatter".into());
    }

    // ---- buffers ----------------------------------------------------------
    let v_buf = device.create_buffer(&wgpu::BufferDescriptor {
        size: (n as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
        label: Some("v"),
    });
    let zeros_f = vec![0f32; n as usize];
    queue.write_buffer(&v_buf, 0, bytemuck::cast_slice(&zeros_f));
    let ge_buf = buf_u32(
        &device,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        &vec![0u32; n as usize],
    );
    let gi_buf = buf_u32(
        &device,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        &vec![0u32; n as usize],
    );
    let dge_buf = buf_u32(
        &device,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        &vec![0u32; n as usize],
    );
    let dgi_buf = buf_u32(
        &device,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        &vec![0u32; n as usize],
    );
    let spiked_buf = buf_u32(
        &device,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        &vec![0u32; n as usize],
    );
    let spikes_buf = device.create_buffer(&wgpu::BufferDescriptor {
        size: ((n as u64) + 1) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
        label: Some("spikes"),
    });
    queue.write_buffer(&spikes_buf, 0, bytemuck::cast_slice(&[0u32]));
    let src_buf = buf_u32(&device, wgpu::BufferUsages::STORAGE, &src);
    let dst_buf = buf_u32(&device, wgpu::BufferUsages::STORAGE, &dst);
    let w_buf = buf_u32(
        &device,
        wgpu::BufferUsages::STORAGE,
        bytemuck::cast_slice(&w),
    );
    let epol_buf = buf_u32(&device, wgpu::BufferUsages::STORAGE, &epol);
    let inj_idx_buf = device.create_buffer(&wgpu::BufferDescriptor {
        size: (INJ_CAP as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
        label: Some("inj_idx"),
    });
    let inj_val_buf = device.create_buffer(&wgpu::BufferDescriptor {
        size: (INJ_CAP as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
        label: Some("inj_val"),
    });
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
        label: Some("params"),
    });
    let rb_spikes = device.create_buffer(&wgpu::BufferDescriptor {
        size: ((n as u64) + 1) * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
        label: Some("rb_spikes"),
    });
    let rb_v = device.create_buffer(&wgpu::BufferDescriptor {
        size: (n as u64) * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
        label: Some("rb_v"),
    });

    // ---- pipelines ---------------------------------------------------------
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/lif_tick.wgsl").into()),
        label: Some("lif_tick"),
    });
    let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl0"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl1"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pl_all = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl-all"),
        bind_group_layouts: &[&bgl0, &bgl1],
        push_constant_ranges: &[],
    });
    let pl_state = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl-state"),
        bind_group_layouts: &[&bgl0],
        push_constant_ranges: &[],
    });
    let mk_pipe = |entry: &str, layout: &wgpu::PipelineLayout| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let pipe_inject = mk_pipe("inject", &pl_all);
    let pipe_scatter = mk_pipe("scatter", &pl_state);
    let pipe_integrate = mk_pipe("integrate", &pl_state);

    let bind0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind0"),
        layout: &bgl0,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: v_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ge_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: gi_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: dge_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: dgi_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: spiked_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: spikes_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: src_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: dst_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: epol_buf.as_entire_binding(),
            },
        ],
    });
    let bind1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind1"),
        layout: &bgl1,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inj_idx_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: inj_val_buf.as_entire_binding(),
            },
        ],
    });

    Ok(WgpuBackend {
        device_name: format!("{} ({:?})", info.name, info.backend),
        device,
        queue,
        bind0,
        bind1,
        pipe_inject,
        pipe_scatter,
        pipe_integrate,
        v_buf,
        spikes_buf,
        inj_idx_buf,
        inj_val_buf,
        rb_spikes,
        rb_v,
        uniform_buf,
        n,
        nnz,
        n_wg_int: n.div_ceil(WG),
        n_wg_sc: nnz.div_ceil(WG),
        pending_inj: Vec::new(),
    })
}

impl WgpuBackend {
    fn submit_tick(&mut self, mut u: Uniforms) -> Result<(), String> {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tick"),
            });
        if !self.pending_inj.is_empty() {
            // dedupe + sum per neuron CPU-side (plain RMW in the shader)
            let mut map: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
            for (i, v) in self.pending_inj.drain(..) {
                *map.entry(i).or_insert(0.0) += v;
            }
            let k = map.len().min(INJ_CAP as usize);
            u.inj_count = k as u32;
            let idx: Vec<u32> = map.keys().copied().take(k).collect();
            let val: Vec<f32> = map.values().copied().take(k).collect();
            self.queue
                .write_buffer(&self.inj_idx_buf, 0, bytemuck::cast_slice(&idx));
            self.queue
                .write_buffer(&self.inj_val_buf, 0, bytemuck::cast_slice(&val));
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("inject"),
                    ..Default::default()
                });
                pass.set_pipeline(&self.pipe_inject);
                pass.set_bind_group(0, &self.bind0, &[]);
                pass.set_bind_group(1, &self.bind1, &[]);
                pass.dispatch_workgroups((k as u32).div_ceil(WG), 1, 1);
            }
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scatter-integrate"),
                ..Default::default()
            });
            // dispatch x-dim is capped at 65535 — spread across y
            const MAX_WG: u32 = 65535;
            let sc_x = self.n_wg_sc.min(MAX_WG);
            let sc_y = self.n_wg_sc.div_ceil(MAX_WG);
            u.wg_x = sc_x;
            pass.set_pipeline(&self.pipe_scatter);
            pass.set_bind_group(0, &self.bind0, &[]);
            pass.dispatch_workgroups(sc_x, sc_y, 1);
            pass.set_pipeline(&self.pipe_integrate);
            pass.dispatch_workgroups(self.n_wg_int, 1, 1);
        }
        enc.copy_buffer_to_buffer(
            &self.spikes_buf,
            0,
            &self.rb_spikes,
            0,
            ((self.n as u64) + 1) * 4,
        );
        enc.copy_buffer_to_buffer(&self.v_buf, 0, &self.rb_v, 0, (self.n as u64) * 4);
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
        self.queue.submit(Some(enc.finish()));
        Ok(())
    }

    fn map_slice<'a>(
        &'a self,
        buf: &'a wgpu::Buffer,
        size: u64,
    ) -> Result<wgpu::BufferView<'a>, String> {
        let (tx, rx) = std::sync::mpsc::channel();
        let slice = buf.slice(..size);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| "gpu map channel closed".to_string())?
            .map_err(|e| format!("gpu map failed: {e}"))?;
        Ok(slice.get_mapped_range())
    }
}

impl WgpuBackend {
    fn state_writer(&mut self, v: &[f32]) -> Result<(), String> {
        if v.len() != self.n as usize {
            return Err(format!(
                "v length mismatch: got {}, device has {}",
                v.len(),
                self.n
            ));
        }
        self.queue
            .write_buffer(&self.v_buf, 0, bytemuck::cast_slice(v));
        Ok(())
    }
}

impl GpuBackend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }
    fn device(&self) -> String {
        self.device_name.clone()
    }
    fn inject(&mut self, neurons: &[u32], current: f32) {
        for &i in neurons {
            self.pending_inj.push((i, current));
        }
    }
    fn write_state(&mut self, v: &[f32]) -> Result<(), String> {
        self.state_writer(v)
    }
    fn tick(&mut self, p: &TickParams) -> Result<GpuTick, String> {
        let u = Uniforms {
            n: self.n,
            nnz: self.nnz,
            wg_x: 1,
            inj_count: 0, // submit_tick raises this when injections are queued
            decay_exc: p.decay_exc,
            decay_inh: p.decay_inh,
            a: p.a,
            v_rest: p.v_rest,
            input_gain: p.input_gain,
            v_thresh: p.v_thresh,
            v_reset: p.v_reset,
            w_scale: p.weight_scale,
            wq: WQ,
            inv_wq: 1.0 / WQ,
        };
        self.submit_tick(u)?;
        // spikes[0] must be zero before the next integrate's atomicAdd
        self.queue
            .write_buffer(&self.spikes_buf, 0, bytemuck::cast_slice(&[0u32]));

        let view = self.map_slice(&self.rb_spikes, ((self.n as u64) + 1) * 4)?;
        let count = u32::from_le_bytes(view[..4].try_into().unwrap()) as usize;
        let spikes: Vec<u32> = bytemuck::cast_slice(&view[4..4 + count * 4]).to_vec();
        drop(view);
        self.rb_spikes.unmap();
        Ok(GpuTick { spikes })
    }
    fn read_v(&mut self, out: &mut [f32]) -> Result<(), String> {
        let view = self.map_slice(&self.rb_v, (self.n as u64) * 4)?;
        let vals: &[f32] = bytemuck::cast_slice(&view);
        out.iter_mut().zip(vals).for_each(|(o, &v)| *o = v);
        drop(view);
        self.rb_v.unmap();
        Ok(())
    }
}
