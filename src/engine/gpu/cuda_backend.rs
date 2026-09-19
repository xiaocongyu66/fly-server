//! CUDA backend (probe stage). Device detection via dlopen'd libcuda so
//! NVIDIA machines list "cuda" in the Settings page; kernel compilation
//! (NVRTC) and launches land in a follow-up pass — selection reports that
//! clearly instead of pretending to work.

use super::GpuProbe;
use libloading::{Library, Symbol};

/// Device probe: dlopen libcuda + cuInit + device name. No context created.
pub fn probe() -> Option<GpuProbe> {
    unsafe {
        let lib = Library::new("libcuda.so.1").ok()?;
        let cu_init: Symbol<unsafe extern "C" fn(u32) -> i32> = lib.get(b"cuInit").ok()?;
        let cu_device_get: Symbol<unsafe extern "C" fn(*mut i32, i32) -> i32> =
            lib.get(b"cuDeviceGet").ok()?;
        let cu_get_name: Symbol<unsafe extern "C" fn(*mut u8, i32, i32) -> i32> =
            lib.get(b"cuDeviceGetName").ok()?;
        if cu_init(0) != 0 {
            return None;
        }
        let mut dev: i32 = 0;
        if cu_device_get(&mut dev, 0) != 0 {
            return None;
        }
        let mut name = [0u8; 128];
        if cu_get_name(name.as_mut_ptr(), 127, dev) != 0 {
            return None;
        }
        let end = name.iter().position(|&c| c == 0).unwrap_or(127);
        let dev_name = String::from_utf8_lossy(&name[..end]).into_owned();
        Some(GpuProbe {
            backend: "cuda",
            device: dev_name,
        })
    }
}
