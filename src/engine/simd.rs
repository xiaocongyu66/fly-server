//! Vectorized hot-loop kernels (P0-2): aarch64 NEON f32x4 with a scalar
//! fallback. Operation order is identical between paths (mul/add chains,
//! no fma) so results are bit-identical — replays stay deterministic
//! regardless of the SIMD switch.

// ---------- scalar reference (also the non-aarch64 path) ----------

pub fn decay(buf: &mut [f32], k: f32) {
    for x in buf.iter_mut() {
        *x *= k;
    }
}

/// v = v + (rest - v)*a + gain*(ge - gi), then threshold/reset.
/// Returns (spike indices, sum of v after reset). Order matches the
/// original scalar loop exactly.
pub fn integrate(
    v: &mut [f32],
    g_exc: &[f32],
    g_inh: &[f32],
    rest: f32,
    a: f32,
    gain: f32,
    thresh: f32,
    reset: f32,
) -> (Vec<u32>, f32) {
    let mut spikes = Vec::new();
    let mut sum = 0f32;
    for i in 0..v.len() {
        let term = (rest - v[i]) * a + gain * (g_exc[i] - g_inh[i]);
        v[i] += term;
        sum += v[i];
        if v[i] >= thresh {
            v[i] = reset;
            spikes.push(i as u32);
        }
    }
    (spikes, sum)
}

// ---------- aarch64 NEON ----------

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use std::arch::aarch64::*;

    /// NEON f32x4 decay. Bit-identical to scalar `decay` (same mul order).
    pub fn decay(buf: &mut [f32], k: f32) {
        let chunks = buf.len() / 4;
        unsafe {
            let kv = vdupq_n_f32(k);
            for c in 0..chunks {
                let p = buf.as_mut_ptr().add(c * 4);
                let x = vld1q_f32(p);
                vst1q_f32(p, vmulq_f32(x, kv));
            }
        }
        for x in &mut buf[chunks * 4..] {
            *x *= k;
        }
    }

    /// NEON f32x4 integrate. Bit-identical to scalar `integrate`:
    /// term = (rest - v)*a + gain*(ge-gi) evaluated as
    ///   t1 = sub(rest, v); t2 = mul(t1, a); t3 = sub(ge, gi);
    ///   t4 = mul(gain, t3); t5 = add(t2, t4); v = add(v, t5)
    /// with no fma, so rounding order matches scalar mul/add exactly.
    pub fn integrate(
        v: &mut [f32],
        g_exc: &[f32],
        g_inh: &[f32],
        rest: f32,
        a: f32,
        gain: f32,
        thresh: f32,
        reset: f32,
    ) -> (Vec<u32>, f32) {
        let n = v.len();
        let chunks = n / 4;
        let mut spikes = Vec::new();
        let mut sum = 0f32;
        unsafe {
            let rest_v = vdupq_n_f32(rest);
            let a_v = vdupq_n_f32(a);
            let gain_v = vdupq_n_f32(gain);
            let mut sum_v = vdupq_n_f32(0.0);
            for c in 0..chunks {
                let p = v.as_mut_ptr().add(c * 4);
                let vv = vld1q_f32(p);
                let ge = vld1q_f32(g_exc.as_ptr().add(c * 4));
                let gi = vld1q_f32(g_inh.as_ptr().add(c * 4));
                let t1 = vsubq_f32(rest_v, vv);
                let t2 = vmulq_f32(t1, a_v);
                let t3 = vsubq_f32(ge, gi);
                let t4 = vmulq_f32(gain_v, t3);
                let t5 = vaddq_f32(t2, t4);
                let nv = vaddq_f32(vv, t5);
                vst1q_f32(p, nv);
                sum_v = vaddq_f32(sum_v, nv);
                // threshold pass (4 lanes; cheap after the vector math)
                let base = (c * 4) as u32;
                for lane in 0..4 {
                    let i = c * 4 + lane;
                    if v[i] >= thresh {
                        v[i] = reset;
                        spikes.push(base + lane as u32);
                    }
                }
            }
            sum += vaddvq_f32(sum_v);
        }
        for i in chunks * 4..n {
            let term = (rest - v[i]) * a + gain * (g_exc[i] - g_inh[i]);
            v[i] += term;
            sum += v[i];
            if v[i] >= thresh {
                v[i] = reset;
                spikes.push(i as u32);
            }
        }
        (spikes, sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_integrate(
        v: &mut [f32],
        g_exc: &[f32],
        g_inh: &[f32],
        rest: f32,
        a: f32,
        gain: f32,
        thresh: f32,
        reset: f32,
    ) -> (Vec<u32>, f32) {
        let mut spikes = Vec::new();
        let mut sum = 0f32;
        for i in 0..v.len() {
            let term = (rest - v[i]) * a + gain * (g_exc[i] - g_inh[i]);
            v[i] += term;
            sum += v[i];
            if v[i] >= thresh {
                v[i] = reset;
                spikes.push(i as u32);
            }
        }
        (spikes, sum)
    }

    #[test]
    fn decay_matches_scalar() {
        let mut a: Vec<f32> = (0..103).map(|i| (i % 17) as f32 * 0.37).collect();
        let mut b = a.clone();
        decay(&mut a, 0.9048);
        for x in b.iter_mut() {
            *x *= 0.9048;
        }
        assert_eq!(a, b, "bit-identical decay");
    }

    #[test]
    fn integrate_matches_scalar() {
        let n = 129; // crosses the 4-lane tail
        let mut a: Vec<f32> = (0..n).map(|i| (i % 23) as f32 * 0.11).collect();
        let ge: Vec<f32> = (0..n).map(|i| ((i * 7) % 31) as f32 * 0.05).collect();
        let gi: Vec<f32> = (0..n).map(|i| ((i * 3) % 13) as f32 * 0.04).collect();
        let mut b = a.clone();

        let (sa, suma) = integrate(&mut a, &ge, &gi, 0.0, 0.975, 1.0, 15.0, 0.0);
        let (sb, sumb) = ref_integrate(&mut b, &ge, &gi, 0.0, 0.975, 1.0, 15.0, 0.0);
        assert_eq!(a, b, "bit-identical integrate");
        assert_eq!(sa, sb);
        assert_eq!(suma, sumb);
    }
}
