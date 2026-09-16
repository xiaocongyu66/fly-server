//! Post-training slots: the connectome wiring is frozen; readout, gains,
//! and sensory encoders are swappable adapters (the LoRA analog).

use crate::substrate::Substrate;
use crate::types::Action;

/// Readout adapter: spike activity -> action space.
pub trait Readout: Send {
    /// Neuron indices this adapter reads from.
    fn pool(&self, s: &Substrate) -> Vec<u32>;
    /// Actions for this tick.
    fn actions(&self, s: &Substrate, spiked: &[u32], v: &[f32], v_thresh: f32) -> Vec<Action>;
}

/// Built-in readout: VNC (ventral nerve cord) neurons are the motor pool.
pub struct DefaultVncReadout {
    pub max_actions: usize,
}

impl DefaultVncReadout {
    pub fn new() -> Self {
        Self { max_actions: 32 }
    }
}

impl Default for DefaultVncReadout {
    fn default() -> Self {
        Self::new()
    }
}

impl Readout for DefaultVncReadout {
    fn pool(&self, s: &Substrate) -> Vec<u32> {
        let regions = &s.header.string_tables.regions;
        (0..s.n_neurons())
            .filter(|&i| regions[s.region[i] as usize].to_lowercase().contains("vnc"))
            .map(|i| i as u32)
            .collect()
    }

    fn actions(&self, s: &Substrate, spiked: &[u32], v: &[f32], v_thresh: f32) -> Vec<Action> {
        let pool = self.pool(s);
        let mut actions: Vec<Action> = spiked
            .iter()
            .filter(|i| pool.contains(i))
            .map(|&i| Action {
                neuron_id: s.root_ids[i as usize],
                rate: (v[i as usize] / v_thresh).clamp(0.0, 1.0),
            })
            .collect();
        actions.sort_by(|a, b| {
            b.rate
                .partial_cmp(&a.rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        actions.truncate(self.max_actions);
        actions
    }
}

/// Synaptic gain modulation slot (static scalar in Phase 1).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GainModulation {
    #[serde(default = "one")]
    pub exc_scale: f32,
    #[serde(default = "one")]
    pub inh_scale: f32,
}
fn one() -> f32 {
    1.0
}
impl Default for GainModulation {
    fn default() -> Self {
        Self {
            exc_scale: 1.0,
            inh_scale: 1.0,
        }
    }
}

/// Factory that instantiates a readout adapter.
type ReadoutFactory = fn() -> Box<dyn Readout>;

/// Adapter registry: id -> readout instance.
pub struct AdapterRegistry {
    factories: Vec<(&'static str, ReadoutFactory)>,
}

impl AdapterRegistry {
    pub fn builtin() -> Self {
        Self {
            factories: vec![("builtin-vnc-readout", || {
                Box::new(DefaultVncReadout::new()) as Box<dyn Readout>
            })],
        }
    }

    pub fn create(&self, id: &str) -> Option<Box<dyn Readout>> {
        self.factories
            .iter()
            .find(|(k, _)| *k == id)
            .map(|(_, f)| f())
    }

    pub fn list(&self) -> Vec<String> {
        self.factories.iter().map(|(k, _)| k.to_string()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::{FlybinHeader, StringTables};

    fn mini() -> Substrate {
        Substrate {
            header: FlybinHeader {
                format_version: 1,
                n_neurons: 2,
                n_edges: 1,
                source: "test".into(),
                string_tables: StringTables {
                    regions: vec!["".into(), "brain_R".into(), "VNC_T1_R".into()],
                    cell_types: vec!["".into()],
                    nt_types: vec!["".into(), "GLUT".into()],
                },
            },
            indptr: vec![0, 1, 1],
            indices: vec![1],
            weights: crate::substrate::flybin::Weights::F32(vec![50.0]),
            root_ids: vec![100, 200],
            region: vec![1, 2],
            cell_type: vec![0, 0],
            nt_type: vec![1, 1],
        }
    }

    #[test]
    fn pool_filters_vnc() {
        let s = mini();
        let r = DefaultVncReadout::new();
        assert_eq!(r.pool(&s), vec![1]);
    }

    #[test]
    fn actions_sort_and_map() {
        let s = mini();
        let r = DefaultVncReadout::new();
        let v = [0.0, 30.0];
        let a = r.actions(&s, &[0, 1], &v, 15.0);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].neuron_id, 200);
        assert!((a[0].rate - 1.0).abs() < 1e-6);
    }

    #[test]
    fn registry_builtin() {
        let reg = AdapterRegistry::builtin();
        assert!(reg.create("builtin-vnc-readout").is_some());
        assert!(reg.create("nope").is_none());
        assert!(reg.list().contains(&"builtin-vnc-readout".to_string()));
    }
}
