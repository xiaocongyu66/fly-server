//! Substrate: connectome ingest + `.flybin` compilation.
//!
//! A substrate is the frozen "base model" of a fly brain: CSR connectivity
//! plus per-neuron metadata. Wiring is never trained; post-training happens
//! only through adapter slots (see `crate::slots`).

pub mod flybin;
pub mod flywire;

use std::collections::HashMap;
use std::path::Path;

pub use flybin::{FlybinHeader, StringTables, Substrate, Weights};

#[derive(Debug, Default, serde::Serialize)]
pub struct CompileReport {
    pub n_neurons: u32,
    pub n_edges_aggregated: u64,
    pub n_raw_rows: u64,
    pub n_rows_skipped_unknown_root: u64,
    pub total_synapses: u64,
    pub tables: (usize, usize, usize), // regions, cell_types, nt_types
}

/// Weight storage switch for substrate compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    /// v2: u8 weights + per-row scale. ~4x smaller; exact for rows with
    /// max syn_count <= 255 (99.99% of FlyWire edges).
    U8,
    /// v1: exact f32 weights.
    F32,
}

/// Compile FlyWire v783 Princeton CSV.gz dumps into a `.flybin` substrate.
pub fn compile_flywire(
    data_dir: &Path,
    out_path: &Path,
    quant: Quant,
) -> std::io::Result<CompileReport> {
    let conn_path = data_dir.join("connections.csv.gz");
    let neurons_path = data_dir.join("neurons.csv.gz");
    for p in [&conn_path, &neurons_path] {
        if !p.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("missing {} (expected FlyWire v783 Princeton dump renamed to connections.csv.gz / neurons.csv.gz)", p.display()),
            ));
        }
    }

    // 1. Neurons: assign dense indices in file order.
    let mut root_to_idx: HashMap<u64, u32> = HashMap::with_capacity(140_000);
    let mut metas: Vec<flywire::NeuronMeta> = Vec::with_capacity(140_000);
    flywire::stream_neurons(&neurons_path, |m| {
        root_to_idx.insert(m.root_id, metas.len() as u32);
        metas.push(m);
        Ok(())
    })?;
    let n = metas.len();
    if n == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "no neurons parsed",
        ));
    }

    // 2. Connections: map roots to indices, keep (pre, post, syn).
    let mut edges: Vec<(u32, u32, u32)> = Vec::with_capacity(4_000_000);
    let mut raw_rows = 0u64;
    let mut skipped = 0u64;
    flywire::stream_connections(&conn_path, |c| {
        raw_rows += 1;
        match (root_to_idx.get(&c.pre), root_to_idx.get(&c.post)) {
            (Some(&p), Some(&q)) => edges.push((p, q, c.syn)),
            _ => skipped += 1,
        }
        Ok(())
    })?;

    // 3. Aggregate duplicate (pre, post) pairs across neuropils.
    edges.sort_unstable();
    let mut agg: Vec<(u32, u32, u32)> = Vec::with_capacity(edges.len());
    for &(p, q, s) in &edges {
        match agg.last_mut() {
            Some((lp, lq, ls)) if *lp == p && *lq == q => *ls = ls.saturating_add(s),
            _ => agg.push((p, q, s)),
        }
    }
    drop(edges);
    let n_edges = agg.len() as u64;
    let total_syn: u64 = agg.iter().map(|&(_, _, s)| s as u64).sum();

    // 4. CSR construction.
    let mut indptr = vec![0u64; n + 1];
    for &(p, _, _) in &agg {
        indptr[p as usize + 1] += 1;
    }
    for i in 1..=n {
        indptr[i] += indptr[i - 1];
    }
    let mut indices = vec![0u32; agg.len()];
    let mut cursor = indptr.clone();
    for (p, q, _s) in &agg {
        let c = cursor[*p as usize] as usize;
        indices[c] = *q;
        cursor[*p as usize] += 1;
    }
    let weights = match quant {
        Quant::F32 => {
            let mut w = vec![0f32; agg.len()];
            let mut c2 = indptr.clone();
            for (p, _, s) in &agg {
                let c = c2[*p as usize] as usize;
                w[c] = *s as f32;
                c2[*p as usize] += 1;
            }
            Weights::F32(w)
        }
        Quant::U8 => {
            // per-row scale: exact when row max <= 255, else max/255
            let mut row_max = vec![0u32; n];
            for &(p, _, s) in &agg {
                if s > row_max[p as usize] {
                    row_max[p as usize] = s;
                }
            }
            let scale: Vec<f32> = (0..n)
                .map(|i| {
                    if row_max[i] <= 255 {
                        1.0
                    } else {
                        row_max[i] as f32 / 255.0
                    }
                })
                .collect();
            let mut w = vec![0u8; agg.len()];
            let mut c2 = indptr.clone();
            for (p, _, s) in &agg {
                let c = c2[*p as usize] as usize;
                w[c] = ((*s as f32 / scale[*p as usize]).round() as u32).min(255) as u8;
                c2[*p as usize] += 1;
            }
            Weights::U8 { w, scale }
        }
    };

    // 5. String tables.
    fn string_table(
        metas: &[flywire::NeuronMeta],
        pick: fn(&flywire::NeuronMeta) -> &str,
    ) -> (Vec<String>, Vec<u32>) {
        let mut map: HashMap<String, u32> = HashMap::new();
        let mut table: Vec<String> = Vec::new();
        let mut ids = Vec::with_capacity(metas.len());
        for m in metas {
            let s = pick(m);
            let s = if s.is_empty() { "" } else { s };
            let next_id = table.len() as u32;
            let id = *map.entry(s.to_string()).or_insert_with(|| {
                table.push(s.to_string());
                next_id
            });
            ids.push(id);
        }
        (table, ids)
    }
    let (regions, region) = string_table(&metas, |m| &m.region);
    let (cell_types, cell_type) = string_table(&metas, |m| &m.cell_type);
    let (nt_types, nt_type) = string_table(&metas, |m| &m.nt_type);
    let table_sizes = (regions.len(), cell_types.len(), nt_types.len());

    let substrate = Substrate {
        header: FlybinHeader {
            format_version: match quant {
                Quant::F32 => 1,
                Quant::U8 => 2,
            },
            n_neurons: n as u32,
            n_edges,
            source: format!("FlyWire v783 Princeton dump ({})", data_dir.display()),
            string_tables: StringTables {
                regions,
                cell_types,
                nt_types,
            },
        },
        indptr,
        indices,
        weights,
        root_ids: metas.iter().map(|m| m.root_id).collect(),
        region,
        cell_type,
        nt_type,
    };
    flybin::write_flybin(out_path, &substrate)?;

    Ok(CompileReport {
        n_neurons: n as u32,
        n_edges_aggregated: n_edges,
        n_raw_rows: raw_rows,
        n_rows_skipped_unknown_root: skipped,
        total_synapses: total_syn,
        tables: table_sizes,
    })
}

/// Load a compiled substrate from disk.
pub fn load(path: &Path) -> std::io::Result<Substrate> {
    flybin::read_flybin(path)
}
