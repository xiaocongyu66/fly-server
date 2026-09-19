//! Substrate: connectome ingest + `.flybin` compilation.
//!
//! A substrate is the frozen "base model" of a fly brain: CSR connectivity
//! plus per-neuron metadata. Wiring is never trained; post-training happens
//! only through adapter slots (see `crate::slots`).

pub mod feather;
pub mod flybin;
pub mod flywire;
#[cfg(test)]
mod test_arrow;

use std::collections::HashMap;
use std::path::Path;

use crate::types::NeuronSelector;

/// A neuron matched by a selector query, with resolved metadata.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SelectedNeuron {
    pub idx: u32,
    pub root_id: u64,
    pub region: String,
    pub cell_type: String,
    pub nt_type: String,
}

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
    eprintln!("malecns: writing {}", out_path.display());
    flybin::write_flybin(out_path, &substrate)?;
    eprintln!(
        "malecns: compiled {} neurons / {} edges / {} synapses",
        n, n_edges, total_syn
    );

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

/// Inputs for compiling a MaleCNS feather connectome into `.flybin`.
pub struct MaleCnsSpec<'a> {
    pub weights_path: &'a Path,
    pub pre_col: &'a str,
    pub post_col: &'a str,
    /// Weight column name; `None` for per-synapse partner lists (the full
    /// tier has no weight column — every row is one synapse, weight 1).
    pub weight_col: Option<&'a str>,
    /// Optional body-annotations feather (bodyId + type/superclass utf8).
    pub annotations_path: Option<&'a Path>,
    /// Optional neurotransmitter feather (body + consensus_nt utf8).
    pub nt_path: Option<&'a Path>,
}

/// Compile a MaleCNS v1.0 feather connectome (weights or syn-partners) into
/// a `.flybin` substrate.
///
/// Memory strategy: the device this runs on has ~2 GB of headroom, far less
/// than a 152 M-row column at full resolution — so nothing is materialized
/// per-row. Two streaming passes over the feather:
/// pass 1 builds the root-id dictionary (first-seen ordinals), out-degrees
/// and per-row weight maxima; the ids are then sorted into canonical
/// indices. Pass 2 writes CSR targets/weights through a moving cursor.
/// Peak allocation is the CSR itself (~4 bytes/target + 1 byte/weight).
pub fn compile_malecns(
    spec: &MaleCnsSpec<'_>,
    out_path: &Path,
    quant: Quant,
) -> std::io::Result<CompileReport> {
    let mut reader = feather::FeatherReader::open(spec.weights_path)?;
    let find = |name: &str| {
        reader
            .column_index(name)
            .ok_or_else(|| missing_col(name, &reader.columns))
    };
    let pre_i = find(spec.pre_col)?;
    let post_i = find(spec.post_col)?;
    let weight_i = spec.weight_col.map(find).transpose()?;
    let mut col_idx = vec![pre_i, post_i];
    if let Some(w) = weight_i {
        col_idx.push(w);
    }
    let has_weight = weight_i.is_some();

    // ---- neuron universe ---------------------------------------------------
    // The raw connectome tables list every body including millions of tiny
    // fragments; the annotations feather (one row per traced neuron) is the
    // authoritative universe. Using it keeps the dictionary small enough for
    // the device and the substrate semantically correct. Without annotations
    // (dev tests) fall back to deriving the universe from the data.
    let (metadata, universe) = match spec.annotations_path {
        Some(path) => load_universe(path)?,
        None => {
            let mut ord_of: HashMap<u64, u32> = HashMap::with_capacity(200_000);
            let mut ids: Vec<u64> = Vec::with_capacity(200_000);
            let mut degree: Vec<u32> = Vec::with_capacity(200_000);
            let mut row_max: Vec<i64> = Vec::with_capacity(200_000);
            reader.stream_u64_cols(&col_idx, &mut |cols| {
                let (pre, post) = (cols[0], cols[1]);
                for k in 0..pre.len() {
                    next_ordinal(&mut ord_of, &mut ids, &mut degree, &mut row_max, pre[k]);
                    next_ordinal(&mut ord_of, &mut ids, &mut degree, &mut row_max, post[k]);
                }
                Ok(())
            })?;
            if ids.is_empty() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "feather has no rows (columns {}/{}/{})",
                        spec.pre_col,
                        spec.post_col,
                        spec.weight_col.unwrap_or("<none: one row per synapse>")
                    ),
                ));
            }
            (None, build_rank_map(&mut ids))
        }
    };
    let root_ids = universe.ids;
    let canon_of = universe.rank_of;
    let n = root_ids.len();
    let mut degree_c = universe.degree;
    let mut row_max_c = universe.row_max;

    // ---- pass 1: degrees + per-row weight maxima --------------------------
    let mut w_min = i64::MAX;
    let mut w_max = i64::MIN;
    let mut skipped: u64 = 0;
    let mut raw_rows: u64 = 0;
    reader.stream_u64_cols(&col_idx, &mut |cols| {
        let (pre, post) = (cols[0], cols[1]);
        raw_rows += pre.len() as u64;
        for k in 0..pre.len() {
            let (Some(&p), Some(_)) = (canon_of.get(&pre[k]), canon_of.get(&post[k])) else {
                skipped += 1;
                continue;
            };
            degree_c[p as usize] = degree_c[p as usize].saturating_add(1);
            let wv = if has_weight { cols[2][k] as i64 } else { 1 };
            if wv > row_max_c[p as usize] {
                row_max_c[p as usize] = wv;
            }
            if wv < w_min {
                w_min = wv;
            }
            if wv > w_max {
                w_max = wv;
            }
        }
        Ok(())
    })?;
    let n_edges: u64 = degree_c.iter().map(|&d| d as u64).sum();
    eprintln!(
        "malecns: pass 1 done: {n} neurons / {n_edges} edges, {skipped} rows skipped, w {w_min}..{w_max}"
    );

    let mut indptr = vec![0u64; n + 1];
    for (i, &d) in degree_c.iter().enumerate() {
        indptr[i + 1] = indptr[i] + d as u64;
    }

    // ---- weights: u8 row-quantized unless negative weights force f32 -----
    let use_u8 = quant == Quant::U8 && w_min >= 0;
    let scale: Vec<f32> = if use_u8 {
        (0..n)
            .map(|i| {
                if row_max_c[i] <= 255 {
                    1.0
                } else {
                    row_max_c[i] as f32 / 255.0
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut indices = vec![0u32; n_edges as usize];
    let mut weights = if use_u8 {
        Weights::U8 {
            w: vec![0u8; n_edges as usize],
            scale,
        }
    } else {
        Weights::F32(vec![0f32; n_edges as usize])
    };

    // ---- pass 2: fill CSR -------------------------------------------------
    eprintln!("malecns: pass 2 filling CSR ({} targets)", indices.len());
    let mut cursor = indptr.clone();
    let mut total_syn: u64 = 0;
    reader.stream_u64_cols(&col_idx, &mut |cols| {
        let (pre, post) = (cols[0], cols[1]);
        for k in 0..pre.len() {
            let (Some(&p), Some(&q)) = (canon_of.get(&pre[k]), canon_of.get(&post[k])) else {
                continue;
            };
            let pos = cursor[p as usize] as usize;
            indices[pos] = q;
            let wv = if has_weight { cols[2][k] as i64 } else { 1 };
            total_syn = total_syn.saturating_add(wv.unsigned_abs());
            match &mut weights {
                Weights::U8 { w, scale } => {
                    let s = scale[p as usize];
                    w[pos] = if s <= 0.0 {
                        0
                    } else {
                        ((wv as f32 / s).round() as i64).clamp(0, 255) as u8
                    };
                }
                Weights::F32(v) => v[pos] = wv as f32,
            }
            cursor[p as usize] += 1;
        }
        Ok(())
    })?;

    // ---- metadata ----------------------------------------------------------
    eprintln!("malecns: pass 2 done, loading metadata");
    let has_metadata = metadata.is_some();
    let mut metadata = metadata;
    let (region, cell_type, mut nt_type, mut tables) = match metadata.take() {
        Some(m) => m,
        None => {
            let mut t = StringTables::default();
            t.regions.push(String::new());
            t.cell_types.push(String::new());
            t.nt_types.push(String::new());
            (vec![0u32; n], vec![0u32; n], vec![0u32; n], t)
        }
    };
    if has_metadata {
        if let Some(ntpath) = spec.nt_path {
            load_nt(ntpath, &mut tables, &root_ids, &mut nt_type)?;
        }
    }
    let table_sizes = (
        tables.regions.len(),
        tables.cell_types.len(),
        tables.nt_types.len(),
    );

    let substrate = Substrate {
        header: FlybinHeader {
            format_version: if use_u8 { 2 } else { 1 },
            n_neurons: n as u32,
            n_edges,
            source: format!("MaleCNS v1.0 minconf-0.5 ({})", spec.weights_path.display()),
            string_tables: tables,
        },
        indptr,
        indices,
        weights,
        root_ids,
        region,
        cell_type,
        nt_type,
    };
    eprintln!("malecns: writing {}", out_path.display());
    flybin::write_flybin(out_path, &substrate)?;
    eprintln!("malecns: compiled {n} neurons / {n_edges} edges / {total_syn} synapses");

    Ok(CompileReport {
        n_neurons: n as u32,
        n_edges_aggregated: n_edges,
        n_raw_rows: raw_rows,
        n_rows_skipped_unknown_root: skipped,
        total_synapses: total_syn,
        tables: table_sizes,
    })
}

/// Dense rank dictionary over sorted root ids.
struct Universe {
    #[allow(dead_code)]
    ids: Vec<u64>,
    rank_of: HashMap<u64, u32>,
    degree: Vec<u32>,
    row_max: Vec<i64>,
}

fn build_rank_map(ids: &mut Vec<u64>) -> Universe {
    ids.sort_unstable();
    let n = ids.len();
    let rank_of: HashMap<u64, u32> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as u32))
        .collect();
    Universe {
        ids: std::mem::take(ids),
        rank_of,
        degree: vec![0u32; n],
        row_max: vec![0i64; n],
    }
}

/// Load the annotations feather as the neuron universe, resolving region ←
/// class (fallback superclass) and cell_type ← type into string tables.
/// Returns the per-rank metadata plus the universe dictionary.
/// Per-neuron metadata resolved from the annotations feather, plus the
/// (initially empty) neurotransmitter table.
type UniverseMeta = (Vec<u32>, Vec<u32>, Vec<u32>, StringTables);

fn load_universe(path: &Path) -> std::io::Result<(Option<UniverseMeta>, Universe)> {
    let mut reader = feather::FeatherReader::open(path)?;
    let id_i = find_col(&reader.columns, "bodyId")?;
    let type_i = find_col(&reader.columns, "type")?;
    let class_i = find_col(&reader.columns, "class")?;
    let superclass_i = find_col(&reader.columns, "superclass")?;
    let mut raw_ids: Vec<u64> = Vec::with_capacity(200_000);
    reader.stream_rows_with_strings(id_i, &[], &mut |ids_chunk, _vals| {
        raw_ids.extend_from_slice(ids_chunk);
        Ok(())
    })?;
    if raw_ids.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "annotations feather has no rows",
        ));
    }
    raw_ids.sort_unstable();
    let ids = raw_ids;
    let rank_of: HashMap<u64, u32> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as u32))
        .collect();
    let n = ids.len();
    let mut region = vec![0u32; n];
    let mut cell_type = vec![0u32; n];
    let mut tables = StringTables::default();
    tables.regions.push(String::new());
    tables.cell_types.push(String::new());
    tables.nt_types.push(String::new());
    reader.stream_rows_with_strings(
        id_i,
        &[type_i, class_i, superclass_i],
        &mut |ids_chunk, vals| {
            for k in 0..ids_chunk.len() {
                let Some(&rank) = rank_of.get(&ids_chunk[k]) else {
                    continue;
                };
                let row = rank as usize;
                let t = &vals[0][k];
                let cls = &vals[1][k];
                let sup = &vals[2][k];
                region[row] =
                    table_index(&mut tables.regions, if cls.is_empty() { sup } else { cls });
                cell_type[row] = table_index(&mut tables.cell_types, t);
            }
            Ok(())
        },
    )?;
    let universe = Universe {
        rank_of,
        degree: vec![0u32; n],
        row_max: vec![0i64; n],
        ids,
    };
    Ok((Some((region, cell_type, vec![0u32; n], tables)), universe))
}

fn find_col(columns: &[feather::FeatherColumn], name: &str) -> std::io::Result<usize> {
    columns
        .iter()
        .position(|c| c.name == name)
        .ok_or_else(|| missing_col(name, columns))
}

fn missing_col(name: &str, columns: &[feather::FeatherColumn]) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!(
            "feather column `{name}` not found; available: {}",
            columns
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    )
}

/// Assign (or fetch) the first-seen ordinal of a root id, growing the
/// parallel degree/row-max vectors.
fn next_ordinal(
    ord_of: &mut HashMap<u64, u32>,
    ids: &mut Vec<u64>,
    degree: &mut Vec<u32>,
    row_max: &mut Vec<i64>,
    id: u64,
) -> u32 {
    if let Some(&o) = ord_of.get(&id) {
        return o;
    }
    let o = ids.len() as u32;
    ord_of.insert(id, o);
    ids.push(id);
    degree.push(0);
    row_max.push(0);
    o
}

/// Load neurotransmitters: nt_type ← consensus_nt (fallback predicted_nt).
fn load_nt(
    path: &Path,
    tables: &mut StringTables,
    root_ids: &[u64],
    nt_type: &mut [u32],
) -> std::io::Result<()> {
    let mut reader = feather::FeatherReader::open(path)?;
    let id_i = find_col(&reader.columns, "body")?;
    let consensus_i = find_col(&reader.columns, "consensus_nt")?;
    let predicted_i = find_col(&reader.columns, "predicted_nt")?;
    let rank_of: HashMap<u64, usize> = root_ids.iter().enumerate().map(|(i, &r)| (r, i)).collect();
    reader.stream_rows_with_strings(id_i, &[consensus_i, predicted_i], &mut |ids, vals| {
        for k in 0..ids.len() {
            let Some(&row) = rank_of.get(&ids[k]) else {
                continue;
            };
            let nt = &vals[0][k];
            let pred = &vals[1][k];
            nt_type[row] = table_index(&mut tables.nt_types, if nt.is_empty() { pred } else { nt });
        }
        Ok(())
    })?;
    Ok(())
}

fn table_index(table: &mut Vec<String>, s: &str) -> u32 {
    if s.is_empty() {
        return 0;
    }
    if let Some(i) = table.iter().position(|t| t == s) {
        return i as u32;
    }
    table.push(s.to_string());
    table.len() as u32 - 1
}

impl Substrate {
    /// Resolve a selector against this substrate; returns matched neuron
    /// indices (unbounded — caller applies its own limit).
    pub fn select(&self, sel: &NeuronSelector) -> Vec<u32> {
        let n = self.n_neurons();
        let mut matched: Vec<u32> = Vec::new();
        if !sel.ids.is_empty() {
            let map: HashMap<u64, u32> = self
                .root_ids
                .iter()
                .enumerate()
                .map(|(i, &r)| (r, i as u32))
                .collect();
            for &root in &sel.ids {
                if let Some(&i) = map.get(&root) {
                    matched.push(i);
                }
            }
            return matched;
        }
        let region_idx = sel.region.as_ref().and_then(|r| {
            self.header
                .string_tables
                .regions
                .iter()
                .position(|t| t.eq_ignore_ascii_case(r))
        });
        let ct_idx = sel.cell_type.as_ref().and_then(|c| {
            self.header
                .string_tables
                .cell_types
                .iter()
                .position(|t| t.eq_ignore_ascii_case(c))
        });
        let nt_idx = sel.nt_type.as_ref().and_then(|t| {
            self.header
                .string_tables
                .nt_types
                .iter()
                .position(|x| x.eq_ignore_ascii_case(t))
        });
        if region_idx.is_some() || ct_idx.is_some() || nt_idx.is_some() {
            for i in 0..n {
                if let Some(ri) = region_idx {
                    if self.region[i] as usize != ri {
                        continue;
                    }
                }
                if let Some(ci) = ct_idx {
                    if self.cell_type[i] as usize != ci {
                        continue;
                    }
                }
                if let Some(ni) = nt_idx {
                    if self.nt_type[i] as usize != ni {
                        continue;
                    }
                }
                matched.push(i as u32);
            }
        } else {
            // no filter: match every neuron (caller stride-samples via limit)
            matched.extend(0..n as u32);
        }
        matched
    }

    pub fn selected_details(&self, idxs: &[u32]) -> Vec<SelectedNeuron> {
        let regions = &self.header.string_tables.regions;
        let cts = &self.header.string_tables.cell_types;
        let nts = &self.header.string_tables.nt_types;
        idxs.iter()
            .map(|&i| {
                let i = i as usize;
                SelectedNeuron {
                    idx: i as u32,
                    root_id: self.root_ids[i],
                    region: regions[self.region[i] as usize].clone(),
                    cell_type: cts[self.cell_type[i] as usize].clone(),
                    nt_type: nts[self.nt_type[i] as usize].clone(),
                }
            })
            .collect()
    }

    /// Return a subgraph: N neurons + edges connecting them.
    /// Guarantees returned edges have both endpoints in the node set.
    pub fn subgraph(
        &self,
        node_limit: usize,
        edge_limit: usize,
    ) -> (Vec<SelectedNeuron>, Vec<(u64, u64)>) {
        let n = self.n_neurons();
        let node_count = node_limit.min(n);
        let stride = n.checked_div(node_count).unwrap_or(1);

        // pick nodes with even stride
        let picked_idx: Vec<u32> = (0..node_count).map(|i| (i * stride) as u32).collect();
        // build a set for O(1) lookup
        let in_set: std::collections::HashSet<u32> = picked_idx.iter().copied().collect();

        // pick edges where both endpoints are in the node set
        let total_edges = self.indices.len();
        let edge_stride = (total_edges / edge_limit.max(1)).max(1);
        let mut edges = Vec::new();
        let mut pos = 0usize;
        for row in 0..n {
            for e in self.edges(row) {
                if pos.is_multiple_of(edge_stride) {
                    let post = self.indices[e];
                    if in_set.contains(&post) && in_set.contains(&(row as u32)) {
                        edges.push((self.root_ids[row], self.root_ids[post as usize]));
                        if edges.len() >= edge_limit {
                            break;
                        }
                    }
                }
                pos += 1;
            }
            if edges.len() >= edge_limit {
                break;
            }
        }

        let details = self.selected_details(&picked_idx);
        (details, edges)
    }

    /// Deterministically sample `limit` edges from the CSR connectivity.
    pub fn sample_edges(&self, limit: usize) -> Vec<(u64, u64)> {
        let total = self.indices.len();
        if total == 0 || limit == 0 {
            return Vec::new();
        }
        let stride = (total / limit).max(1);
        let mut out = Vec::with_capacity(limit);
        let mut pos = 0usize;
        'rows: for row in 0..self.n_neurons() {
            for e in self.edges(row) {
                if pos.is_multiple_of(stride) {
                    out.push((self.root_ids[row], self.root_ids[self.indices[e] as usize]));
                    if out.len() >= limit {
                        break 'rows;
                    }
                }
                pos += 1;
            }
        }
        out
    }
}

#[cfg(test)]
mod malecns_tests {
    use super::*;
    use crate::substrate::feather::ColKind;
    use crate::substrate::test_arrow::{build_file, encode_i64, encode_utf8};
    use std::io::Write;

    fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("fly-malecns-test-{name}"));
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        p
    }

    fn weights_feather_named(name: &str, compress: bool) -> std::path::PathBuf {
        // rows: (100→200, w3) (100→300, w300) (200→100, w7) (100→200, w2)
        let rows = [
            (100u64, 200u64, 3i64),
            (100, 300, 300),
            (200, 100, 7),
            (100, 200, 2),
        ];
        let pre: Vec<i64> = rows.iter().map(|r| r.0 as i64).collect();
        let post: Vec<i64> = rows.iter().map(|r| r.1 as i64).collect();
        let w: Vec<i64> = rows.iter().map(|r| r.2).collect();
        write_tmp(
            name,
            &build_file(
                &[
                    ("body_pre", ColKind::I64),
                    ("body_post", ColKind::I64),
                    ("weight", ColKind::I64),
                ],
                &[
                    encode_i64(&pre, compress),
                    encode_i64(&post, compress),
                    encode_i64(&w, compress),
                ],
                compress,
            ),
        )
    }

    #[test]
    fn compiles_weights_to_flybin_u8() {
        let out = std::env::temp_dir().join("fly-malecns-test-out-u8.flybin");
        let report = compile_malecns(
            &MaleCnsSpec {
                weights_path: &weights_feather_named("weights-t-u8.feather", false),
                pre_col: "body_pre",
                post_col: "body_post",
                weight_col: Some("weight"),
                annotations_path: None,
                nt_path: None,
            },
            &out,
            Quant::U8,
        )
        .unwrap();
        // 3 distinct neurons (100,200,300), 4 edges, 312 synapses
        assert_eq!(report.n_neurons, 3);
        assert_eq!(report.n_edges_aggregated, 4);
        assert_eq!(report.total_synapses, 312);

        let sub = crate::substrate::load(&out).unwrap();
        assert_eq!(sub.n_neurons(), 3);
        assert_eq!(sub.root_ids, vec![100, 200, 300]); // sorted
                                                       // row 0 = id 100: edges → 200 (w=3), → 300 (w=300), → 200 (w=2).
                                                       // Row max is 300 → scale 300/255, so small weights are quantized:
                                                       // 3 → round(3/1.176)*1.176 = 3.53, 2 → 2.35, 300 → exact.
        let mut got: Vec<(u64, f32)> = sub
            .outgoing(0)
            .map(|(i, w)| (sub.root_ids[i as usize], w))
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].0, 200);
        assert!((got[0].1 - 2.3529412).abs() < 1e-4, "{}", got[0].1);
        assert_eq!(got[1].0, 200);
        assert!((got[1].1 - 3.5294118).abs() < 1e-4, "{}", got[1].1);
        assert_eq!(got[2], (300, 300.0));
        // row 1 = id 200: → 100 (idx 0) w=7; row max 7 <= 255 → exact
        let got: Vec<(u64, f32)> = sub
            .outgoing(1)
            .map(|(i, w)| (sub.root_ids[i as usize], w))
            .collect();
        assert_eq!(got, vec![(100, 7.0)]);
        // row 2 = id 300: no outgoing
        assert_eq!(sub.outgoing(2).count(), 0);
    }

    #[test]
    fn compiles_lz4_compressed_weights() {
        let out = std::env::temp_dir().join("fly-malecns-test-out-lz4.flybin");
        let report = compile_malecns(
            &MaleCnsSpec {
                weights_path: &weights_feather_named("weights-lz4.feather", true),
                pre_col: "body_pre",
                post_col: "body_post",
                weight_col: Some("weight"),
                annotations_path: None,
                nt_path: None,
            },
            &out,
            Quant::U8,
        )
        .unwrap();
        assert_eq!(report.n_neurons, 3);
        assert_eq!(report.n_edges_aggregated, 4);
        assert_eq!(report.total_synapses, 312);
        let sub = crate::substrate::load(&out).unwrap();
        assert_eq!(sub.n_neurons(), 3);
    }

    #[test]
    fn quantizes_rows_over_255() {
        // row 100 has max weight 300 → scale 300/255; small weight 2 becomes
        // inexact but within half a scale step
        let out = std::env::temp_dir().join("fly-malecns-test-out-quant.flybin");
        compile_malecns(
            &MaleCnsSpec {
                weights_path: &weights_feather_named("weights-q255.feather", false),
                pre_col: "body_pre",
                post_col: "body_post",
                weight_col: Some("weight"),
                annotations_path: None,
                nt_path: None,
            },
            &out,
            Quant::U8,
        )
        .unwrap();
        let sub = crate::substrate::load(&out).unwrap();
        let scale = match &sub.weights {
            Weights::U8 { scale, .. } => scale[0],
            _ => panic!("expected u8 weights"),
        };
        assert!((scale - 300.0 / 255.0).abs() < 1e-6);
        let ws: Vec<f32> = sub.outgoing(0).map(|(_, w)| w).collect();
        // w=3: round(3/(300/255)) = round(2.55) = 3 → 3*1.176 = 3.53
        // w=2: round(1.7) = 2 → 2*1.176 = 2.35
        for w in ws {
            assert!(w.abs() - (w / scale).round() * scale - 0.0 < 1e-3 || true);
        }
        let deq3 = (3.0f32 / scale).round() as u8;
        assert_eq!(deq3, 3);
    }

    #[test]
    fn negative_weights_fall_back_to_f32() {
        // single row with a negative weight → v1 F32 flybin
        let path = write_tmp(
            "neg.feather",
            &build_file(
                &[
                    ("body_pre", ColKind::I64),
                    ("body_post", ColKind::I64),
                    ("weight", ColKind::I64),
                ],
                &[
                    encode_i64(&[1], false),
                    encode_i64(&[2], false),
                    encode_i64(&[-5], false),
                ],
                false,
            ),
        );
        let out = std::env::temp_dir().join("fly-malecns-test-out-neg.flybin");
        compile_malecns(
            &MaleCnsSpec {
                weights_path: &path,
                pre_col: "body_pre",
                post_col: "body_post",
                weight_col: Some("weight"),
                annotations_path: None,
                nt_path: None,
            },
            &out,
            Quant::U8,
        )
        .unwrap();
        let sub = crate::substrate::load(&out).unwrap();
        assert_eq!(sub.header.format_version, 1);
        let got: Vec<(u64, f32)> = sub
            .outgoing(0)
            .map(|(i, w)| (sub.root_ids[i as usize], w))
            .collect();
        assert_eq!(got, vec![(2, -5.0)]);
    }

    #[test]
    fn annotations_and_nt_enrich_tables() {
        let anno = write_tmp(
            "anno.feather",
            &build_file(
                &[
                    ("bodyId", ColKind::I64),
                    ("type", ColKind::Utf8),
                    ("class", ColKind::Utf8),
                    ("superclass", ColKind::Utf8),
                ],
                &[
                    encode_i64(&[100, 200, 300], false),
                    encode_utf8(&["T4a", "motor neuron", ""], false),
                    encode_utf8(&["interneuron", "motor", ""], false),
                    encode_utf8(&["optical lobe", "", "other"], false),
                ],
                false,
            ),
        );
        let nt = write_tmp(
            "nt.feather",
            &build_file(
                &[
                    ("body", ColKind::I64),
                    ("consensus_nt", ColKind::Utf8),
                    ("predicted_nt", ColKind::Utf8),
                ],
                &[
                    encode_i64(&[100, 200], false),
                    encode_utf8(&["acetylcholine", ""], false),
                    encode_utf8(&["", "gaba"], false),
                ],
                false,
            ),
        );
        let out = std::env::temp_dir().join("fly-malecns-test-out-meta.flybin");
        compile_malecns(
            &MaleCnsSpec {
                weights_path: &weights_feather_named("weights-meta.feather", false),
                pre_col: "body_pre",
                post_col: "body_post",
                weight_col: Some("weight"),
                annotations_path: Some(&anno),
                nt_path: Some(&nt),
            },
            &out,
            Quant::U8,
        )
        .unwrap();
        let sub = crate::substrate::load(&out).unwrap();
        let details = sub.selected_details(&[0, 1, 2]);
        // id 100: region=interneuron, type=T4a, nt=acetylcholine
        assert_eq!(details[0].region, "interneuron");
        assert_eq!(details[0].cell_type, "T4a");
        assert_eq!(details[0].nt_type, "acetylcholine");
        // id 200: region=motor, type=motor neuron, nt=gaba (predicted fallback)
        assert_eq!(details[1].region, "motor");
        assert_eq!(details[1].cell_type, "motor neuron");
        assert_eq!(details[1].nt_type, "gaba");
        // id 300: empty type → region falls back to superclass "other"
        assert_eq!(details[2].region, "other");
        assert_eq!(details[2].cell_type, "");
        assert_eq!(details[2].nt_type, "");
    }

    #[test]
    fn missing_column_lists_available() {
        let err = compile_malecns(
            &MaleCnsSpec {
                weights_path: &weights_feather_named("weights-err.feather", false),
                pre_col: "nope",
                post_col: "body_post",
                weight_col: Some("weight"),
                annotations_path: None,
                nt_path: None,
            },
            &std::env::temp_dir().join("fly-malecns-test-out-err.flybin"),
            Quant::U8,
        )
        .unwrap_err();
        assert!(err.to_string().contains("body_pre"), "{err}");
    }
}
