//! `.flybin` — compiled substrate format (the "GGUF" of a connectome).
//!
//! v2 layout (little-endian):
//!   [0..8)    magic `FLYBIN\x01\x00`
//!   [8..12)   u32 header length
//!   [12..12+H) UTF-8 JSON header: {version, n_neurons, n_edges, string_tables}
//!   then:     u64  indptr[n+1]
//!             u32  indices[nnz]
//!             u8   weights[nnz]          (syn_count quantized per-row; exact
//!                                         for rows whose max count <= 255,
//!                                         which is 99.99% of the dataset)
//!             f32  row_scale[n]          (dequant: count = w * row_scale[row])
//!             u64  root_ids[n]
//!             u32  region[n]   ┐ indices into string tables
//!             u32  cell_type[n]│
//!             u32  nt_type[n]  ┘

use serde::{Deserialize, Serialize};

pub const MAGIC: [u8; 8] = *b"FLYBIN\x01\x00";

#[derive(Debug, Serialize, Deserialize)]
pub struct FlybinHeader {
    pub format_version: u32,
    pub n_neurons: u32,
    pub n_edges: u64,
    pub source: String,
    pub string_tables: StringTables,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct StringTables {
    pub regions: Vec<String>,
    pub cell_types: Vec<String>,
    pub nt_types: Vec<String>,
}

/// Weight storage: quantized u8 (v2, ~4x smaller) or exact f32 (v1).
#[derive(Debug)]
pub enum Weights {
    /// Exact syn_count per edge.
    F32(Vec<f32>),
    /// syn_count quantized per-row: count = u8_value * row_scale[row].
    /// Exact for rows whose max count <= 255 (99.99% of FlyWire).
    U8 { w: Vec<u8>, scale: Vec<f32> },
}

/// Zero-allocation outgoing-edges iterator.
pub enum OutgoingIter<'a> {
    F32(std::iter::Zip<std::slice::Iter<'a, u32>, std::slice::Iter<'a, f32>>),
    U8(
        std::iter::Zip<std::slice::Iter<'a, u32>, std::slice::Iter<'a, u8>>,
        f32,
    ),
}

impl Iterator for OutgoingIter<'_> {
    type Item = (u32, f32);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            OutgoingIter::F32(it) => it.next().map(|(&i, &v)| (i, v)),
            OutgoingIter::U8(it, s) => it.next().map(|(&i, &v)| (i, v as f32 * *s)),
        }
    }
}

#[derive(Debug)]
pub struct Substrate {
    pub header: FlybinHeader,
    pub indptr: Vec<u64>,
    pub indices: Vec<u32>,
    pub weights: Weights,
    pub root_ids: Vec<u64>,
    pub region: Vec<u32>,
    pub cell_type: Vec<u32>,
    pub nt_type: Vec<u32>,
}

impl Substrate {
    pub fn n_neurons(&self) -> usize {
        self.header.n_neurons as usize
    }

    pub fn edges(&self, pre: usize) -> std::ops::Range<usize> {
        self.indptr[pre] as usize..self.indptr[pre + 1] as usize
    }

    /// Post-synaptic partners and dequantized weights of `pre`.
    /// Zero-allocation: concrete enum instead of Box<dyn Iterator> so the
    /// scatter hot path never touches the heap.
    pub fn outgoing(&self, pre: usize) -> OutgoingIter<'_> {
        let range = self.edges(pre);
        let (start, end) = (range.start, range.end);
        match &self.weights {
            Weights::F32(w) => {
                OutgoingIter::F32(self.indices[start..end].iter().zip(w[start..end].iter()))
            }
            Weights::U8 { w, scale } => {
                let s = scale[pre];
                OutgoingIter::U8(self.indices[start..end].iter().zip(w[start..end].iter()), s)
            }
        }
    }
}

pub fn write_flybin(path: &std::path::Path, s: &Substrate) -> std::io::Result<()> {
    use std::io::Write;
    let header_json = serde_json::to_vec(&s.header).map_err(std::io::Error::other)?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    out.write_all(&MAGIC)?;
    out.write_all(&(header_json.len() as u32).to_le_bytes())?;
    out.write_all(&header_json)?;
    for v in &s.indptr {
        out.write_all(&v.to_le_bytes())?;
    }
    for v in &s.indices {
        out.write_all(&v.to_le_bytes())?;
    }
    match &s.weights {
        Weights::F32(w) => {
            for v in w {
                out.write_all(&v.to_le_bytes())?;
            }
        }
        Weights::U8 { w, scale } => {
            out.write_all(w)?;
            for v in scale {
                out.write_all(&v.to_le_bytes())?;
            }
        }
    }
    for v in &s.root_ids {
        out.write_all(&v.to_le_bytes())?;
    }
    for v in &s.region {
        out.write_all(&v.to_le_bytes())?;
    }
    for v in &s.cell_type {
        out.write_all(&v.to_le_bytes())?;
    }
    for v in &s.nt_type {
        out.write_all(&v.to_le_bytes())?;
    }
    out.flush()
}

pub fn read_flybin(path: &std::path::Path) -> std::io::Result<Substrate> {
    use std::io::Read;
    let mut buf = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 8];
    buf.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad magic: not a .flybin file",
        ));
    }
    let mut len_b = [0u8; 4];
    buf.read_exact(&mut len_b)?;
    let hlen = u32::from_le_bytes(len_b) as usize;
    let mut header_json = vec![0u8; hlen];
    buf.read_exact(&mut header_json)?;
    let header: FlybinHeader = serde_json::from_slice(&header_json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let n = header.n_neurons as usize;
    let nnz = header.n_edges as usize;

    let mut read_vec = |count: usize| -> std::io::Result<Vec<u8>> {
        let mut v = vec![0u8; count];
        buf.read_exact(&mut v)?;
        Ok(v)
    };

    let indptr = cast_u64(&read_vec((n + 1) * 8)?)?;
    let indices = cast_u32(&read_vec(nnz * 4)?)?;
    let weights = match header.format_version {
        1 => Weights::F32(cast_f32(&read_vec(nnz * 4)?)?),
        2 => {
            let w = read_vec(nnz)?;
            let scale = cast_f32(&read_vec(n * 4)?)?;
            Weights::U8 { w, scale }
        }
        v => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported flybin format_version {v} (recompile the substrate)"),
            ))
        }
    };
    let root_ids = cast_u64(&read_vec(n * 8)?)?;
    let region = cast_u32(&read_vec(n * 4)?)?;
    let cell_type = cast_u32(&read_vec(n * 4)?)?;
    let nt_type = cast_u32(&read_vec(n * 4)?)?;

    Ok(Substrate {
        header,
        indptr,
        indices,
        weights,
        root_ids,
        region,
        cell_type,
        nt_type,
    })
}

fn cast_u64(v: &[u8]) -> std::io::Result<Vec<u64>> {
    bytemuck_try(v)
}
fn cast_u32(v: &[u8]) -> std::io::Result<Vec<u32>> {
    bytemuck_try(v)
}
fn cast_f32(v: &[u8]) -> std::io::Result<Vec<f32>> {
    bytemuck_try(v)
}

#[allow(clippy::chunks_exact_to_as_chunks)]
fn bytemuck_try<T: FromLeBytesSafe + Copy>(v: &[u8]) -> std::io::Result<Vec<T>> {
    let chunks = v.chunks_exact(std::mem::size_of::<T>());
    if !chunks.remainder().is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "aligned read failed: truncated data",
        ));
    }
    Ok(chunks.map(|c| T::from_le_bytes_safe(c)).collect())
}

trait FromLeBytesSafe: Copy {
    fn from_le_bytes_safe(b: &[u8]) -> Self;
}
macro_rules! impl_le {
    ($t:ty) => {
        impl FromLeBytesSafe for $t {
            fn from_le_bytes_safe(b: &[u8]) -> Self {
                let mut a = [0u8; std::mem::size_of::<$t>()];
                a.copy_from_slice(b);
                <$t>::from_le_bytes(a)
            }
        }
    };
}
impl_le!(u64);
impl_le!(u32);
impl_le!(f32);
