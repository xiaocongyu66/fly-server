//! Streaming parsers for the FlyWire v783 Princeton CSV.gz dumps.
//!
//! connections: `pre_root_id, post_root_id, neuropil, syn_count, nt_type`
//!   (multiple rows per (pre, post) pair across neuropils — caller aggregates)
//! neurons:     `Root ID, ..., Primary Cell Type, ..., Predicted NT type, ...`

use std::io::{BufRead, BufReader, Read};

#[derive(Debug, Clone)]
pub struct RawConn {
    pub pre: u64,
    pub post: u64,
    pub syn: u32,
}

#[derive(Debug, Clone, Default)]
pub struct NeuronMeta {
    pub root_id: u64,
    pub region: String,
    pub cell_type: String,
    pub nt_type: String,
}

fn open_gz(path: &std::path::Path) -> std::io::Result<Box<dyn BufRead>> {
    let f = std::fs::File::open(path)?;
    let gz = flate2::read::MultiGzDecoder::new(f);
    Ok(Box::new(BufReader::with_capacity(1 << 16, gz)))
}

fn split_csv_line(line: &str) -> Vec<String> {
    // FlyWire dump columns contain no quoted commas in the used columns,
    // but handle simple quotes defensively.
    let mut out: Vec<String> = Vec::with_capacity(24);
    let mut cur = String::new();
    let mut in_q = false;
    for ch in line.chars() {
        match ch {
            '"' => in_q = !in_q,
            ',' if !in_q => {
                out.push(std::mem::take(&mut cur));
            }
            c => cur.push(c),
        }
    }
    out.push(cur);
    out
}

fn header_index(header: &str, wanted: &[&str]) -> std::io::Result<Vec<usize>> {
    let cols: Vec<String> = split_csv_line(header)
        .into_iter()
        .map(|s| s.trim().to_lowercase())
        .collect();
    let mut idx = Vec::with_capacity(wanted.len());
    for w in wanted {
        let wl = w.to_lowercase();
        idx.push(cols.iter().position(|c| c == &wl).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("column `{w}` not found in header: {header}"),
            )
        })?);
    }
    Ok(idx)
}

/// Stream connections CSV.gz, yielding raw (pre, post, syn) rows.
pub fn stream_connections(
    path: &std::path::Path,
    mut f: impl FnMut(RawConn) -> std::io::Result<()>,
) -> std::io::Result<u64> {
    let r = open_gz(path)?;
    let mut lines = r.lines();
    let header = lines.next().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "empty connections file")
    })??;
    let idx = header_index(&header, &["pre_root_id", "post_root_id", "syn_count"])?;
    let (i_pre, i_post, i_syn) = (idx[0], idx[1], idx[2]);
    let mut count = 0u64;
    for line in lines {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let cols = split_csv_line(&line);
        let pre: u64 = cols[i_pre].trim().parse().map_err(|e| bad_row(&line, e))?;
        let post: u64 = cols[i_post].trim().parse().map_err(|e| bad_row(&line, e))?;
        let syn: u32 = cols[i_syn].trim().parse().unwrap_or(0);
        if syn == 0 {
            continue;
        }
        f(RawConn { pre, post, syn })?;
        count += 1;
    }
    Ok(count)
}

/// Stream neurons CSV.gz, yielding per-neuron metadata.
pub fn stream_neurons(
    path: &std::path::Path,
    mut f: impl FnMut(NeuronMeta) -> std::io::Result<()>,
) -> std::io::Result<u64> {
    let r = open_gz(path)?;
    let mut lines = r.lines();
    let header = lines.next().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "empty neurons file")
    })??;
    let idx = header_index(
        &header,
        &[
            "Root ID",
            "Top in/out region",
            "Primary Cell Type",
            "Predicted NT type",
        ],
    )?;
    let (i_id, i_region, i_ct, i_nt) = (idx[0], idx[1], idx[2], idx[3]);
    let mut count = 0u64;
    for line in lines {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let cols = split_csv_line(&line);
        let root_id: u64 = match cols[i_id].trim().parse() {
            Ok(v) => v,
            Err(_) => continue, // trailing blank / malformed row
        };
        let meta = NeuronMeta {
            root_id,
            region: cols
                .get(i_region)
                .map(|s| s.trim())
                .unwrap_or("")
                .to_string(),
            cell_type: cols.get(i_ct).map(|s| s.trim()).unwrap_or("").to_string(),
            nt_type: cols.get(i_nt).map(|s| s.trim()).unwrap_or("").to_string(),
        };
        f(meta)?;
        count += 1;
    }
    Ok(count)
}

fn bad_row(line: &str, e: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("bad row `{line}`: {e}"),
    )
}

/// Small helper so callers can sanity-check a gz file without full parse.
pub fn peek_header(path: &std::path::Path) -> std::io::Result<String> {
    let mut r = open_gz(path)?;
    let mut first = String::new();
    r.read_line(&mut first)?;
    Ok(first)
}

/// Exposed for tests: whether a file opens as gzip.
pub fn is_gzip(path: &std::path::Path) -> std::io::Result<bool> {
    let mut f = std::fs::File::open(path)?;
    let mut m = [0u8; 2];
    f.read_exact(&mut m)?;
    Ok(m == [0x1f, 0x8b])
}
