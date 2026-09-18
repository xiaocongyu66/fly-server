//! Multi-tier brain substrate downloads with real progress tracking.
//!
//! Tiers:
//! - lite:     FlyWire v783 Princeton CSV.gz dump (whole female brain) —
//!   downloads and compiles end-to-end with the built-in CSV pipeline.
//! - standard: MaleCNS v1.0 connectome-weights feather (neuron-to-neuron).
//! - full:     MaleCNS v1.0 syn-partners feather (synapse-level partners).
//!
//! standard/full download and persist to disk, but compilation requires an
//! Arrow/feather ingest which is not implemented — status reports
//! `downloaded` with an explicit hint instead of pretending.

use serde::Serialize;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub struct DatasetDef {
    pub tier: &'static str,
    pub base_url: &'static str,
    pub bytes: u64,
    /// (remote_filename, local_filename) pairs to download
    pub files: &'static [(&'static str, &'static str)],
}

pub const DATASETS: &[DatasetDef] = &[
    DatasetDef {
        tier: "lite",
        base_url: "https://raw.githubusercontent.com/ruvnet/RuVector/research/connectome-ruvector/examples/connectome-fly/assets",
        bytes: 27_554_000 + 2_200_000,
        files: &[
            ("connections_princeton.csv.gz", "connections.csv.gz"),
            ("neurons.csv.gz", "neurons.csv.gz"),
        ],
    },
    DatasetDef {
        tier: "standard",
        base_url: "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome",
        bytes: 1_050_000_000,
        files: &[("connectome-weights-male-cns-v1.0-minconf-0.5.feather", "connectome_weights.feather")],
    },
    DatasetDef {
        tier: "full",
        base_url: "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome",
        bytes: 2_970_000_000,
        files: &[("syn-partners-male-cns-v1.0-minconf-0.5-traced-only.feather", "syn_partners.feather")],
    },
];

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Idle,
    Downloading,
    Downloaded,
    Compiled,
    Error,
}

#[derive(Debug, Clone, Serialize)]
pub struct TierState {
    pub tier: String,
    pub status: Status,
    pub downloaded: u64,
    pub total: u64,
    pub error: Option<String>,
    pub hint: Option<String>,
}

impl TierState {
    fn new(tier: &str, total: u64) -> Self {
        Self {
            tier: tier.to_string(),
            status: Status::Idle,
            downloaded: 0,
            total,
            error: None,
            hint: None,
        }
    }
}

pub struct DatasetStore {
    states: Mutex<HashMap<String, TierState>>,
    data_dir: PathBuf,
}

impl DatasetStore {
    pub fn new(data_dir: PathBuf) -> Self {
        let mut states = HashMap::new();
        for d in DATASETS {
            let mut st = TierState::new(d.tier, d.bytes);
            // check if any non-empty files exist in the tier directory
            let tier_dir = data_dir.join(d.tier);
            let has_files = std::fs::read_dir(&tier_dir)
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .any(|e| e.metadata().map(|m| m.len() > 1000).unwrap_or(false))
                })
                .unwrap_or(false);
            if has_files {
                // integrity check: verify file sizes match expectations
                let total_size: u64 = d
                    .files
                    .iter()
                    .filter_map(|(_, local)| {
                        std::fs::metadata(tier_dir.join(local))
                            .ok()
                            .map(|m| m.len())
                    })
                    .sum();
                if total_size == 0 {
                    // remove incomplete files, force re-download
                    for (_, local) in d.files {
                        let _ = std::fs::remove_file(tier_dir.join(local));
                    }
                    states.insert(d.tier.to_string(), st);
                    continue;
                }
                if d.tier == "lite" {
                    let flybin = data_dir
                        .parent()
                        .unwrap_or(&data_dir)
                        .join("substrates")
                        .join("lite.flybin");
                    if flybin.exists() {
                        st.status = Status::Compiled;
                        st.hint = Some("already compiled: lite.flybin".into());
                    } else {
                        st.status = Status::Downloaded;
                        st.hint = Some("files on disk, compile to enable".into());
                    }
                } else {
                    st.status = Status::Downloaded;
                    st.hint = Some("feather ingest not implemented".into());
                }
            }
            states.insert(d.tier.to_string(), st);
        }
        Self {
            states: Mutex::new(states),
            data_dir,
        }
    }

    pub fn list(&self) -> Vec<TierState> {
        let mut out: Vec<TierState> = self.states.lock().unwrap().values().cloned().collect();
        out.sort_by_key(|s| match s.tier.as_str() {
            "lite" => 0,
            "standard" => 1,
            _ => 2,
        });
        out
    }

    fn set<F: FnOnce(&mut TierState)>(&self, tier: &str, f: F) {
        if let Some(st) = self.states.lock().unwrap().get_mut(tier) {
            f(st);
        }
    }

    /// Check availability and mark the tier as downloading. The actual IO
    /// thread is spawned by the caller via `spawn_download_thread` (needs the
    /// Arc to publish progress from the worker).
    pub fn prepare_download(&self, tier: &str) -> Result<(), String> {
        let def = DATASETS
            .iter()
            .find(|d| d.tier == tier)
            .ok_or_else(|| format!("unknown tier `{tier}`"))?;
        let st = self.states.lock().unwrap();
        let cur = st.get(tier).ok_or("unknown tier")?;
        if cur.status == Status::Downloading {
            return Err("download already in flight".into());
        }
        // check if all final files exist with correct size
        let tier_dir = self.data_dir.join(tier);
        let per_file = def.bytes / def.files.len().max(1) as u64;
        let all_complete = def.files.iter().all(|(_, local)| {
            let p = tier_dir.join(local);
            match p.metadata() {
                Ok(m) => m.len() >= per_file * 9 / 10, // allow 10% tolerance
                Err(_) => false,
            }
        });
        if all_complete {
            return Err("files already downloaded".into());
        }
        Ok(())
    }

    /// Spawn the background IO thread. `self: &Arc<Self>` lets the thread
    /// publish progress back into the shared store.
    pub fn start_download(self: &Arc<Self>, tier: &str) -> Result<(), String> {
        self.prepare_download(tier)?;
        let def = DATASETS
            .iter()
            .find(|d| d.tier == tier)
            .ok_or("unknown tier")?;
        let store = Arc::clone(self);
        let tier_s = tier.to_string();

        let out_dir = self.data_dir.join(tier);
        let files: Vec<(String, String)> = def
            .files
            .iter()
            .map(|(remote, local)| (format!("{}/{}", def.base_url, remote), local.to_string()))
            .collect();
        std::thread::spawn(move || {
            let _ = std::fs::create_dir_all(&out_dir);
            let mut downloaded_total = 0u64;
            for (url, local_name) in &files {
                let final_path = out_dir.join(local_name);
                let part_path = final_path.with_extension("part");
                // resume: if .part exists, get its size for Range header
                let resume_from = part_path.metadata().map(|m| m.len()).unwrap_or(0);
                let mut ok = false;
                for attempt in 0..3 {
                    match download_with_resume(url, &part_path, resume_from, &|got| {
                        store.update_progress(tier_s.as_str(), downloaded_total + got)
                    }) {
                        Ok(bytes) => {
                            downloaded_total += bytes;
                            ok = true;
                            break;
                        }
                        Err(e) => {
                            if attempt < 2 {
                                eprintln!(
                                    "download retry {}/3 for {}: {}",
                                    attempt + 1,
                                    local_name,
                                    e
                                );
                                std::thread::sleep(std::time::Duration::from_secs(2));
                            } else {
                                let _ = std::fs::remove_file(&part_path);
                                store.mark_error(
                                    tier_s.as_str(),
                                    &format!("{} (3 retries exhausted)", e),
                                );
                                return;
                            }
                        }
                    }
                }
                if !ok {
                    return;
                }
                // rename .part → final name
                if part_path.exists() {
                    std::fs::rename(&part_path, &final_path).ok();
                }
            }
            store.set_total(tier_s.as_str(), downloaded_total);
            store.mark_done(tier_s.as_str());
        });
        Ok(())
    }

    pub fn update_progress(&self, tier: &str, downloaded: u64) {
        self.set(tier, |st| {
            st.downloaded = downloaded;
        });
    }

    pub fn set_total(&self, tier: &str, total: u64) {
        self.set(tier, |st| st.total = total);
    }

    /// Download a URL to a file, resuming from an existing `.part` file
    /// using HTTP Range requests. Appends to the file rather than
    /// overwriting when resuming.
    fn download_with_resume(
        url: &str,
        part_path: &std::path::Path,
        resume_from: u64,
        on_progress: &dyn Fn(u64),
    ) -> Result<u64, String> {
        let mut req = ureq::get(url).timeout(std::time::Duration::from_secs(3600));
        if resume_from > 0 {
            req = req.set("Range", &format!("bytes={}-", resume_from));
        }
        let resp = req.call().map_err(|e| format!("http: {e}"))?;

        let status = resp.status();
        // if server returns 200 for a range request, it doesn't support ranges
        // → start from scratch
        let is_resume = status == 206 && resume_from > 0;

        let content_length = resp
            .header("Content-Length")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        let mut reader = resp.into_reader();

        // open file: append if resuming, create if fresh
        let mut file = if is_resume {
            std::fs::OpenOptions::new()
                .append(true)
                .open(part_path)
                .map_err(|e| format!("open for append: {e}"))?
        } else {
            std::fs::File::create(part_path).map_err(|e| format!("create: {e}"))?
        };

        let start_offset = if is_resume { resume_from } else { 0 };
        let mut buf = [0u8; 64 * 1024];
        let mut got = 0u64;
        loop {
            let n = reader.read(&mut buf).map_err(|e| format!("read: {e}"))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])
                .map_err(|e| format!("write: {e}"))?;
            got += n as u64;
            on_progress(start_offset + got);
        }
        file.flush().ok();

        // verify against Content-Length
        if content_length > 0 {
            let expected_total = if is_resume {
                resume_from + content_length
            } else {
                content_length
            };
            let actual_size = part_path.metadata().map(|m| m.len()).unwrap_or(0);
            if actual_size < expected_total {
                return Err(format!(
                    "truncated: got {} bytes, expected {}",
                    actual_size, expected_total
                ));
            }
        }
        Ok(got)
    }

    pub fn mark_done(&self, tier: &str) {
        let data_dir = self.data_dir.clone();
        self.set(tier, move |st| {
            st.downloaded = st.total;
            if st.tier == "lite" {
                // auto-compile lite tier using the built-in CSV pipeline
                let lite_dir = data_dir.join("datasets").join("lite");
                let out = data_dir.join("substrates").join("lite.flybin");
                match crate::substrate::compile_flywire(
                    &lite_dir,
                    &out,
                    crate::substrate::Quant::U8,
                ) {
                    Ok(r) => {
                        st.status = Status::Compiled;
                        st.hint = Some(format!(
                            "compiled: {} neurons / {} edges → lite.flybin",
                            r.n_neurons, r.n_edges_aggregated
                        ));
                    }
                    Err(e) => {
                        st.status = Status::Error;
                        st.error = Some(format!("compile failed: {e}"));
                    }
                }
            } else {
                st.status = Status::Downloaded;
                st.hint = Some(
                    "feather ingest not implemented: compile requires the arrow crate (roadmap)"
                        .into(),
                );
            }
        });
    }

    pub fn mark_error(&self, tier: &str, err: &str) {
        self.set(tier, |st| {
            st.status = Status::Error;
            st.error = Some(err.to_string());
        });
    }
}

/// One-shot download to a path (no progress callback) — bootstrap path.
pub fn download_to(url: &str, out_path: &std::path::Path) -> Result<u64, String> {
    stream_to_file(url, out_path, 0, &|_| {})
}

/// Stream a URL to a file, reporting progress in bytes.
pub fn stream_to_file(
    url: &str,
    out_path: &std::path::Path,
    expected: u64,
    on_progress: &dyn Fn(u64),
) -> Result<u64, String> {
    let resp = ureq::get(url)
        .timeout(std::time::Duration::from_secs(3600))
        .call()
        .map_err(|e| format!("http: {e}"))?;
    let content_length = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let mut reader = resp.into_reader();
    let mut file = std::fs::File::create(out_path).map_err(|e| format!("create: {e}"))?;
    let mut buf = [0u8; 64 * 1024];
    let mut got = 0u64;
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("write: {e}"))?;
        got += n as u64;
        on_progress(got);
    }
    file.flush().ok();
    let expected_size = if content_length > 0 {
        content_length
    } else {
        expected
    };
    if expected_size > 0 && got < expected_size {
        return Err(format!(
            "truncated: got {} bytes, expected {}",
            got, expected_size
        ));
    }
    Ok(got)
}
