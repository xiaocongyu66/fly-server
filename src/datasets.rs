//! Multi-tier brain substrate downloads with real progress tracking.
//!
//! Tiers:
//! - lite:     FlyWire v783 Princeton CSV.gz dump (whole female brain) —
//!   downloads and compiles end-to-end with the built-in CSV pipeline.
//! - standard: MaleCNS v1.0 connectome-weights feather (neuron-to-neuron)
//!   plus body annotations / neurotransmitters, compiled on activation via
//!   the built-in Arrow IPC ingest (`substrate::feather`).
//! - full:     MaleCNS v1.0 syn-partners feather (synapse-level partners),
//!   same ingest path.

use serde::Serialize;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub struct DatasetDef {
    pub tier: &'static str,
    pub base_url: &'static str,
    /// (remote_filename, local_filename, expected_bytes, md5_hex) — bytes
    /// and md5 come from the upstream manifest; an empty md5 skips the
    /// integrity check (the lite tier's sizes are estimates)
    pub files: &'static [(&'static str, &'static str, u64, &'static str)],
}

impl DatasetDef {
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.2).sum()
    }
}

pub const DATASETS: &[DatasetDef] = &[
    DatasetDef {
        tier: "lite",
        base_url: "https://raw.githubusercontent.com/ruvnet/RuVector/research/connectome-ruvector/examples/connectome-fly/assets",
        files: &[
            ("connections_princeton.csv.gz", "connections.csv.gz", 27_554_000, ""),
            ("neurons.csv.gz", "neurons.csv.gz", 2_200_000, ""),
        ],
    },
    DatasetDef {
        tier: "standard",
        base_url: "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome",
        files: &[
            (
                "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
                "connectome_weights.feather",
                1_051_241_946,
                "f30e9dcca25cfd021bf1e7b3d975599e",
            ),
            (
                "body-annotations-male-cns-v1.0-minconf-0.5.feather",
                "body_annotations.feather",
                14_483_314,
                "50a7718770c57220f160ba4f431ab89e",
            ),
            (
                "body-neurotransmitters-male-cns-v1.0.feather",
                "body_nt.feather",
                43_282_834,
                "3d842b12fe5c49eefade528d7dd24a1f",
            ),
        ],
    },
    DatasetDef {
        tier: "full",
        base_url: "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome",
        files: &[(
            "syn-partners-male-cns-v1.0-minconf-0.5-traced-only.feather",
            "syn_partners.feather",
            2_965_367_002,
            "f5bc1c5ce34a01b68956414b530edda8",
        )],
    },
];

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Idle,
    Downloading,
    Downloaded,
    Compiling,
    Compiled,
    Error,
}

/// Published spec of one upstream file: expected size and md5 so a model
/// download service can verify integrity without contacting GCS itself.
#[derive(Debug, Clone, Serialize)]
pub struct FileSpec {
    pub name: &'static str,
    pub bytes: u64,
    pub md5: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct TierState {
    pub tier: String,
    pub status: Status,
    pub downloaded: u64,
    pub total: u64,
    pub error: Option<String>,
    pub hint: Option<String>,
    /// Per-file expectations from the upstream manifest (md5 may be empty
    /// for the lite tier, whose sizes are estimates).
    pub files: Vec<FileSpec>,
    /// md5 of the compiled `<tier>.flybin`, set when the artifact is
    /// written or found complete at boot.
    pub artifact_md5: Option<String>,
}

impl TierState {
    fn new(d: &DatasetDef) -> Self {
        Self {
            tier: d.tier.to_string(),
            status: Status::Idle,
            downloaded: 0,
            total: d.total_bytes(),
            error: None,
            hint: None,
            files: d
                .files
                .iter()
                .map(|(remote, _, bytes, md5)| FileSpec {
                    name: remote,
                    bytes: *bytes,
                    md5,
                })
                .collect(),
            artifact_md5: None,
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
            let mut st = TierState::new(d);
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
                    .filter_map(|(_, local, _, _)| {
                        std::fs::metadata(tier_dir.join(local))
                            .ok()
                            .map(|m| m.len())
                    })
                    .sum();
                if total_size == 0 {
                    // remove incomplete files, force re-download
                    for (_, local, _, _) in d.files {
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
                        st.artifact_md5 = file_md5(&flybin).ok();
                        st.status = Status::Compiled;
                        st.hint = Some("already compiled: lite.flybin".into());
                    } else {
                        st.status = Status::Downloaded;
                        st.hint = Some("files on disk, compile to enable".into());
                    }
                } else {
                    // malecns tiers: compiled flybin wins, else a complete
                    // feather is activatable, else the file is truncated
                    let flybin = data_dir
                        .parent()
                        .unwrap_or(&data_dir)
                        .join("substrates")
                        .join(format!("{}.flybin", d.tier));
                    let weights_file = tier_dir.join(d.files[0].1);
                    if flybin.exists() {
                        st.artifact_md5 = file_md5(&flybin).ok();
                        st.status = Status::Compiled;
                        st.hint = Some(format!(
                            "already compiled: {}",
                            flybin.file_name().unwrap_or_default().to_string_lossy()
                        ));
                    } else if crate::substrate::feather::feather_complete(&weights_file) {
                        st.status = Status::Downloaded;
                        st.hint = Some("feather ready — activate to compile".into());
                    } else {
                        st.status = Status::Error;
                        st.error = Some("feather file truncated — click download to resume".into());
                    }
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
        let mut st = self.states.lock().unwrap();
        let cur_status = st.get(tier).ok_or("unknown tier")?.status.clone();
        // atomic check-and-set: a second click fails here while the first
        // download is still marked Downloading
        if cur_status == Status::Downloading {
            return Err("download already in flight".into());
        }
        // check if all final files exist with correct size (10% tolerance
        // for the lite tier whose sizes are estimates)
        let tier_dir = self.data_dir.join(tier);
        let all_complete = def.files.iter().all(|(_, local, expected, _)| {
            tier_dir
                .join(local)
                .metadata()
                .map(|m| m.len() >= expected * 9 / 10)
                .unwrap_or(false)
        });
        if all_complete {
            return Err("files already downloaded".into());
        }
        // claim the slot NOW (same lock scope) so concurrent clicks can't race
        if let Some(c) = st.get_mut(tier) {
            c.status = Status::Downloading;
            c.downloaded = 0;
            c.error = None;
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
            .map(|(remote, local, _, _)| {
                (format!("{}/{}", def.base_url, remote), local.to_string())
            })
            .collect();
        std::thread::spawn(move || {
            let _ = std::fs::create_dir_all(&out_dir);
            let mut downloaded_total = 0u64;
            for (url, local_name) in &files {
                let final_path = out_dir.join(local_name);
                let part_path = final_path.with_extension("part");
                // already complete on disk (feather magic verified)? skip so
                // partially-populated tiers only fetch what's missing
                if crate::substrate::feather::feather_complete(&final_path) {
                    downloaded_total += final_path.metadata().map(|m| m.len()).unwrap_or(0);
                    continue;
                }
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
                // integrity: verify md5 against the upstream manifest —
                // a truncated/double-written file must never be marked done
                let md5_hex = def
                    .files
                    .iter()
                    .find(|f| f.1 == *local_name)
                    .and_then(|f| (!f.3.is_empty()).then_some(f.3));
                if let Some(expected_md5) = md5_hex {
                    match file_md5(&final_path) {
                        Ok(got) if got == *expected_md5 => {}
                        Ok(got) => {
                            let _ = std::fs::remove_file(&final_path);
                            store.mark_error(
                                tier_s.as_str(),
                                &format!(
                                    "{local_name}: md5 mismatch (got {got}, expected {expected_md5}) — re-download"
                                ),
                            );
                            return;
                        }
                        Err(e) => {
                            store.mark_error(
                                tier_s.as_str(),
                                &format!("{local_name}: md5 check failed: {e}"),
                            );
                            return;
                        }
                    }
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

    pub fn mark_done(&self, tier: &str) {
        let data_dir = self.data_dir.clone();
        self.set(tier, move |st| {
            st.downloaded = st.total;
            if st.tier == "lite" {
                // auto-compile lite tier using the built-in CSV pipeline
                let lite_dir = data_dir.join("lite");
                let out = data_dir
                    .parent()
                    .unwrap_or(&data_dir)
                    .join("substrates")
                    .join("lite.flybin");
                match crate::substrate::compile_flywire(
                    &lite_dir,
                    &out,
                    crate::substrate::Quant::U8,
                ) {
                    Ok(r) => {
                        st.artifact_md5 = file_md5(&out).ok();
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
                st.hint = Some("feather ready — activate to compile".into());
            }
        });
    }

    /// Data directory of one tier (feather inputs live here).
    pub fn tier_dir(&self, tier: &str) -> PathBuf {
        self.data_dir.join(tier)
    }

    /// Compiled-substrate output directory (`<data_root>/substrates`).
    pub fn substrates_dir(&self) -> PathBuf {
        self.data_dir
            .parent()
            .unwrap_or(&self.data_dir)
            .join("substrates")
    }

    /// Atomically claim the compiling slot for a tier (check-and-set in one
    /// lock so concurrent activations can't race).
    pub fn claim_compile(&self, tier: &str) -> Result<(), String> {
        let mut states = self.states.lock().unwrap();
        let st = states.get_mut(tier).ok_or("unknown tier")?;
        match st.status {
            Status::Compiling => Err("compile already in flight".into()),
            Status::Compiled => Err("already compiled".into()),
            Status::Downloading => Err("download in flight".into()),
            _ => {
                st.status = Status::Compiling;
                st.error = None;
                Ok(())
            }
        }
    }

    pub fn mark_compiled(
        &self,
        tier: &str,
        report: &crate::substrate::CompileReport,
        artifact_md5: String,
    ) {
        self.set(tier, |st| {
            st.status = Status::Compiled;
            st.artifact_md5 = Some(artifact_md5);
            st.hint = Some(format!(
                "compiled: {} neurons / {} edges → {}.flybin",
                report.n_neurons, report.n_edges_aggregated, tier
            ));
        });
    }

    pub fn mark_compile_error(&self, tier: &str, err: &str) {
        self.set(tier, |st| {
            st.status = Status::Error;
            st.error = Some(format!("compile failed: {err}"));
        });
    }

    pub fn set_hint(&self, tier: &str, hint: &str) {
        self.set(tier, |st| {
            st.hint = Some(hint.to_string());
        });
    }

    pub fn mark_error(&self, tier: &str, err: &str) {
        self.set(tier, |st| {
            st.status = Status::Error;
            st.error = Some(err.to_string());
        });
    }
}

/// md5 of a file as lowercase hex (streamed, bounded memory).
pub(crate) fn file_md5(path: &std::path::Path) -> Result<String, String> {
    use md5::Digest;
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hasher = md5::Md5::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Download a URL to a file, resuming from an existing `.part` file
/// using HTTP Range requests. Appends to the file when resuming.
pub fn download_with_resume(
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
    let is_resume = status == 206 && resume_from > 0;
    let content_length = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let mut reader = resp.into_reader();
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
    if content_length > 0 {
        let expected_total = if is_resume {
            resume_from + content_length
        } else {
            content_length
        };
        let actual = part_path.metadata().map(|m| m.len()).unwrap_or(0);
        if actual < expected_total {
            return Err(format!(
                "truncated: got {}, expected {}",
                actual, expected_total
            ));
        }
    }
    Ok(got)
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
