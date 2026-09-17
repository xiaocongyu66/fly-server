use fly_server::engine::EngineConfig;
use fly_server::{api, substrate};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = match args.get(1).map(String::as_str) {
        Some("compile-substrate") => cmd_compile(&args[2..]),
        Some("serve") => cmd_serve(&args[2..]),
        Some("bench") => cmd_bench(&args[2..]),
        Some("--version" | "-V" | "version") => {
            println!("fly-server {}", fly_server::VERSION);
            0
        }
        _ => {
            eprintln!(
                "fly-server {} — OpenAI-style API for connectome LIF simulation\n\n\
                 USAGE:\n  \
                 fly-server compile-substrate --data <dir> --out <substrate.flybin> [--quant u8|f32]\n  \
                 fly-server serve --substrate <substrate.flybin> [--port 8000] [--threads N] [--simd on|off]\n  \
                 fly-server bench --substrate <substrate.flybin> [--ticks 2000] [--threads N] [--simd on|off]\n\n\
                 COMMANDS:\n  \
                 compile-substrate  Compile FlyWire v783 Princeton CSV.gz dumps into .flybin\n  \
                 serve              Start the API server (sessions, observe/step, SSE activity)\n  \
                 bench              Throughput benchmark over a live session\n\n\
                 OPTIMIZATION SWITCHES (default: quant=u8 at compile time, threads=1, simd=on):\n  \
                 --quant u8|f32     Substrate weight storage (u8 = ~4x smaller, near-lossless)\n  \
                 --threads N        Worker threads; >1 enables parallel tick phases (1 = exact replays)\n  \
                 --simd on|off      NEON f32x4 decay/integrate kernels (aarch64)",
                fly_server::VERSION
            );
            2
        }
    };
    std::process::exit(code);
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn cmd_compile(args: &[String]) -> i32 {
    let data = flag(args, "--data").unwrap_or_else(|| "data".into());
    let out = flag(args, "--out").unwrap_or_else(|| "substrate.flybin".into());
    let quant = match flag(args, "--quant").as_deref() {
        Some("f32") => substrate::Quant::F32,
        _ => substrate::Quant::U8,
    };
    let t0 = Instant::now();
    eprintln!("compiling substrate from {} (quant: {:?}) ...", data, quant);
    match substrate::compile_flywire(&PathBuf::from(&data), &PathBuf::from(&out), quant) {
        Ok(r) => {
            eprintln!(
                "ok: {} neurons, {} edges (from {} raw rows, {} skipped), {} synapses, tables(region/ct/nt) {:?}, {:?}",
                r.n_neurons, r.n_edges_aggregated, r.n_raw_rows, r.n_rows_skipped_unknown_root, r.total_synapses, r.tables, t0.elapsed()
            );
            eprintln!("wrote {}", out);
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

fn parse_engine_cfg(args: &[String]) -> EngineConfig {
    let mut cfg = EngineConfig::default();
    if let Some(t) = flag(args, "--threads").and_then(|v| v.parse::<usize>().ok()) {
        cfg.n_threads = t.max(1);
    }
    match flag(args, "--simd").as_deref() {
        Some("off" | "false" | "0") => cfg.use_simd = false,
        _ => cfg.use_simd = true,
    }
    cfg
}

fn cmd_bench(args: &[String]) -> i32 {
    let substrate_path = match flag(args, "--substrate") {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("error: --substrate <path> required");
            return 1;
        }
    };
    let ticks: u32 = flag(args, "--ticks")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    // sustained sensory load: inject current into N neurons every 10 ticks
    // so the scatter phase carries real traffic (0 spikes would only measure
    // barrier overhead)
    let stim: u32 = flag(args, "--stim")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let cfg = parse_engine_cfg(args);
    let s = match substrate::load(&substrate_path) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("error loading substrate: {e}");
            return 1;
        }
    };
    eprintln!(
        "bench: {} neurons, {} edges, threads={}, simd={}, stim={}, {} ticks",
        s.n_neurons(),
        s.header.n_edges,
        cfg.n_threads,
        cfg.use_simd,
        stim,
        ticks
    );
    let n_neurons = s.n_neurons() as u32;
    let stim_ids: Vec<u32> = (0..stim)
        .map(|k| (k * (n_neurons / stim.max(1)).max(1)) as u32 % n_neurons)
        .collect();
    let mut engine = fly_server::engine::Engine::new(s, cfg);
    // warm-up
    let _ = engine.run_ticks(50, &mut |_, _| {});
    let t0 = Instant::now();
    let mut total_spikes = 0u64;
    let mut done = 0u32;
    let mut last = fly_server::engine::TickReport {
        tick: 0,
        t_ms: 0.0,
        n_spikes: 0,
        mean_v: 0.0,
    };
    while done < ticks {
        let batch = 10.min(ticks - done);
        engine.inject(&stim_ids, 30.0);
        last = engine.run_ticks(batch, &mut |r: &fly_server::engine::TickReport, _view| {
            total_spikes += r.n_spikes as u64;
        });
        done += batch;
    }
    let wall = t0.elapsed().as_secs_f64();
    let tps = ticks as f64 / wall;
    println!(
        "{{\"ticks\": {}, \"wall_s\": {:.3}, \"ticks_per_sec\": {:.0}, \"total_spikes\": {}, \"mean_v\": {:.3}, \"sim_ms\": {:.1}}}",
        ticks,
        wall,
        tps,
        total_spikes,
        last.mean_v,
        last.t_ms
    );
    0
}

/// Default data directory: `$HOME/.fly-server` (override with --data-dir).
fn default_data_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".fly-server")
}

/// Create the standard directory layout and return the root.
fn init_data_dir(root: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(root.join("datasets"))?;
    std::fs::create_dir_all(root.join("substrates"))?;
    std::fs::create_dir_all(root.join("logs"))?;
    Ok(())
}

fn cmd_serve(args: &[String]) -> i32 {
    let data_root = flag(args, "--data-dir")
        .map(PathBuf::from)
        .unwrap_or_else(default_data_dir);
    if let Err(e) = init_data_dir(&data_root) {
        eprintln!("error: cannot init data dir {}: {e}", data_root.display());
        return 1;
    }
    let port: u16 = flag(args, "--port")
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);
    let host = flag(args, "--host").unwrap_or_else(|| "0.0.0.0".into());
    let admin_user = flag(args, "--admin-user").unwrap_or_else(|| "admin".into());
    let admin_pass = flag(args, "--admin-pass").unwrap_or_else(|| "flyserver".into());
    let admin_dist = flag(args, "--admin-dist").map(PathBuf::from);
    let cfg = parse_engine_cfg(args);

    // substrate resolution: --substrate explicit, else data_dir/substrates/*.flybin
    let substrate_path: PathBuf = match flag(args, "--substrate") {
        Some(p) => PathBuf::from(p),
        None => {
            let dir = data_root.join("substrates");
            match std::fs::read_dir(&dir)
                .map(|rd| {
                    let mut v: Vec<PathBuf> = rd
                        .filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| p.extension().map(|x| x == "flybin").unwrap_or(false))
                        .collect();
                    v.sort();
                    v
                })
                .map(|mut v| v.pop())
                .unwrap_or(None)
            {
                Some(p) => {
                    eprintln!("auto: using {}", p.display());
                    p
                }
                None => {
                    // auto-deploy: download the lite tier and compile it
                    eprintln!("auto: no substrate found — bootstrapping lite tier (FlyWire v783, ~28 MB)…");
                    let lite_dir = data_root.join("datasets").join("lite");
                    std::fs::create_dir_all(&lite_dir).ok();
                    let raw = lite_dir.join("connections.csv.gz");
                    let neurons = lite_dir.join("neurons.csv.gz");
                    let lite_url = |name: &str| {
                        format!(
                            "https://raw.githubusercontent.com/ruvnet/RuVector/research/connectome-ruvector/examples/connectome-fly/assets/{name}"
                        )
                    };
                    if fly_server::datasets::download_to(
                        &lite_url("connections_princeton.csv.gz"),
                        &raw,
                    )
                    .is_err()
                    {
                        eprintln!("error: lite download failed — pass --substrate explicitly");
                        return 1;
                    }
                    if fly_server::datasets::download_to(&lite_url("neurons.csv.gz"), &neurons)
                        .is_err()
                    {
                        eprintln!("error: neurons download failed");
                        return 1;
                    }
                    let out = data_root.join("substrates").join("lite.flybin");
                    match substrate::compile_flywire(&lite_dir, &out, substrate::Quant::U8) {
                        Ok(r) => eprintln!(
                            "auto: compiled {} neurons / {} edges → {}",
                            r.n_neurons,
                            r.n_edges_aggregated,
                            out.display()
                        ),
                        Err(e) => {
                            eprintln!("error: compile failed: {e}");
                            return 1;
                        }
                    }
                    out
                }
            }
        }
    };
    let substrate = match substrate::load(&substrate_path) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("error loading substrate: {e}");
            return 1;
        }
    };
    let substrate_id = substrate_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "substrate".into());
    let admin = Arc::new(fly_server::admin::AdminStore::new(
        &admin_user,
        &admin_pass,
        data_root.join("fly.db"),
    ));
    let datasets = Arc::new(fly_server::datasets::DatasetStore::new(
        data_root.join("datasets"),
    ));
    eprintln!(
        "fly-server {} — substrate \"{}\": {} neurons, {} edges, threads={}, simd={}, admin=\"{}\", listening on {}:{}",
        fly_server::VERSION, substrate_id, substrate.n_neurons(), substrate.header.n_edges, cfg.n_threads, cfg.use_simd, admin_user, host, port
    );
    match api::run_server(
        substrate,
        substrate_id,
        port,
        cfg,
        admin_dist,
        admin,
        datasets,
        &host,
    ) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}
