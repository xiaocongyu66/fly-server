use fly_server::substrate;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = match args.get(1).map(String::as_str) {
        Some("compile-substrate") => cmd_compile(&args[2..]),
        Some("serve") => cmd_serve(&args[2..]),
        Some("--version" | "-V" | "version") => {
            println!("fly-server {}", fly_server::VERSION);
            0
        }
        _ => {
            eprintln!(
                "fly-server {} — OpenAI-style API for connectome LIF simulation\n\n\
                 USAGE:\n  \
                 fly-server compile-substrate --data <dir> --out <substrate.flybin>\n  \
                 fly-server serve --substrate <substrate.flybin> [--port 8000]\n\n\
                 COMMANDS:\n  \
                 compile-substrate  Compile FlyWire v783 Princeton CSV.gz dumps into .flybin\n  \
                 serve              Start the API server (sessions, observe/step, SSE activity)",
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
    let t0 = std::time::Instant::now();
    eprintln!("compiling substrate from {} ...", data);
    match substrate::compile_flywire(&PathBuf::from(&data), &PathBuf::from(&out)) {
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

fn cmd_serve(args: &[String]) -> i32 {
    let substrate_path = match flag(args, "--substrate") {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("error: --substrate <path> required");
            return 1;
        }
    };
    let _port: u16 = flag(args, "--port").and_then(|p| p.parse().ok()).unwrap_or(8000);
    match substrate::load(&substrate_path) {
        Ok(s) => {
            eprintln!(
                "loaded substrate: {} neurons, {} edges from {}",
                s.n_neurons(),
                s.header.n_edges,
                s.header.source
            );
            eprintln!("serve: HTTP layer not yet implemented in this build");
            0
        }
        Err(e) => {
            eprintln!("error loading substrate: {e}");
            1
        }
    }
}
