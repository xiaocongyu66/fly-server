//! Admin pages: login, dashboard, sessions, activity, keys.

use crate::api;
use dioxus::prelude::*;

fn card(title: &str, value: String, accent: &str) -> Element {
    rsx! {
        div { class: "rounded-lg border bg-card p-5 shadow-sm",
            div { class: "text-xs font-medium text-muted-foreground", "{title}" }
            div { class: "mt-1 text-2xl font-semibold {accent}", "{value}" }
        }
    }
}

// ---------- login ----------

#[component]
pub fn Login(on_done: EventHandler) -> Element {
    let mut user = use_signal(|| "admin".to_string());
    let mut pass = use_signal(String::new);
    let mut err = use_signal(String::new);
    let mut busy = use_signal(|| false);

    rsx! {
        div { class: "min-h-screen flex items-center justify-center bg-background",
            div { class: "w-80 rounded-lg border bg-card p-6 shadow-sm",
                h1 { class: "text-lg font-semibold mb-1", "🪰 fly-admin" }
                p { class: "text-xs text-muted-foreground mb-4", "Sign in to the fly-server console" }
                input {
                    class: "w-full mb-2 rounded-md border bg-transparent px-3 py-2 text-sm",
                    placeholder: "username",
                    value: "{user}",
                    oninput: move |e| user.set(e.value()),
                }
                input {
                    class: "w-full mb-2 rounded-md border bg-transparent px-3 py-2 text-sm",
                    r#type: "password",
                    placeholder: "password",
                    value: "{pass}",
                    oninput: move |e| pass.set(e.value()),
                }
                if !err.read().is_empty() {
                    div { class: "mb-2 text-xs text-red-500", "{err}" }
                }
                button {
                    class: "w-full rounded-md bg-primary px-3 py-2 text-sm text-primary-foreground hover:opacity-90 disabled:opacity-50",
                    disabled: "{busy}",
                    onclick: move |_| {
                        to_owned![user, pass, err, busy, on_done];
                        async move {
                            busy.set(true);
                            err.set(String::new());
                            match api::post("/v1/admin/login", serde_json::json!({
                                "username": user.read().clone(),
                                "password": pass.read().clone(),
                            })).await {
                                Ok(v) => {
                                    api::set_token(v["token"].as_str().unwrap_or(""));
                                    on_done.call(());
                                }
                                Err(e) => err.set(e),
                            }
                            busy.set(false);
                        }
                    },
                    if busy() { "Signing in…" } else { "Sign in" }
                }
                p { class: "mt-3 text-[10px] text-muted-foreground", "default: admin / flyserver" }
            }
        }
    }
}

// ---------- dashboard ----------

#[component]
pub fn Dashboard() -> Element {
    let mut substrate_info = use_signal(String::new);
    let mut n_neurons = use_signal(|| "—".to_string());
    let mut n_edges = use_signal(|| "—".to_string());
    let mut sessions = use_signal(|| "—".to_string());
    let mut ticks = use_signal(|| "—".to_string());
    let mut requests = use_signal(|| "—".to_string());
    let mut err = use_signal(String::new);

    use_effect(move || {
        to_owned![substrate_info, n_neurons, n_edges, sessions, ticks, requests, err];
        spawn(async move {
            match api::get("/v1/models").await {
                Ok(v) => {
                    let s = &v["data"][0];
                    substrate_info.set(s["id"].as_str().unwrap_or("?").to_string());
                    n_neurons.set(s["n_neurons"].to_string());
                    n_edges.set(s["n_edges"].to_string());
                }
                Err(e) => err.set(e),
            }
            if let Ok(v) = api::get("/v1/sessions").await {
                sessions.set(v["data"].as_array().map(|a| a.len().to_string()).unwrap_or("0".into()));
            }
            if let Ok(v) = api::get("/v1/admin/usage").await {
                ticks.set(v["total"]["ticks"].to_string());
                requests.set(v["total"]["requests"].to_string());
            }
        });
    });

    rsx! {
        div { class: "p-6 space-y-4",
            h1 { class: "text-2xl font-semibold tracking-tight", "Dashboard" }
            if !err.read().is_empty() {
                div { class: "text-sm text-red-500", "{err}" }
            }
            p { class: "text-sm text-muted-foreground", "substrate: {substrate_info}" }
            div { class: "grid grid-cols-2 md:grid-cols-5 gap-3",
                {card("Neurons", n_neurons.read().clone(), "")}
                {card("Synapses", n_edges.read().clone(), "")}
                {card("Sessions", sessions.read().clone(), "")}
                {card("Billed ticks", ticks.read().clone(), "")}
                {card("API requests", requests.read().clone(), "")}
            }
        }
    }
}

// ---------- sessions + stimulator ----------

#[component]
pub fn Sessions() -> Element {
    let mut rows = use_signal(Vec::<serde_json::Value>::new);
    let mut err = use_signal(String::new);
    let mut selected = use_signal(String::new);
    let mut region = use_signal(String::new);
    let mut current = use_signal(|| 30.0f64);
    let mut steps = use_signal(|| 100u64);
    let mut last_result = use_signal(String::new);
    let regions = use_signal(Vec::<String>::new);

    let refresh = move |_| {
        to_owned![rows, err];
        spawn(async move {
            match api::get("/v1/sessions").await {
                Ok(v) => rows.set(v["data"].as_array().cloned().unwrap_or_default()),
                Err(e) => err.set(e),
            }
        });
    };

    use_effect(move || {
        to_owned![rows, err, regions];
        spawn(async move {
            match api::get("/v1/models").await {
                Ok(v) => {
                    let rs: Vec<String> = v["data"][0]["regions"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                        .unwrap_or_default();
                    regions.set(rs);
                }
                Err(e) => err.set(e),
            }
            match api::get("/v1/sessions").await {
                Ok(v) => rows.set(v["data"].as_array().cloned().unwrap_or_default()),
                Err(e) => err.set(e),
            }
        });
    });

    rsx! {
        div { class: "p-6 space-y-4",
            h1 { class: "text-2xl font-semibold tracking-tight", "Sessions" }
            if !err.read().is_empty() {
                div { class: "text-sm text-red-500", "{err}" }
            }
            div { class: "flex gap-2",
                button {
                    class: "rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground hover:opacity-90",
                    onclick: move |_| {
                        to_owned![rows, err];
                        spawn(async move {
                            if let Err(e) = api::post("/v1/sessions", serde_json::json!({})).await {
                                err.set(e);
                            }
                            if let Ok(v) = api::get("/v1/sessions").await {
                                rows.set(v["data"].as_array().cloned().unwrap_or_default());
                            }
                        });
                    },
                    "+ New session"
                }
                button {
                    class: "rounded-md border px-3 py-1.5 text-sm hover:bg-accent",
                    onclick: refresh,
                    "Refresh"
                }
            }
            table { class: "w-full text-sm border rounded-lg overflow-hidden",
                thead { class: "bg-muted/50 text-left",
                    tr { th { class: "p-2", "id" } th { class: "p-2", "tick" } th { class: "p-2", "dt_ms" } th { class: "p-2", "" } }
                }
                tbody {
                    for row in rows.read().iter() {
                        {let id = row["id"].as_str().unwrap_or("").to_string();
                        let tick = row["current_tick"].to_string();
                        let dt = row["dt_ms"].to_string();
                        let id2 = id.clone();
                        rsx! {
                            tr { class: "border-t",
                                td { class: "p-2 font-mono text-xs", class: if selected.read().as_str() == id { "p-2 font-mono text-xs bg-accent" } else { "p-2 font-mono text-xs" },
                                    onclick: move |_| selected.set(id2.clone()), "{id}" }
                                td { class: "p-2", "{tick}" }
                                td { class: "p-2", "{dt}" }
                                td { class: "p-2 text-right",
                                    button {
                                        class: "text-xs text-red-500 hover:underline",
                                        onclick: move |_| {
                                            to_owned![rows, err];
                                            let id = id.clone();
                                            spawn(async move {
                                                let _ = api::del(&format!("/v1/sessions/{id}")).await;
                                                if let Ok(v) = api::get("/v1/sessions").await {
                                                    rows.set(v["data"].as_array().cloned().unwrap_or_default());
                                                }
                                            });
                                        },
                                        "delete"
                                    }
                                }
                            }
                        }}
                    }
                }
            }

            h2 { class: "text-lg font-semibold pt-2", "Stimulator" }
            p { class: "text-xs text-muted-foreground", "selected session: {selected}" }
            div { class: "grid grid-cols-2 md:grid-cols-4 gap-3 items-end",
                select {
                    class: "rounded-md border bg-transparent px-3 py-2 text-sm",
                    value: "{region}",
                    onchange: move |e| region.set(e.value()),
                    option { value: "", "— region —" }
                    for r in regions.read().iter() {
                        option { value: "{r}", "{r}" }
                    }
                }
                div {
                    label { class: "text-xs text-muted-foreground", "current" }
                    input {
                        class: "w-full rounded-md border bg-transparent px-3 py-2 text-sm",
                        value: "{current}",
                        oninput: move |e| current.set(e.value().parse().unwrap_or(30.0)),
                    }
                }
                div {
                    label { class: "text-xs text-muted-foreground", "steps" }
                    input {
                        class: "w-full rounded-md border bg-transparent px-3 py-2 text-sm",
                        value: "{steps}",
                        oninput: move |e| steps.set(e.value().parse().unwrap_or(100)),
                    }
                }
                button {
                    class: "rounded-md bg-primary px-3 py-2 text-sm text-primary-foreground hover:opacity-90",
                    onclick: move |_| {
                        to_owned![selected, region, current, steps, last_result, err];
                        spawn(async move {
                            let sid = selected.read().clone();
                            if sid.is_empty() {
                                err.set("select a session first".into());
                                return;
                            }
                            let obs = serde_json::json!({
                                "modality": "visual",
                                "target": {"region": region.read().clone()},
                                "current": current.read().clone(),
                                "duration_ticks": 1,
                            });
                            if let Err(e) = api::post(&format!("/v1/sessions/{sid}/observe"), obs).await {
                                err.set(e);
                                return;
                            }
                            match api::post(&format!("/v1/sessions/{sid}/step"), serde_json::json!({"steps": steps.read().clone()})).await {
                                Ok(v) => last_result.set(format!(
                                    "tick {} · {} spikes · {} actions",
                                    v["tick"], v["n_spikes"], v["actions"].as_array().map(|a| a.len()).unwrap_or(0)
                                )),
                                Err(e) => err.set(e),
                            }
                        });
                    },
                    "Inject + step"
                }
            }
            if !last_result.read().is_empty() {
                div { class: "text-sm text-green-600 font-mono", "{last_result}" }
            }
        }
    }
}

// ---------- live activity (SSE) ----------

#[component]
pub fn Activity() -> Element {
    let mut session_id = use_signal(String::new);
    let mut connected = use_signal(|| false);
    let mut bars = use_signal(Vec::<u64>::new);
    let mut total = use_signal(|| 0u64);

    rsx! {
        div { class: "p-6 space-y-4",
            h1 { class: "text-2xl font-semibold tracking-tight", "Live activity" }
            div { class: "flex gap-2 items-end",
                input {
                    class: "w-72 rounded-md border bg-transparent px-3 py-2 text-sm font-mono",
                    placeholder: "session id (sess_…)",
                    value: "{session_id}",
                    oninput: move |e| session_id.set(e.value()),
                }
                button {
                    class: "rounded-md bg-primary px-3 py-2 text-sm text-primary-foreground hover:opacity-90",
                    onclick: move |_| {
                        to_owned![session_id, connected, bars, total];
                        bars.set(Vec::new());
                        total.set(0);
                        let url = format!(
                            "{}/v1/sessions/{}/activity?token={}",
                            api::base(),
                            session_id.read(),
                            api::token().unwrap_or_default()
                        );
                        spawn(async move {
                            let es = web_sys::EventSource::new(&url).unwrap();
                            let cb = wasm_bindgen::closure::Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
                                let data = e.data().as_string().unwrap_or_default();
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                                    let n = v["n_spikes"].as_u64().unwrap_or(0);
                                    total.with_mut(|t| *t += n);
                                    let mut b = bars.write();
                                    b.push(n);
                                    if b.len() > 120 { b.remove(0); }
                                }
                            }) as Box<dyn FnMut(web_sys::MessageEvent)>);
                            es.set_onmessage(Some(cb.as_ref().unchecked_ref()));
                            cb.forget();
                            connected.set(true);
                        });
                    },
                    if connected() { "Connected ✓" } else { "Connect" }
                }
                div { class: "text-sm text-muted-foreground", "total spikes: {total}" }
            }
            div { class: "flex items-end gap-[2px] h-40 border rounded-lg p-2 bg-muted/30",
                for (i, b) in bars.read().iter().enumerate() {
                    {let h = (*b).clamp(0, 2000) as f64 / 20.0;
                    let id_key = format!("{i}");
                    rsx! {
                        div {
                            key: "{id_key}",
                            class: "w-2 rounded-t bg-primary/70",
                            style: "height: {h.max(2.0)}px",
                            title: "{b}"
                        }
                    }}
                }
            }
            p { class: "text-xs text-muted-foreground", "runs a session observe/step loop from the Stimulator page while watching this" }
        }
    }
}

// ---------- keys ----------

#[component]
pub fn Keys() -> Element {
    let mut rows = use_signal(Vec::<serde_json::Value>::new);
    let mut name = use_signal(|| "default".to_string());
    let mut new_secret = use_signal(String::new);
    let mut err = use_signal(String::new);

    let refresh = move |_| {
        to_owned![rows, err];
        spawn(async move {
            match api::get("/v1/admin/keys").await {
                Ok(v) => rows.set(v.as_array().cloned().unwrap_or_default()),
                Err(e) => err.set(e),
            }
        });
    };

    use_effect(move || {
        to_owned![rows, err];
        spawn(async move {
            match api::get("/v1/admin/keys").await {
                Ok(v) => rows.set(v.as_array().cloned().unwrap_or_default()),
                Err(e) => err.set(e),
            }
        });
    });

    rsx! {
        div { class: "p-6 space-y-4",
            h1 { class: "text-2xl font-semibold tracking-tight", "API keys & billing" }
            if !err.read().is_empty() {
                div { class: "text-sm text-red-500", "{err}" }
            }
            div { class: "flex gap-2 items-end",
                input {
                    class: "rounded-md border bg-transparent px-3 py-2 text-sm",
                    value: "{name}",
                    oninput: move |e| name.set(e.value()),
                }
                button {
                    class: "rounded-md bg-primary px-3 py-2 text-sm text-primary-foreground hover:opacity-90",
                    onclick: move |_| {
                        to_owned![rows, err, new_secret];
                        spawn(async move {
                            match api::post("/v1/admin/keys", serde_json::json!({"name": name.read().clone()})).await {
                                Ok(v) => new_secret.set(v["secret"].as_str().unwrap_or("").to_string()),
                                Err(e) => err.set(e),
                            }
                            if let Ok(v) = api::get("/v1/admin/keys").await {
                                rows.set(v.as_array().cloned().unwrap_or_default());
                            }
                        });
                    },
                    "+ Create key"
                }
                button {
                    class: "rounded-md border px-3 py-2 text-sm hover:bg-accent",
                    onclick: refresh,
                    "Refresh"
                }
            }
            if !new_secret.read().is_empty() {
                div { class: "rounded-md border border-amber-300 bg-amber-50 p-3 text-xs",
                    span { class: "font-semibold", "Copy the secret now — shown once: " }
                    code { class: "font-mono", "{new_secret}" }
                }
            }
            table { class: "w-full text-sm border rounded-lg overflow-hidden",
                thead { class: "bg-muted/50 text-left",
                    tr {
                        th { class: "p-2", "id" } th { class: "p-2", "name" }
                        th { class: "p-2", "enabled" } th { class: "p-2", "requests" }
                        th { class: "p-2", "ticks" } th { class: "p-2", "sessions" }
                        th { class: "p-2", "" }
                    }
                }
                tbody {
                    for row in rows.read().iter() {
                        {let id = row["id"].as_str().unwrap_or("").to_string();
                        let enabled = row["enabled"].as_bool().unwrap_or(false);
                        rsx! {
                            tr { class: "border-t",
                                td { class: "p-2 font-mono text-xs", "{id}" }
                                td { class: "p-2", "{row['name']}" }
                                td { class: "p-2", if enabled { "✓" } else { "✗" } }
                                td { class: "p-2", "{row['usage']['requests']}" }
                                td { class: "p-2", "{row['usage']['ticks']}" }
                                td { class: "p-2", "{row['usage']['sessions_created']}" }
                                td { class: "p-2 text-right",
                                    button {
                                        class: "text-xs text-red-500 hover:underline",
                                        onclick: move |_| {
                                            to_owned![rows, err];
                                            let id = id.clone();
                                            spawn(async move {
                                                let _ = api::post(&format!("/v1/admin/keys/{id}/disable"), serde_json::json!({})).await;
                                                if let Ok(v) = api::get("/v1/admin/keys").await {
                                                    rows.set(v.as_array().cloned().unwrap_or_default());
                                                }
                                            });
                                        },
                                        "disable"
                                    }
                                }
                            }
                        }}
                    }
                }
            }
        }
    }
}
