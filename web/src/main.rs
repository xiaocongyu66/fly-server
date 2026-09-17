mod api;
mod pages;

use dioxus::prelude::*;
use pages::*;

#[derive(Clone, Routable, Debug, PartialEq)]
enum Route {
    #[layout(Shell)]
    #[route("/")]
    Dashboard {},
    #[route("/sessions")]
    Sessions {},
    #[route("/activity")]
    Activity {},
    #[route("/keys")]
    Keys {},
    #[end_layout]
    #[route("/:..route")]
    NotFound { route: Vec<String> },
}

fn main() {
    // client-only render: the registry crate pulls dioxus/fullstack which
    // force-enables dioxus-web/hydrate via feature unification, and static
    // hosting has no injected hydration data (atob(undefined) crash).
    // hydrate(false) at runtime is the only working off-switch.
    dioxus::LaunchBuilder::web()
        .with_cfg(dioxus_web::Config::new().hydrate(false))
        .launch(|| rsx! { Router::<Route> {} });
}

#[component]
fn Shell() -> Element {
    // auth gate: no token -> login screen
    let mut authed = use_signal(|| api::token().is_some());
    if !authed() {
        return rsx! { Login { on_done: move |_| authed.set(true) } };
    }

    let logout = move |_| {
        api::clear_token();
        authed.set(false);
    };

    rsx! {
        div { class: "min-h-screen bg-background text-foreground flex",
            aside { class: "w-52 shrink-0 border-r p-4 flex flex-col",
                div { class: "text-lg font-bold mb-4", "🪰 fly-admin" }
                nav { class: "space-y-1",
                    NavLink { to: Route::Dashboard {}, label: "Dashboard" }
                    NavLink { to: Route::Sessions {}, label: "Sessions" }
                    NavLink { to: Route::Activity {}, label: "Live activity" }
                    NavLink { to: Route::Keys {}, label: "Keys & billing" }
                }
                div { class: "mt-auto",
                    button {
                        class: "text-xs text-muted-foreground hover:text-foreground",
                        onclick: logout,
                        "Sign out"
                    }
                }
            }
            main { class: "flex-1 overflow-x-hidden", Outlet::<Route> {} }
        }
    }
}

#[component]
fn NavLink(to: Route, label: &'static str) -> Element {
    rsx! {
        Link {
            to: to.clone(),
            class: "block rounded-md px-3 py-2 text-sm hover:bg-accent hover:text-accent-foreground",
            "{label}"
        }
    }
}

#[component]
fn NotFound(route: Vec<String>) -> Element {
    rsx! { div { class: "p-6", "404 — {route:?}" } }
}
