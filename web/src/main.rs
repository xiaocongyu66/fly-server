use dioxus::prelude::*;

#[derive(Clone, Routable, Debug, PartialEq)]
enum Route {
    #[layout(SidebarLayout)]
    #[route("/")]
    Dashboard {},
    #[route("/sessions")]
    Sessions {},
    #[end_layout]
    #[route("/:..route")]
    NotFound { route: Vec<String> },
}

#[component]
fn Dashboard() -> Element {
    rsx! {
        div { class: "p-6",
            h1 { class: "text-2xl font-semibold tracking-tight", "Dashboard" }
            p { class: "text-sm text-muted-foreground", "fly-server overview" }
        }
    }
}

#[component]
fn Sessions() -> Element {
    rsx! { div { class: "p-6", h1 { class: "text-2xl font-semibold", "Sessions" } } }
}

#[component]
fn NotFound(route: Vec<String>) -> Element {
    rsx! { div { class: "p-6", "404 — {route:?}" } }
}

#[component]
fn SidebarLayout() -> Element {
    rsx! {
        div { class: "min-h-screen bg-background text-foreground",
            aside { class: "w-56 border-r p-4",
                div { class: "text-lg font-bold mb-4", "🪰 fly-admin" }
                nav { class: "space-y-1",
                    Link { to: Route::Dashboard {}, class: "block rounded-md px-3 py-2 text-sm hover:bg-accent", "Dashboard" }
                    Link { to: Route::Sessions {}, class: "block rounded-md px-3 py-2 text-sm hover:bg-accent", "Sessions" }
                }
            }
            main { class: "flex-1", Outlet::<Route> {} }
        }
    }
}

fn main() {
    // client-only app: hydration requires SSR-injected data that static
    // hosting does not have (atob(undefined) crash) — render fresh instead
    dioxus::LaunchBuilder::web()
        .with_cfg(dioxus_web::Config::new().hydrate(false))
        .launch(|| rsx! { Router::<Route> {} });
}
