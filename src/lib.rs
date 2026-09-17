pub mod admin;
pub mod api;
pub mod datasets;
pub mod engine;
pub mod error;
pub mod session;
pub mod slots;
pub mod substrate;
pub mod train;
pub mod types;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
