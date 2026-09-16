//! Admin auth, API keys, and per-key usage accounting.
//!
//! - One admin account (CLI-provided user/password, salted SHA-256).
//! - Login returns a random bearer token (in-memory, 24h expiry).
//! - API keys (`fly_sk_...`) gate /v1 business endpoints; admin token also
//!   works everywhere. Keys persist to `admin_keys.json` next to the binary.
//! - Usage (requests / ticks / sessions) is tracked per key id.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const TOKEN_TTL: Duration = Duration::from_secs(24 * 3600);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: String,     // "key_<short>"
    pub secret: String, // "fly_sk_<hex>" — only shown once at creation;
                        // list_keys() projects to KeyView which omits it
    pub name: String,
    pub created_at: u64,
    pub enabled: bool,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct KeyUsage {
    pub requests: u64,
    pub ticks: u64,
    pub sessions_created: u64,
}

#[derive(Debug, Serialize)]
pub struct KeyView {
    pub id: String,
    pub name: String,
    pub created_at: u64,
    pub enabled: bool,
    pub usage: KeyUsage,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct Persisted {
    keys: Vec<ApiKey>,
    usage: HashMap<String, KeyUsage>,
}

pub struct AdminStore {
    admin_user: String,
    admin_pass_hash: [u8; 32],
    salt: String,
    tokens: Mutex<HashMap<String, Instant>>,
    keys: Mutex<HashMap<String, ApiKey>>, // by id
    usage: Mutex<HashMap<String, KeyUsage>>,
    store_path: std::path::PathBuf,
}

fn rand_hex(n_bytes: usize) -> String {
    let mut buf = vec![0u8; n_bytes];
    getrandom::getrandom(&mut buf).expect("system RNG unavailable");
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl AdminStore {
    pub fn new(admin_user: &str, admin_pass: &str, store_path: std::path::PathBuf) -> Self {
        let salt = rand_hex(8);
        let admin_pass_hash: [u8; 32] = {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(salt.as_bytes());
            h.update(admin_pass.as_bytes());
            h.finalize().into()
        };
        let store = Self {
            admin_user: admin_user.to_string(),
            admin_pass_hash,
            salt,
            tokens: Mutex::new(HashMap::new()),
            keys: Mutex::new(HashMap::new()),
            usage: Mutex::new(HashMap::new()),
            store_path,
        };
        store.load_disk();
        store
    }

    fn load_disk(&self) {
        if let Ok(text) = std::fs::read_to_string(&self.store_path) {
            if let Ok(p) = serde_json::from_str::<Persisted>(&text) {
                let mut keys = self.keys.lock().unwrap();
                for k in p.keys {
                    keys.insert(k.id.clone(), k);
                }
                *self.usage.lock().unwrap() = p.usage;
            }
        }
    }

    fn save_disk(&self) {
        let p = Persisted {
            keys: self.keys.lock().unwrap().values().cloned().collect(),
            usage: self.usage.lock().unwrap().clone(),
        };
        if let Ok(text) = serde_json::to_string_pretty(&p) {
            let _ = std::fs::write(&self.store_path, text);
        }
    }

    pub fn login(&self, user: &str, pass: &str) -> Option<String> {
        use sha2::{Digest, Sha256};
        if user != self.admin_user {
            return None;
        }
        let mut h = Sha256::new();
        h.update(self.salt.as_bytes());
        h.update(pass.as_bytes());
        let got: [u8; 32] = h.finalize().into();
        if got != self.admin_pass_hash {
            return None;
        }
        let token = format!("fly_at_{}", rand_hex(24));
        self.tokens
            .lock()
            .unwrap()
            .insert(token.clone(), Instant::now() + TOKEN_TTL);
        Some(token)
    }

    pub fn verify_token(&self, token: &str) -> bool {
        let mut tokens = self.tokens.lock().unwrap();
        // drop expired
        tokens.retain(|_, exp| *exp > Instant::now());
        tokens.contains_key(token)
    }

    pub fn create_key(&self, name: &str) -> ApiKey {
        let key = ApiKey {
            id: format!("key_{}", rand_hex(4)),
            secret: format!("fly_sk_{}", rand_hex(24)),
            name: name.to_string(),
            created_at: now_secs(),
            enabled: true,
        };
        self.keys
            .lock()
            .unwrap()
            .insert(key.id.clone(), key.clone());
        self.usage
            .lock()
            .unwrap()
            .entry(key.id.clone())
            .or_default();
        self.save_disk();
        key
    }

    pub fn list_keys(&self) -> Vec<KeyView> {
        let usage = self.usage.lock().unwrap();
        let mut out: Vec<KeyView> = self
            .keys
            .lock()
            .unwrap()
            .values()
            .map(|k| KeyView {
                id: k.id.clone(),
                name: k.name.clone(),
                created_at: k.created_at,
                enabled: k.enabled,
                usage: usage.get(&k.id).cloned().unwrap_or_default(),
            })
            .collect();
        out.sort_by_key(|k| std::cmp::Reverse(k.created_at));
        out
    }

    pub fn set_key_enabled(&self, id: &str, enabled: bool) -> bool {
        let hit = self
            .keys
            .lock()
            .unwrap()
            .get_mut(id)
            .map(|k| k.enabled = enabled)
            .is_some();
        if hit {
            self.save_disk();
        }
        hit
    }

    /// Returns the key id when the secret is valid and enabled.
    pub fn verify_key(&self, secret: &str) -> Option<String> {
        self.keys
            .lock()
            .unwrap()
            .values()
            .find(|k| k.secret == secret && k.enabled)
            .map(|k| k.id.clone())
    }

    /// Auth resolution: admin token takes precedence, then API key secret.
    /// Returns ("admin", None) or ("key", Some(key_id)).
    pub fn authorize(&self, bearer: &str) -> Option<(String, Option<String>)> {
        if self.verify_token(bearer) {
            return Some(("admin".into(), None));
        }
        self.verify_key(bearer).map(|id| ("key".into(), Some(id)))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_usage(
        &self,
        key_id: Option<&str>,
        requests: u64,
        ticks: u64,
        sessions_created: u64,
    ) {
        if let Some(id) = key_id {
            let mut usage = self.usage.lock().unwrap();
            let u = usage.entry(id.to_string()).or_default();
            u.requests += requests;
            u.ticks += ticks;
            u.sessions_created += sessions_created;
            drop(usage);
            self.save_disk();
        }
    }

    pub fn total_usage(&self) -> KeyUsage {
        let usage = self.usage.lock().unwrap();
        let mut t = KeyUsage::default();
        for u in usage.values() {
            t.requests += u.requests;
            t.ticks += u.ticks;
            t.sessions_created += u.sessions_created;
        }
        t
    }
}
