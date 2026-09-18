//! Admin auth, API keys, and per-key usage accounting — SQLite backed.
//!
//! Everything lives in `<data_dir>/fly.db` (auto-migrated on startup):
//! - `api_keys(id, secret, name, created_at, enabled)`
//! - `usage(key_id, requests, ticks, sessions)`
//! - admin password config comes from CLI; tokens stay in memory (24h).

use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rusqlite::Connection;

const TOKEN_TTL: Duration = Duration::from_secs(24 * 3600);

#[derive(Debug, Clone, Serialize)]
pub struct ApiKey {
    pub id: String,     // "key_<short>"
    pub secret: String, // "fly_sk_<hex>" — only shown once at creation;
    // list_keys() projects to KeyView which omits it
    pub name: String,
    pub created_at: u64,
    pub enabled: bool,
    /// Rate limits: requests per minute, ticks per minute, lifetime tick budget
    pub rpm_limit: u64, // 0 = unlimited
    pub tpm_limit: u64,   // ticks per minute; 0 = unlimited
    pub tick_budget: u64, // lifetime tick budget; 0 = unlimited
}

#[derive(Debug, Default, Clone, Serialize)]
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
    pub rpm_limit: u64,
    pub tpm_limit: u64,
    pub tick_budget: u64,
}

pub struct AdminStore {
    conn: Mutex<Connection>,
    admin_user: String,
    admin_pass_hash: [u8; 32],
    salt: String,
    tokens: Mutex<HashMap<String, Instant>>,
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
    pub fn new(admin_user: &str, admin_pass: &str, db_path: PathBuf) -> Self {
        let conn = Connection::open(&db_path).expect("open fly.db");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    secret TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS usage (
                    key_id TEXT PRIMARY KEY,
                    requests INTEGER NOT NULL DEFAULT 0,
                    ticks INTEGER NOT NULL DEFAULT 0,
                    sessions_created INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS rate_windows (
                    key_id TEXT NOT NULL,
                    minute INTEGER NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    ticks INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (key_id, minute)
                );
        ",
        )
        .expect("create tables fly.db");
        // idempotent column migration: ignore "duplicate column" errors
        for col in ["rpm_limit", "tpm_limit", "tick_budget"] {
            let sql = format!("ALTER TABLE api_keys ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0");
            if let Err(e) = conn.execute_batch(&sql) {
                let msg = e.to_string();
                if !msg.contains("duplicate column") {
                    panic!("migrate fly.db: {e}");
                }
            }
        }

        let salt = rand_hex(8);
        let admin_pass_hash: [u8; 32] = {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(salt.as_bytes());
            h.update(admin_pass.as_bytes());
            h.finalize().into()
        };
        Self {
            conn: Mutex::new(conn),
            admin_user: admin_user.to_string(),
            admin_pass_hash,
            salt,
            tokens: Mutex::new(HashMap::new()),
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
        tokens.retain(|_, exp| *exp > Instant::now());
        tokens.contains_key(token)
    }

    pub fn create_key(
        &self,
        name: &str,
        rpm_limit: u64,
        tpm_limit: u64,
        tick_budget: u64,
    ) -> ApiKey {
        let key = ApiKey {
            id: format!("key_{}", rand_hex(4)),
            secret: format!("fly_sk_{}", rand_hex(24)),
            name: name.to_string(),
            created_at: now_secs(),
            enabled: true,
            rpm_limit,
            tpm_limit,
            tick_budget,
        };
        self.conn
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO api_keys (id, secret, name, created_at, enabled, rpm_limit, tpm_limit, tick_budget) VALUES (?1, ?2, ?3, ?4, 1, ?5, ?6, ?7)",
                rusqlite::params![key.id, key.secret, key.name, key.created_at as i64, key.rpm_limit as i64, key.tpm_limit as i64, key.tick_budget as i64],
            )
            .ok();
        self.conn
            .lock()
            .unwrap()
            .execute(
                "INSERT OR IGNORE INTO usage (key_id) VALUES (?1)",
                rusqlite::params![key.id],
            )
            .ok();
        key
    }

    pub fn list_keys(&self) -> Vec<KeyView> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT k.id, k.name, k.created_at, k.enabled,
                        COALESCE(u.requests,0), COALESCE(u.ticks,0), COALESCE(u.sessions_created,0),
                        k.rpm_limit, k.tpm_limit, k.tick_budget
                 FROM api_keys k LEFT JOIN usage u ON u.key_id = k.id
                 ORDER BY k.created_at DESC",
            )
            .expect("list query");
        let rows = stmt
            .query_map([], |row| {
                Ok(KeyView {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    created_at: row.get::<_, i64>(2)? as u64,
                    enabled: row.get::<_, i64>(3)? != 0,
                    usage: KeyUsage {
                        requests: row.get::<_, i64>(4)? as u64,
                        ticks: row.get::<_, i64>(5)? as u64,
                        sessions_created: row.get::<_, i64>(6)? as u64,
                    },
                    rpm_limit: row.get::<_, i64>(7)? as u64,
                    tpm_limit: row.get::<_, i64>(8)? as u64,
                    tick_budget: row.get::<_, i64>(9)? as u64,
                })
            })
            .expect("list query map");
        rows.filter_map(|r| r.ok()).collect()
    }

    pub fn delete_key(&self, id: &str) -> bool {
        let conn = self.conn.lock().unwrap();
        let ok = conn
            .execute("DELETE FROM api_keys WHERE id = ?1", rusqlite::params![id])
            .map(|n| n > 0)
            .unwrap_or(false);
        if ok {
            conn.execute("DELETE FROM usage WHERE key_id = ?1", rusqlite::params![id])
                .ok();
        }
        ok
    }

    pub fn set_key_enabled(&self, id: &str, enabled: bool) -> bool {
        self.conn
            .lock()
            .unwrap()
            .execute(
                "UPDATE api_keys SET enabled = ?1 WHERE id = ?2",
                rusqlite::params![enabled as i64, id],
            )
            .map(|n| n > 0)
            .unwrap_or(false)
    }

    /// Returns the key id when the secret is valid and enabled.
    pub fn verify_key(&self, secret: &str) -> Option<String> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id FROM api_keys WHERE secret = ?1 AND enabled = 1",
            rusqlite::params![secret],
            |row| row.get::<_, String>(0),
        )
        .ok()
    }

    /// Auth resolution: admin token takes precedence, then API key secret.
    pub fn authorize(&self, bearer: &str) -> Option<(String, Option<String>)> {
        if self.verify_token(bearer) {
            return Some(("admin".into(), None));
        }
        self.verify_key(bearer).map(|id| ("key".into(), Some(id)))
    }

    /// Enforce per-key limits: RPM/TPM (per-minute sliding window via the
    /// minute-bucket table) and lifetime tick budget. Returns an ApiError
    /// when a limit is exceeded, None when the request may proceed.
    pub fn check_rate_limit(&self, key_id: &str) -> Option<crate::error::ApiError> {
        let conn = self.conn.lock().unwrap();
        // fetch limits + usage + window counters
        let row = conn
            .query_row(
                "SELECT k.rpm_limit, k.tpm_limit, k.tick_budget,
                        COALESCE(u.ticks,0)
                 FROM api_keys k LEFT JOIN usage u ON u.key_id = k.id
                 WHERE k.id = ?1",
                rusqlite::params![key_id],
                |r| {
                    Ok((
                        r.get::<_, i64>(0)? as u64,
                        r.get::<_, i64>(1)? as u64,
                        r.get::<_, i64>(2)? as u64,
                        r.get::<_, i64>(3)? as u64,
                    ))
                },
            )
            .ok()?;
        let (rpm, tpm, budget, total_ticks) = row;

        // minute window counters
        let now_min = (now_secs() / 60) as i64;
        let (req_win, tick_win) = conn
            .query_row(
                "SELECT COALESCE(requests,0), COALESCE(ticks,0) FROM rate_windows WHERE key_id=?1 AND minute=?2",
                rusqlite::params![key_id, now_min],
                |r| Ok((r.get::<_, i64>(0)? as u64, r.get::<_, i64>(1)? as u64)),
            )
            .unwrap_or((0, 0));

        if rpm > 0 && req_win >= rpm {
            return Some(crate::error::ApiError {
                err_type: "rate_limit_error",
                code: "rpm_exceeded",
                message: format!("RPM limit {rpm} exceeded for key {key_id}"),
                param: None,
            });
        }
        if tpm > 0 && tick_win >= tpm {
            return Some(crate::error::ApiError {
                err_type: "rate_limit_error",
                code: "tpm_exceeded",
                message: format!("TPM limit {tpm} exceeded for key {key_id}"),
                param: None,
            });
        }
        if budget > 0 && total_ticks >= budget {
            return Some(crate::error::ApiError {
                err_type: "insufficient_quota",
                code: "tick_budget_exhausted",
                message: format!("lifetime tick budget {budget} exhausted for key {key_id}"),
                param: None,
            });
        }
        None
    }

    pub fn record_usage(
        &self,
        key_id: Option<&str>,
        requests: u64,
        ticks: u64,
        sessions_created: u64,
    ) {
        if let Some(id) = key_id {
            let conn = self.conn.lock().unwrap();
            conn.execute(
                "UPDATE usage SET requests = requests + ?1, ticks = ticks + ?2,
                 sessions_created = sessions_created + ?3 WHERE key_id = ?4",
                rusqlite::params![requests as i64, ticks as i64, sessions_created as i64, id],
            )
            .ok();
            // minute window counters (for RPM/TPM enforcement)
            let now_min = (now_secs() / 60) as i64;
            conn.execute(
                "INSERT INTO rate_windows (key_id, minute, requests, ticks) VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(key_id, minute) DO UPDATE SET
                   requests = requests + ?3, ticks = ticks + ?4",
                rusqlite::params![id, now_min, requests as i64, ticks as i64],
            )
            .ok();
        }
    }

    pub fn total_usage(&self) -> KeyUsage {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT COALESCE(SUM(requests),0), COALESCE(SUM(ticks),0), COALESCE(SUM(sessions_created),0) FROM usage",
            [],
            |row| {
                Ok(KeyUsage {
                    requests: row.get::<_, i64>(0)? as u64,
                    ticks: row.get::<_, i64>(1)? as u64,
                    sessions_created: row.get::<_, i64>(2)? as u64,
                })
            },
        )
        .unwrap_or_default()
    }
}
