//! Session state snapshots — the storage layer behind fork reuse
//! (the "cache hit" of simulation: restoring tick N beats simulating it).

use std::collections::{BTreeMap, HashMap};

pub struct SnapshotStore {
    per_session: HashMap<String, BTreeMap<u64, Vec<u8>>>,
    cap_per_session: usize,
}

impl SnapshotStore {
    pub fn new(cap_per_session: usize) -> Self {
        Self { per_session: HashMap::new(), cap_per_session: cap_per_session.max(1) }
    }

    pub fn put(&mut self, session_id: &str, tick: u64, blob: Vec<u8>) {
        let m = self.per_session.entry(session_id.to_string()).or_default();
        m.insert(tick, blob);
        while m.len() > self.cap_per_session {
            if let Some(oldest) = m.keys().next().copied() {
                m.remove(&oldest);
            } else {
                break;
            }
        }
    }

    pub fn get(&self, session_id: &str, tick: u64) -> Option<&Vec<u8>> {
        self.per_session.get(session_id)?.get(&tick)
    }

    pub fn len(&self, session_id: &str) -> usize {
        self.per_session.get(session_id).map(|m| m.len()).unwrap_or(0)
    }

    pub fn total_len(&self) -> usize {
        self.per_session.values().map(|m| m.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overwrite_same_tick() {
        let mut s = SnapshotStore::new(4);
        s.put("a", 10, vec![1]);
        s.put("a", 10, vec![2]);
        assert_eq!(s.get("a", 10), Some(&vec![2]));
        assert_eq!(s.len("a"), 1);
    }

    #[test]
    fn evicts_oldest_beyond_cap() {
        let mut s = SnapshotStore::new(2);
        s.put("a", 1, vec![1]);
        s.put("a", 2, vec![2]);
        s.put("a", 3, vec![3]);
        assert_eq!(s.get("a", 1), None, "oldest evicted");
        assert_eq!(s.get("a", 3), Some(&vec![3]));
        assert_eq!(s.len("a"), 2);
    }

    #[test]
    fn sessions_isolated() {
        let mut s = SnapshotStore::new(4);
        s.put("a", 1, vec![1]);
        s.put("b", 1, vec![2]);
        assert_eq!(s.get("a", 1), Some(&vec![1]));
        assert_eq!(s.get("b", 1), Some(&vec![2]));
        assert_eq!(s.total_len(), 2);
    }
}
