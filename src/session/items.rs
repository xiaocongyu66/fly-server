//! Experiment log per session (OpenAI conversations.items semantics).

use crate::types::{ListItemsQuery, ListItemsResponse, SessionItem};

pub struct ItemLog {
    items: Vec<SessionItem>,
    seq: u64,
}

fn unix_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

impl ItemLog {
    pub fn new() -> Self {
        Self { items: Vec::new(), seq: 0 }
    }

    pub fn append(&mut self, session_id: &str, kind: &str, tick: u64, body: serde_json::Value) -> SessionItem {
        self.seq += 1;
        let item = SessionItem {
            id: format!("item_{}_{}", unix_nanos(), self.seq),
            object: "session.item".into(),
            session_id: session_id.to_string(),
            created_at: (unix_nanos() / 1_000_000_000) as u64,
            tick,
            kind: kind.to_string(),
            body,
        };
        self.items.push(item.clone());
        item
    }

    pub fn list(&self, q: &ListItemsQuery) -> ListItemsResponse {
        let limit = q.limit.unwrap_or(20).clamp(1, 100) as usize;
        let desc = q.order.as_deref().unwrap_or("desc") != "asc";
        let cursor = q.after.as_ref().and_then(|a| self.items.iter().position(|i| i.id == *a));
        let invalid_cursor = q.after.is_some() && cursor.is_none();

        // desc: newest first. cursor p means "items before p, reversed".
        // asc: oldest first. cursor p means "items after p, forward".
        let slice: Vec<SessionItem> = if invalid_cursor {
            Vec::new()
        } else {
            match (desc, cursor) {
                (true, Some(p)) => self.items[..p].iter().rev().take(limit).cloned().collect(),
                (true, None) => self.items.iter().rev().take(limit).cloned().collect(),
                (false, Some(p)) => self.items[p + 1..].iter().take(limit).cloned().collect(),
                (false, None) => self.items.iter().take(limit).cloned().collect(),
            }
        };
        let has_more = match (desc, cursor) {
            (true, Some(p)) => p > limit,
            (true, None) => self.items.len() > limit,
            (false, Some(p)) => self.items.len() > p + 1 + limit,
            (false, None) => self.items.len() > limit,
        };
        ListItemsResponse {
            object: "list".into(),
            first_id: slice.first().map(|i| i.id.clone()),
            last_id: slice.last().map(|i| i.id.clone()),
            data: slice,
            has_more,
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl Default for ItemLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log10() -> ItemLog {
        let mut log = ItemLog::new();
        for i in 0..10 {
            log.append("sess_x", "step", i as u64, serde_json::json!({"i": i}));
        }
        log
    }

    #[test]
    fn default_desc_order() {
        let r = log10().list(&ListItemsQuery::default());
        assert_eq!(r.data.len(), 10);
        assert!(r.data[0].tick >= r.data[1].tick, "desc order");
    }

    #[test]
    fn after_cursor_paginates() {
        let log = log10();
        let q1 = ListItemsQuery { limit: Some(3), ..Default::default() };
        let first = log.list(&q1);
        assert_eq!(first.data.len(), 3);
        assert_eq!(first.data[0].tick, 9, "desc newest first");
        assert!(first.has_more);

        let q2 = ListItemsQuery { limit: Some(3), after: first.last_id.clone(), order: Some("desc".into()) };
        let second = log.list(&q2);
        assert_eq!(second.data.len(), 3);
        assert_eq!(second.data[0].tick, 6, "next page continues");
        assert_ne!(first.data[0].id, second.data[0].id);

        // exhausted page: cursor at oldest item -> empty
        let all = log.list(&ListItemsQuery::default());
        let q3 = ListItemsQuery { limit: Some(3), after: all.last_id.clone(), order: Some("desc".into()) };
        assert_eq!(log.list(&q3).data.len(), 0);
    }

    #[test]
    fn asc_order() {
        let r = log10().list(&ListItemsQuery { order: Some("asc".into()), ..Default::default() });
        assert!(r.data[0].tick <= r.data[1].tick, "asc order");
    }

    #[test]
    fn limit_clamped() {
        let r = log10().list(&ListItemsQuery { limit: Some(500), ..Default::default() });
        assert_eq!(r.data.len(), 10);
        let r2 = log10().list(&ListItemsQuery { limit: Some(3), ..Default::default() });
        assert_eq!(r2.data.len(), 3);
    }
}
