use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetReason {
    FirstFrame,
    Discontinuity,
    Presentation,
    Reconfigured,
}

#[derive(Debug, Default)]
pub struct History {
    pub generation: u64,
    pub frame_id: u64,
    last: Option<Duration>,
}
impl History {
    pub fn valid(&self, now: Duration) -> bool {
        self.last.is_some_and(|last| {
            now.checked_sub(last)
                .is_some_and(|gap| gap <= Duration::from_millis(250))
        })
    }
    pub fn commit(&mut self, now: Duration) {
        self.frame_id += 1;
        self.last = Some(now);
    }
    pub fn reset(&mut self) {
        self.generation += 1;
        self.last = None;
    }
    pub fn write_index(&self) -> usize {
        (self.frame_id % 2) as usize
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn advances_only_on_commit_and_resets_after_a_gap() {
        let mut h = History::default();
        assert!(!h.valid(Duration::ZERO));
        assert_eq!(h.write_index(), 0);
        h.commit(Duration::ZERO);
        assert_eq!(h.write_index(), 1);
        assert!(h.valid(Duration::from_millis(250)));
        assert!(!h.valid(Duration::from_millis(251)));
        h.reset();
        assert!(!h.valid(Duration::from_millis(20)));
        assert_eq!(h.generation, 1);
    }
}
