use ash::vk;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PresentIdRoute {
    generation: u64,
    physical: vk::SwapchainKHR,
}

impl PresentIdRoute {
    #[cfg(test)]
    pub(crate) const fn generation(self) -> u64 {
        self.generation
    }

    pub(crate) const fn physical(self) -> vk::SwapchainKHR {
        self.physical
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PresentIdRouteError {
    Empty,
    NonMonotonic,
    Pending,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PresentIdReservation {
    ids: Vec<u64>,
    generation: u64,
    physical: vk::SwapchainKHR,
}

#[derive(Debug, Default)]
pub(crate) struct PresentIdHistory {
    routes: BTreeMap<u64, PresentIdRoute>,
    last_committed: Option<u64>,
    pending: Option<PresentIdReservation>,
}

impl PresentIdHistory {
    pub(crate) const fn new() -> Self {
        Self {
            routes: BTreeMap::new(),
            last_committed: None,
            pending: None,
        }
    }

    pub(crate) fn stage(
        &mut self,
        generation: u64,
        physical: vk::SwapchainKHR,
        ids: &[u64],
    ) -> Result<PresentIdReservation, PresentIdRouteError> {
        if ids.is_empty() {
            return Err(PresentIdRouteError::Empty);
        }
        if self.pending.is_some() {
            return Err(PresentIdRouteError::Pending);
        }
        let mut previous = self.last_committed;
        for id in ids {
            if previous.is_some_and(|previous| *id <= previous) {
                return Err(PresentIdRouteError::NonMonotonic);
            }
            previous = Some(*id);
        }
        let reservation = PresentIdReservation {
            ids: ids.to_vec(),
            generation,
            physical,
        };
        self.pending = Some(reservation.clone());
        Ok(reservation)
    }

    pub(crate) fn commit(&mut self, reservation: PresentIdReservation) -> bool {
        if self.pending.as_ref() != Some(&reservation) {
            return false;
        }
        for id in &reservation.ids {
            self.routes.insert(
                *id,
                PresentIdRoute {
                    generation: reservation.generation,
                    physical: reservation.physical,
                },
            );
        }
        self.last_committed = reservation.ids.last().copied();
        self.pending = None;
        true
    }

    pub(crate) fn rollback(&mut self, reservation: PresentIdReservation) -> bool {
        if self.pending.as_ref() != Some(&reservation) {
            return false;
        }
        self.pending = None;
        true
    }

    pub(crate) fn resolve(&self, id: u64) -> Option<PresentIdRoute> {
        self.routes.get(&id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::{PresentIdHistory, PresentIdRouteError};
    use ash::vk;
    use ash::vk::Handle;

    #[test]
    fn present_id_ranges_remain_routable_across_three_generations() {
        let mut history = PresentIdHistory::new();
        let first = history
            .stage(0, vk::SwapchainKHR::from_raw(11), &[10, 12])
            .unwrap();
        history.commit(first);
        let second = history
            .stage(1, vk::SwapchainKHR::from_raw(12), &[20])
            .unwrap();
        history.commit(second);
        let third = history
            .stage(2, vk::SwapchainKHR::from_raw(13), &[31, 40])
            .unwrap();
        history.commit(third);

        assert_eq!(history.resolve(10).unwrap().generation(), 0);
        assert_eq!(
            history.resolve(20).unwrap().physical(),
            vk::SwapchainKHR::from_raw(12)
        );
        assert_eq!(history.resolve(40).unwrap().generation(), 2);
    }

    #[test]
    fn gaps_are_allowed_but_non_monotonic_ids_are_rejected() {
        let mut history = PresentIdHistory::new();
        let first = history
            .stage(0, vk::SwapchainKHR::from_raw(21), &[4, 9])
            .unwrap();
        history.commit(first);

        assert_eq!(
            history.stage(1, vk::SwapchainKHR::from_raw(22), &[8]),
            Err(PresentIdRouteError::NonMonotonic)
        );
        assert!(
            history
                .stage(1, vk::SwapchainKHR::from_raw(22), &[10, 100])
                .is_ok()
        );
    }

    #[test]
    fn rollback_does_not_consume_the_present_id_range() {
        let mut history = PresentIdHistory::new();
        let pending = history
            .stage(0, vk::SwapchainKHR::from_raw(31), &[50])
            .unwrap();
        history.rollback(pending);

        assert!(
            history
                .stage(0, vk::SwapchainKHR::from_raw(31), &[50])
                .is_ok()
        );
    }

    #[test]
    fn suboptimal_commit_keeps_old_generation_lookup_available() {
        let mut history = PresentIdHistory::new();
        let old = history
            .stage(0, vk::SwapchainKHR::from_raw(41), &[70])
            .unwrap();
        history.commit(old);
        let replacement = history
            .stage(1, vk::SwapchainKHR::from_raw(42), &[80])
            .unwrap();
        history.commit(replacement);

        assert_eq!(
            history.resolve(70).unwrap().physical(),
            vk::SwapchainKHR::from_raw(41)
        );
        assert_eq!(
            history.resolve(80).unwrap().physical(),
            vk::SwapchainKHR::from_raw(42)
        );
    }

    #[test]
    fn a_second_pending_present_is_rejected_until_the_first_is_committed() {
        let mut history = PresentIdHistory::new();
        let pending = history
            .stage(0, vk::SwapchainKHR::from_raw(51), &[90])
            .unwrap();

        assert_eq!(
            history.stage(0, vk::SwapchainKHR::from_raw(51), &[91]),
            Err(PresentIdRouteError::Pending)
        );
        history.commit(pending);
        assert!(
            history
                .stage(0, vk::SwapchainKHR::from_raw(51), &[91])
                .is_ok()
        );
    }
}
