use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AcquireError {
    PhysicalBusy,
    NoLogicalSlot,
}

/// Application-visible swapchain keys live in a tagged namespace.  Vulkan
/// never dereferences non-dispatchable handles; the layer resolves this token
/// before every downstream call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LogicalSwapchainHandle(u64);

impl LogicalSwapchainHandle {
    const TAG: u64 = 1 << 63;
    const PAYLOAD: u64 = !Self::TAG;

    pub(crate) const fn from_counter(counter: u64) -> Self {
        Self(Self::TAG | (counter & Self::PAYLOAD))
    }

    pub(crate) const fn from_raw(raw: u64) -> Option<Self> {
        if (raw & Self::TAG != 0) && raw != Self::TAG {
            Some(Self(raw))
        } else {
            None
        }
    }

    pub(crate) const fn raw(self) -> u64 {
        self.0
    }

    pub(crate) const fn is_reserved(raw: u64) -> bool {
        Self::from_raw(raw).is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PresentError {
    UnknownLogicalSlot,
    LogicalSlotIdle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReleaseError {
    DuplicateLogicalSlot,
    LogicalSlotIdle,
    OutOfRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PhysicalImage {
    generation: u64,
    index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReleasePlan {
    generation: u64,
    logical_indices: Vec<u32>,
    physical_indices: Vec<u32>,
}

impl ReleasePlan {
    pub(crate) fn physical_indices(&self) -> &[u32] {
        &self.physical_indices
    }
}

/// Maps application-visible image slots to the images acquired from the
/// current downstream swapchain generation.  It deliberately contains no
/// Vulkan handles, so the ownership policy is deterministic and testable.
#[derive(Debug, Clone)]
pub(crate) struct Mapping {
    generation: u64,
    logical_slots: Vec<Option<PhysicalImage>>,
    reserved_slots: Vec<bool>,
}

impl Mapping {
    pub(crate) fn new(generation: u64, logical_image_count: usize) -> Self {
        Self {
            generation,
            logical_slots: vec![None; logical_image_count],
            reserved_slots: vec![false; logical_image_count],
        }
    }

    pub(crate) const fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn is_idle(&self) -> bool {
        !self.logical_slots.iter().any(Option::is_some)
            && !self.reserved_slots.iter().any(|reserved| *reserved)
    }

    #[cfg(test)]
    pub(crate) fn acquire(&mut self, physical_index: u32) -> Result<u32, AcquireError> {
        let logical_index = self.reserve_slot()?;
        if let Err(error) = self.bind_reserved(logical_index, physical_index) {
            self.cancel_reservation(logical_index);
            return Err(error);
        }
        Ok(logical_index)
    }

    pub(crate) fn reserve_slot(&mut self) -> Result<u32, AcquireError> {
        let Some((index, reserved)) = self
            .reserved_slots
            .iter_mut()
            .enumerate()
            .find(|(index, reserved)| !**reserved && self.logical_slots[*index].is_none())
        else {
            return Err(AcquireError::NoLogicalSlot);
        };
        *reserved = true;
        Ok(index as u32)
    }

    pub(crate) fn bind_reserved(
        &mut self,
        logical_index: u32,
        physical_index: u32,
    ) -> Result<(), AcquireError> {
        if self
            .logical_slots
            .iter()
            .flatten()
            .any(|mapped| mapped.generation == self.generation && mapped.index == physical_index)
        {
            return Err(AcquireError::PhysicalBusy);
        }
        let index = logical_index as usize;
        if !self.reserved_slots.get(index).copied().unwrap_or(false)
            || self.logical_slots.get(index).is_none_or(Option::is_some)
        {
            return Err(AcquireError::NoLogicalSlot);
        }
        self.reserved_slots[index] = false;
        self.logical_slots[index] = Some(PhysicalImage {
            generation: self.generation,
            index: physical_index,
        });
        Ok(())
    }

    pub(crate) fn cancel_reservation(&mut self, logical_index: u32) {
        if let Some(reserved) = self.reserved_slots.get_mut(logical_index as usize) {
            *reserved = false;
        }
    }

    pub(crate) fn present(&mut self, logical_index: u32) -> Result<u32, PresentError> {
        let Some(slot) = self.logical_slots.get_mut(logical_index as usize) else {
            return Err(PresentError::UnknownLogicalSlot);
        };
        let Some(mapped) = slot.take() else {
            return Err(PresentError::LogicalSlotIdle);
        };
        if mapped.generation != self.generation {
            return Err(PresentError::LogicalSlotIdle);
        }
        Ok(mapped.index)
    }

    pub(crate) fn resolve(&self, logical_index: u32) -> Option<u32> {
        self.logical_slots
            .get(logical_index as usize)
            .and_then(|slot| *slot)
            .filter(|mapped| mapped.generation == self.generation)
            .map(|mapped| mapped.index)
    }

    pub(crate) fn plan_release(
        &self,
        logical_indices: &[u32],
    ) -> Result<ReleasePlan, ReleaseError> {
        let mut physical_indices = Vec::with_capacity(logical_indices.len());
        for (position, logical_index) in logical_indices.iter().enumerate() {
            if logical_indices[..position].contains(logical_index) {
                return Err(ReleaseError::DuplicateLogicalSlot);
            }
            let Some(mapped) = self
                .logical_slots
                .get(*logical_index as usize)
                .and_then(|slot| *slot)
                .filter(|mapped| mapped.generation == self.generation)
            else {
                if (*logical_index as usize) >= self.logical_slots.len() {
                    return Err(ReleaseError::OutOfRange);
                }
                return Err(ReleaseError::LogicalSlotIdle);
            };
            physical_indices.push(mapped.index);
        }
        Ok(ReleasePlan {
            generation: self.generation,
            logical_indices: logical_indices.to_vec(),
            physical_indices,
        })
    }

    pub(crate) fn commit_release(&mut self, plan: &ReleasePlan) -> bool {
        if self.generation != plan.generation
            || plan.logical_indices.len() != plan.physical_indices.len()
        {
            return false;
        }
        let still_current = plan.logical_indices.iter().zip(&plan.physical_indices).all(
            |(logical_index, physical_index)| {
                self.logical_slots
                    .get(*logical_index as usize)
                    .and_then(|slot| *slot)
                    == Some(PhysicalImage {
                        generation: self.generation,
                        index: *physical_index,
                    })
            },
        );
        if !still_current {
            return false;
        }
        for logical_index in &plan.logical_indices {
            self.logical_slots[*logical_index as usize] = None;
        }
        true
    }

    pub(crate) fn replace_generation(&mut self, generation: u64) -> bool {
        if !self.is_idle() {
            return false;
        }
        self.generation = generation;
        true
    }

    #[allow(dead_code)] // Used by the preflight path and directly covered here.
    pub(crate) fn direct_fallback(preflight_succeeded: bool, virtualized: bool) -> bool {
        !preflight_succeeded || !virtualized
    }
}

pub(crate) fn rewrite_present_array(
    mapping: &Mapping,
    physical_swapchain: vk::SwapchainKHR,
    logical_index: u32,
) -> Result<(vk::SwapchainKHR, u32), vk::Result> {
    mapping
        .resolve(logical_index)
        .map(|physical_index| (physical_swapchain, physical_index))
        .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)
}

pub(crate) struct OldSwapchain;

impl OldSwapchain {
    pub(crate) const fn translate(
        requested: Option<u64>,
        logical: u64,
        current_physical: u64,
    ) -> Option<u64> {
        match requested {
            Some(handle) if handle == logical => Some(current_physical),
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AcquireError, LogicalSwapchainHandle, Mapping, OldSwapchain, rewrite_present_array,
    };
    use ash::vk;
    use ash::vk::Handle;

    #[test]
    fn assigns_and_releases_logical_slots_for_physical_images() {
        let mut mapping = Mapping::new(7, 3);

        assert_eq!(mapping.acquire(2), Ok(0));
        assert_eq!(mapping.acquire(0), Ok(1));
        assert_eq!(mapping.present(0), Ok(2));
        assert_eq!(mapping.acquire(1), Ok(0));
        assert_eq!(mapping.present(1), Ok(0));
    }

    #[test]
    fn rejects_a_second_mapping_for_an_acquired_physical_image() {
        let mut mapping = Mapping::new(1, 2);

        assert_eq!(mapping.acquire(3), Ok(0));
        assert_eq!(mapping.acquire(3), Err(AcquireError::PhysicalBusy));
    }

    #[test]
    fn invalidates_mappings_only_when_the_generation_is_idle() {
        let mut mapping = Mapping::new(4, 2);
        assert_eq!(mapping.acquire(0), Ok(0));

        assert!(!mapping.replace_generation(5));
        assert_eq!(mapping.present(0), Ok(0));
        assert!(mapping.replace_generation(5));
        assert_eq!(mapping.generation(), 5);
        assert_eq!(mapping.resolve(0), None);
    }

    #[test]
    fn maps_different_logical_and_physical_image_counts() {
        let mut mapping = Mapping::new(9, 3);

        assert_eq!(mapping.acquire(5), Ok(0));
        assert_eq!(mapping.acquire(1), Ok(1));
        assert_eq!(mapping.acquire(3), Ok(2));
        assert_eq!(mapping.present(1), Ok(1));
        assert_eq!(mapping.acquire(0), Ok(1));
    }

    #[test]
    fn reserves_a_logical_slot_before_unequal_count_downstream_acquire() {
        let mut mapping = Mapping::new(1, 1);
        let reservation = mapping.reserve_slot().unwrap();

        assert_eq!(mapping.reserve_slot(), Err(AcquireError::NoLogicalSlot));
        assert_eq!(mapping.bind_reserved(reservation, 7), Ok(()));
        assert_eq!(mapping.present(reservation), Ok(7));
    }

    #[test]
    fn controlled_downstream_double_receives_rewritten_present_array() {
        let mut mapping = Mapping::new(9, 2);
        let logical_index = mapping.acquire(4).unwrap();
        let physical_swapchain = vk::SwapchainKHR::from_raw(33);

        let rewritten = rewrite_present_array(&mapping, physical_swapchain, logical_index);

        assert_eq!(rewritten, Ok((physical_swapchain, 4)));
        assert_eq!(mapping.present(logical_index), Ok(4));
    }

    #[test]
    fn exact_native_confirmation_requires_recreation_before_virtualization() {
        use std::time::{Duration, Instant};
        use tuxscaling_display::{
            Extent, PresentationNegotiation, PresentationState, Rect, SurfaceExtent,
        };

        let target = Rect::new(-1920, 0, 1920, 1080);
        let now = Instant::now();
        let mut negotiation = PresentationNegotiation::direct();
        assert!(negotiation.request_borderless(target, now));
        assert!(negotiation.borderless_requested(now));

        assert!(!negotiation.observe(
            Rect::new(-1920, 0, 1920, 1040),
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1040)),
            now + Duration::from_millis(1),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);
        assert!(!negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1080)),
            now + Duration::from_millis(2),
        ));
        assert!(negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1080)),
            now + Duration::from_millis(3),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);
        assert!(negotiation.output_recreation_ready());
        assert!(negotiation.native_observation_is_current(
            target,
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1080)),
            now + Duration::from_millis(3),
        ));
        assert!(negotiation.output_recreated(
            target,
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1080)),
            now + Duration::from_millis(3),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Virtualized);
    }

    #[test]
    fn persistent_negotiation_fails_open_after_monotonic_deadline() {
        use std::time::{Duration, Instant};
        use tuxscaling_display::{
            Extent, NEGOTIATION_TIMEOUT, NegotiationFailure, PresentationNegotiation,
            PresentationState, Rect, SurfaceExtent,
        };

        let now = Instant::now();
        let target = Rect::new(0, 0, 1920, 1080);
        let mut negotiation = PresentationNegotiation::direct();
        assert!(negotiation.request_borderless(target, now));
        assert!(negotiation.borderless_requested(now));
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);
        assert!(!negotiation.observe(
            Rect::new(0, 0, 1920, 1040),
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1040)),
            now + NEGOTIATION_TIMEOUT + Duration::from_nanos(1),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Failed);
        assert_eq!(
            negotiation.failure(),
            Some(NegotiationFailure::DeadlineExpired)
        );
    }

    #[test]
    fn recreated_authoritative_state_stays_virtualized_on_first_present() {
        use std::time::{Duration, Instant};
        use tuxscaling_display::{
            Extent, PresentationNegotiation, PresentationState, Rect, SurfaceExtent,
        };

        let now = Instant::now();
        let target = Rect::new(0, 0, 1920, 1080);
        let native = SurfaceExtent::fixed(Extent::new(1920, 1080));
        let mut surface = PresentationNegotiation::direct();
        assert!(surface.request_borderless(target, now));
        assert!(surface.borderless_requested(now));
        assert!(!surface.observe(target, true, native, now + Duration::from_secs(1)));
        assert!(surface.observe(target, true, native, now + Duration::from_secs(2)));
        assert!(surface.output_recreated(target, true, native, now + Duration::from_secs(2)));

        let mut swapchain = surface;
        assert_eq!(swapchain.public_state(), PresentationState::Virtualized);
        assert!(!swapchain.observe(target, true, native, now + Duration::from_secs(3)));
        assert_eq!(swapchain.public_state(), PresentationState::Virtualized);
    }

    #[test]
    fn translates_a_logical_old_swapchain_to_its_current_physical_handle() {
        let current_physical = 91;

        assert_eq!(
            OldSwapchain::translate(Some(42), 42, current_physical),
            Some(current_physical)
        );
        assert_eq!(
            OldSwapchain::translate(Some(77), 42, current_physical),
            Some(77)
        );
        assert_eq!(OldSwapchain::translate(None, 42, current_physical), None);
    }

    #[test]
    fn release_plan_preserves_mapping_until_downstream_success() {
        let mut mapping = Mapping::new(7, 3);
        let first = mapping.acquire(4).unwrap();
        let second = mapping.acquire(1).unwrap();
        let plan = mapping.plan_release(&[second, first]).unwrap();

        assert_eq!(plan.physical_indices(), &[1, 4]);
        assert_eq!(mapping.resolve(first), Some(4));
        assert!(mapping.commit_release(&plan));
        assert_eq!(mapping.resolve(first), None);
        assert_eq!(mapping.resolve(second), None);
    }

    #[test]
    fn failed_downstream_release_does_not_commit_the_plan() {
        let mut mapping = Mapping::new(71, 1);
        let logical = mapping.acquire(6).unwrap();
        let plan = mapping.plan_release(&[logical]).unwrap();
        let downstream_result = vk::Result::ERROR_DEVICE_LOST;

        if downstream_result == vk::Result::SUCCESS {
            assert!(mapping.commit_release(&plan));
        }

        assert_eq!(mapping.resolve(logical), Some(6));
    }

    #[test]
    fn release_plan_rejects_duplicate_idle_and_out_of_range_slots() {
        let mut mapping = Mapping::new(8, 2);
        let acquired = mapping.acquire(3).unwrap();

        assert!(mapping.plan_release(&[acquired, acquired]).is_err());
        assert!(mapping.plan_release(&[1]).is_err());
        assert!(mapping.plan_release(&[2]).is_err());
        assert!(mapping.plan_release(&[]).is_ok());
    }

    #[test]
    fn stale_release_plan_cannot_commit_after_generation_change() {
        let mut mapping = Mapping::new(9, 1);
        let logical = mapping.acquire(5).unwrap();
        let plan = mapping.plan_release(&[logical]).unwrap();
        assert_eq!(mapping.present(logical), Ok(5));
        assert!(mapping.replace_generation(10));

        assert!(!mapping.commit_release(&plan));
        assert_eq!(mapping.generation(), 10);
    }

    #[test]
    fn direct_fallback_does_not_install_a_mapping() {
        assert!(Mapping::direct_fallback(true, false));
        assert!(Mapping::direct_fallback(false, true));
        assert!(!Mapping::direct_fallback(true, true));
    }

    #[test]
    fn logical_handle_uses_the_reserved_namespace_and_never_equals_a_physical_handle() {
        let physical = 0x1234;
        let logical = LogicalSwapchainHandle::from_counter(1);

        assert_ne!(logical.raw(), physical);
        assert!(LogicalSwapchainHandle::from_raw(logical.raw()).is_some());
        assert!(LogicalSwapchainHandle::from_raw(physical).is_none());
    }
}
