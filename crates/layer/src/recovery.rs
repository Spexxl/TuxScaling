use ash::vk;

use tuxscaling_display::PresentationState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PhysicalGeneration {
    id: u64,
    handle: vk::SwapchainKHR,
    extent: vk::Extent2D,
    image_count: usize,
}

impl PhysicalGeneration {
    pub(crate) const fn new(
        id: u64,
        handle: vk::SwapchainKHR,
        extent: vk::Extent2D,
        image_count: usize,
    ) -> Self {
        Self {
            id,
            handle,
            extent,
            image_count,
        }
    }

    pub(crate) const fn id(self) -> u64 {
        self.id
    }

    pub(crate) const fn handle(self) -> vk::SwapchainKHR {
        self.handle
    }

    pub(crate) const fn extent(self) -> vk::Extent2D {
        self.extent
    }

    pub(crate) const fn image_count(self) -> usize {
        self.image_count
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrameBinding {
    logical_index: u32,
    physical_index: u32,
}

impl FrameBinding {
    pub(crate) const fn new(logical_index: u32, physical_index: u32) -> Self {
        Self {
            logical_index,
            physical_index,
        }
    }

    pub(crate) const fn logical_index(self) -> u32 {
        self.logical_index
    }

    pub(crate) const fn physical_index(self) -> u32 {
        self.physical_index
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PresentationPath {
    SpatialBypass,
    Temporal,
    Direct,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LogicalSnapshot {
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    game_extent: vk::Extent2D,
}

impl LogicalSnapshot {
    pub(crate) fn images(&self) -> &[vk::Image] {
        &self.images
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LogicalSwapchainContract {
    logical: LogicalSnapshot,
    physical: PhysicalGeneration,
    state: PresentationState,
}

impl LogicalSwapchainContract {
    pub(crate) fn new(
        handle: vk::SwapchainKHR,
        images: Vec<vk::Image>,
        game_extent: vk::Extent2D,
        physical: PhysicalGeneration,
        state: PresentationState,
    ) -> Result<Self, &'static str> {
        if handle == vk::SwapchainKHR::null()
            || images.is_empty()
            || game_extent.width == 0
            || game_extent.height == 0
            || physical.handle() == vk::SwapchainKHR::null()
            || physical.extent().width == 0
            || physical.extent().height == 0
            || physical.image_count() == 0
        {
            return Err("invalid swapchain generation contract");
        }
        Ok(Self {
            logical: LogicalSnapshot {
                handle,
                images,
                game_extent,
            },
            physical,
            state,
        })
    }

    pub(crate) const fn handle(&self) -> vk::SwapchainKHR {
        self.logical.handle
    }

    pub(crate) fn logical_images(&self) -> &[vk::Image] {
        &self.logical.images
    }

    pub(crate) const fn game_extent(&self) -> vk::Extent2D {
        self.logical.game_extent
    }

    pub(crate) const fn generation(&self) -> PhysicalGeneration {
        self.physical
    }

    pub(crate) fn logical_snapshot(&self) -> LogicalSnapshot {
        self.logical.clone()
    }

    pub(crate) fn replace_generation(
        &mut self,
        physical: PhysicalGeneration,
        state: PresentationState,
    ) -> Result<(), &'static str> {
        if physical.id() <= self.physical.id()
            || physical.handle() == vk::SwapchainKHR::null()
            || physical.extent().width == 0
            || physical.extent().height == 0
            || physical.image_count() == 0
        {
            return Err("physical generation is not publishable");
        }
        self.physical = physical;
        self.state = state;
        Ok(())
    }

    pub(crate) fn set_state(&mut self, state: PresentationState) {
        self.state = state;
    }

    pub(crate) fn bind(&self, binding: FrameBinding) -> Result<FrameBinding, &'static str> {
        if binding.logical_index() as usize >= self.logical.images.len()
            || binding.physical_index() as usize >= self.physical.image_count()
        {
            return Err("frame index is outside the active generation");
        }
        Ok(binding)
    }

    pub(crate) const fn presentation_path(&self) -> PresentationPath {
        match self.state {
            PresentationState::Negotiating => PresentationPath::SpatialBypass,
            PresentationState::Virtualized => PresentationPath::Temporal,
            PresentationState::Direct | PresentationState::Failed => PresentationPath::Direct,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LeaseCleanup {
    acquired: bool,
    restored: bool,
    restore_count: u32,
}

impl LeaseCleanup {
    pub(crate) const fn acquired() -> Self {
        Self {
            acquired: true,
            restored: false,
            restore_count: 0,
        }
    }

    pub(crate) fn restore_once(&mut self) -> bool {
        if !self.acquired || self.restored {
            return false;
        }
        self.restored = true;
        self.restore_count += 1;
        true
    }

    pub(crate) const fn restore_count(self) -> u32 {
        self.restore_count
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FrameBinding, LeaseCleanup, LogicalSwapchainContract, PhysicalGeneration, PresentationPath,
    };
    use ash::{vk, vk::Handle};
    use tuxscaling_display::PresentationState;

    fn extent(width: u32, height: u32) -> vk::Extent2D {
        vk::Extent2D { width, height }
    }

    fn generation(
        id: u64,
        handle: u64,
        width: u32,
        height: u32,
        image_count: usize,
    ) -> PhysicalGeneration {
        PhysicalGeneration::new(
            id,
            vk::SwapchainKHR::from_raw(handle),
            extent(width, height),
            image_count,
        )
    }

    #[test]
    fn initial_eligible_creation_exposes_logical_identity_while_negotiating() {
        let logical_handle = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0001);
        let logical_images = vec![vk::Image::from_raw(11), vk::Image::from_raw(12)];
        let contract = LogicalSwapchainContract::new(
            logical_handle,
            logical_images.clone(),
            extent(1280, 720),
            generation(0, 21, 1280, 720, 2),
            PresentationState::Negotiating,
        )
        .unwrap();

        assert_eq!(contract.handle(), logical_handle);
        assert_eq!(contract.logical_images(), logical_images.as_slice());
        assert_eq!(contract.game_extent(), extent(1280, 720));
        assert_eq!(
            contract.presentation_path(),
            PresentationPath::SpatialBypass
        );
    }

    #[test]
    fn logical_resolution_survives_native_observations() {
        let mut contract = LogicalSwapchainContract::new(
            vk::SwapchainKHR::from_raw(0x8000_0000_0000_0002),
            vec![vk::Image::from_raw(31), vk::Image::from_raw(32)],
            extent(1280, 720),
            generation(0, 41, 1280, 720, 2),
            PresentationState::Negotiating,
        )
        .unwrap();

        contract
            .replace_generation(
                generation(1, 42, 3440, 1408, 3),
                PresentationState::Negotiating,
            )
            .unwrap();
        contract
            .replace_generation(
                generation(2, 43, 3440, 1440, 3),
                PresentationState::Virtualized,
            )
            .unwrap();

        assert_eq!(contract.game_extent(), extent(1280, 720));
        assert_eq!(contract.generation().extent(), extent(3440, 1440));
    }

    #[test]
    fn physical_replacement_preserves_logical_handle_images_extent_and_indices() {
        let logical_handle = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0003);
        let logical_images = vec![vk::Image::from_raw(51), vk::Image::from_raw(52)];
        let mut contract = LogicalSwapchainContract::new(
            logical_handle,
            logical_images.clone(),
            extent(1280, 720),
            generation(0, 61, 1280, 720, 3),
            PresentationState::Negotiating,
        )
        .unwrap();
        let before = contract.logical_snapshot();

        contract
            .replace_generation(
                generation(1, 62, 3440, 1440, 3),
                PresentationState::Virtualized,
            )
            .unwrap();

        assert_eq!(contract.logical_snapshot(), before);
        assert_eq!(contract.handle(), logical_handle);
        assert_eq!(contract.logical_images(), logical_images.as_slice());
        assert_eq!(
            contract.bind(FrameBinding::new(1, 2)),
            Ok(FrameBinding::new(1, 2))
        );
    }

    #[test]
    fn unequal_logical_and_physical_counts_bind_during_processing_and_present() {
        let contract = LogicalSwapchainContract::new(
            vk::SwapchainKHR::from_raw(0x8000_0000_0000_0004),
            vec![vk::Image::from_raw(71), vk::Image::from_raw(72)],
            extent(1280, 720),
            generation(0, 81, 3440, 1440, 3),
            PresentationState::Virtualized,
        )
        .unwrap();

        assert_eq!(
            contract.bind(FrameBinding::new(1, 2)),
            Ok(FrameBinding::new(1, 2))
        );
        assert_eq!(contract.presentation_path(), PresentationPath::Temporal);
    }

    #[test]
    fn every_failure_path_restores_a_lease_exactly_once() {
        let mut cleanup = LeaseCleanup::acquired();

        assert!(cleanup.restore_once());
        assert!(!cleanup.restore_once());
        assert_eq!(cleanup.restore_count(), 1);
    }
}
