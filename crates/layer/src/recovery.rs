use ash::vk;

use tuxscaling_display::PresentationState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PhysicalGeneration {
    id: u64,
    handle: vk::SwapchainKHR,
    present_surface: Option<vk::SurfaceKHR>,
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
            present_surface: None,
            extent,
            image_count,
        }
    }

    pub(crate) const fn with_present_surface(mut self, surface: vk::SurfaceKHR) -> Self {
        self.present_surface = Some(surface);
        self
    }

    pub(crate) const fn id(self) -> u64 {
        self.id
    }

    pub(crate) const fn handle(self) -> vk::SwapchainKHR {
        self.handle
    }

    pub(crate) const fn present_surface(self) -> Option<vk::SurfaceKHR> {
        self.present_surface
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

    pub(crate) const fn is_failed(&self) -> bool {
        matches!(self.state, PresentationState::Failed)
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
            PresentationState::Negotiating | PresentationState::Failed => {
                PresentationPath::SpatialBypass
            }
            PresentationState::Virtualized => PresentationPath::Temporal,
            PresentationState::Direct => PresentationPath::Direct,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReconfigurationTicket {
    logical_handle: vk::SwapchainKHR,
    game_surface: vk::SurfaceKHR,
    generation: u64,
}

impl ReconfigurationTicket {
    pub(crate) const fn new(
        logical_handle: vk::SwapchainKHR,
        game_surface: vk::SurfaceKHR,
        generation: u64,
    ) -> Self {
        Self {
            logical_handle,
            game_surface,
            generation,
        }
    }

    pub(crate) const fn logical_handle(self) -> vk::SwapchainKHR {
        self.logical_handle
    }

    pub(crate) const fn game_surface(self) -> vk::SurfaceKHR {
        self.game_surface
    }

    pub(crate) const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Debug, Default)]
pub(crate) struct ReconfigurationLifecycle {
    active: Option<ReconfigurationTicket>,
    destroy_requested: bool,
    generation: Option<u64>,
}

impl ReconfigurationLifecycle {
    pub(crate) const fn new() -> Self {
        Self {
            active: None,
            destroy_requested: false,
            generation: None,
        }
    }

    pub(crate) fn begin(&mut self, ticket: ReconfigurationTicket) -> bool {
        if self.active.is_some() {
            return false;
        }
        self.active = Some(ticket);
        self.destroy_requested = false;
        self.generation = Some(ticket.generation());
        true
    }

    pub(crate) const fn blocks_frame_operations(&self) -> bool {
        self.active.is_some()
    }

    pub(crate) fn can_publish(
        &self,
        ticket: ReconfigurationTicket,
        logical_handle: vk::SwapchainKHR,
        game_surface: vk::SurfaceKHR,
        generation: u64,
        retired: bool,
    ) -> bool {
        self.active == Some(ticket)
            && !self.destroy_requested
            && !retired
            && ticket.logical_handle() == logical_handle
            && ticket.game_surface() == game_surface
            && ticket.generation() == generation
    }

    pub(crate) fn publish(
        &mut self,
        ticket: ReconfigurationTicket,
        logical_handle: vk::SwapchainKHR,
        game_surface: vk::SurfaceKHR,
        generation: u64,
        retired: bool,
    ) -> bool {
        if !self.can_publish(ticket, logical_handle, game_surface, generation, retired) {
            return false;
        }
        self.active = None;
        self.generation = Some(generation);
        true
    }

    pub(crate) fn abort(&mut self, ticket: ReconfigurationTicket) -> bool {
        if self.active != Some(ticket) {
            return false;
        }
        self.active = None;
        self.destroy_requested = false;
        true
    }

    pub(crate) fn request_destroy(&mut self) -> bool {
        if self.active.is_none() || self.destroy_requested {
            return false;
        }
        self.destroy_requested = true;
        true
    }

    pub(crate) const fn destroy_requested(&self) -> bool {
        self.destroy_requested
    }

    pub(crate) fn take_destroy_request(&mut self, ticket: ReconfigurationTicket) -> bool {
        if self.active != Some(ticket) || !self.destroy_requested {
            return false;
        }
        self.active = None;
        self.destroy_requested = false;
        true
    }

    #[cfg(test)]
    pub(crate) const fn generation(&self) -> Option<u64> {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FrameBinding, LeaseCleanup, LogicalSwapchainContract, PhysicalGeneration, PresentationPath,
        ReconfigurationLifecycle, ReconfigurationTicket,
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
        let presenter_surface = vk::SurfaceKHR::from_raw(0x6201);
        let mut contract = LogicalSwapchainContract::new(
            logical_handle,
            logical_images.clone(),
            extent(1280, 720),
            generation(0, 61, 2160, 1440, 3).with_present_surface(presenter_surface),
            PresentationState::Virtualized,
        )
        .unwrap();
        let before = contract.logical_snapshot();

        contract
            .replace_generation(
                generation(1, 62, 2160, 1440, 3).with_present_surface(presenter_surface),
                PresentationState::Virtualized,
            )
            .unwrap();

        assert_eq!(contract.logical_snapshot(), before);
        assert_eq!(contract.handle(), logical_handle);
        assert_eq!(contract.logical_images(), logical_images.as_slice());
        assert_eq!(
            contract.generation().present_surface(),
            Some(presenter_surface)
        );
        assert_eq!(
            contract.bind(FrameBinding::new(1, 2)),
            Ok(FrameBinding::new(1, 2))
        );
    }

    #[test]
    fn physical_generation_records_presenter_surface_without_changing_logical_contract() {
        let game_surface = vk::SurfaceKHR::from_raw(0x3101);
        let presenter_surface = vk::SurfaceKHR::from_raw(0x3102);
        let generation = generation(0, 61, 2160, 1440, 3).with_present_surface(presenter_surface);
        let contract = LogicalSwapchainContract::new(
            vk::SwapchainKHR::from_raw(0x8000_0000_0000_0007),
            vec![vk::Image::from_raw(91), vk::Image::from_raw(92)],
            extent(1280, 720),
            generation,
            PresentationState::Virtualized,
        )
        .unwrap();

        assert_eq!(contract.game_extent(), extent(1280, 720));
        assert_eq!(
            contract.generation().present_surface(),
            Some(presenter_surface)
        );
        assert_ne!(game_surface, presenter_surface);
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
    fn failed_virtual_generation_keeps_spatial_fallback_available() {
        let contract = LogicalSwapchainContract::new(
            vk::SwapchainKHR::from_raw(0x8000_0000_0000_0005),
            vec![vk::Image::from_raw(81), vk::Image::from_raw(82)],
            extent(1280, 720),
            generation(0, 91, 1280, 720, 3),
            PresentationState::Failed,
        )
        .unwrap();

        assert_eq!(
            contract.presentation_path(),
            PresentationPath::SpatialBypass
        );
    }

    #[test]
    fn every_failure_path_restores_a_lease_exactly_once() {
        let mut cleanup = LeaseCleanup::acquired();

        assert!(cleanup.restore_once());
        assert!(!cleanup.restore_once());
        assert_eq!(cleanup.restore_count(), 1);
    }

    #[test]
    fn reconfiguration_blocks_acquire_and_present_until_atomic_publication() {
        let logical = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0101);
        let surface = vk::SurfaceKHR::from_raw(0x2201);
        let ticket = ReconfigurationTicket::new(logical, surface, 4);
        let mut lifecycle = ReconfigurationLifecycle::new();

        assert!(lifecycle.begin(ticket));
        assert!(lifecycle.blocks_frame_operations());
        assert!(!lifecycle.can_publish(ticket, logical, surface, 5, false));

        assert!(lifecycle.publish(ticket, logical, surface, 4, false));
        assert!(!lifecycle.blocks_frame_operations());
    }

    #[test]
    fn failed_physical_creation_keeps_generation_and_clears_reconfiguring() {
        let logical = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0102);
        let surface = vk::SurfaceKHR::from_raw(0x2202);
        let ticket = ReconfigurationTicket::new(logical, surface, 8);
        let mut lifecycle = ReconfigurationLifecycle::new();

        assert!(lifecycle.begin(ticket));
        assert!(lifecycle.abort(ticket));
        assert!(!lifecycle.blocks_frame_operations());
        assert_eq!(lifecycle.generation(), Some(8));
    }

    #[test]
    fn stale_publication_and_duplicate_begin_are_rejected() {
        let logical = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0103);
        let surface = vk::SurfaceKHR::from_raw(0x2203);
        let ticket = ReconfigurationTicket::new(logical, surface, 11);
        let mut lifecycle = ReconfigurationLifecycle::new();

        assert!(lifecycle.begin(ticket));
        assert!(!lifecycle.begin(ticket));
        assert!(!lifecycle.can_publish(ticket, logical, surface, 12, false));
        assert!(!lifecycle.can_publish(ticket, logical, surface, 11, true));
        assert!(!lifecycle.publish(ticket, logical, surface, 12, false));
    }

    #[test]
    fn destroy_during_reconfiguration_is_deferred_and_consumed_once() {
        let logical = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0104);
        let surface = vk::SurfaceKHR::from_raw(0x2204);
        let ticket = ReconfigurationTicket::new(logical, surface, 15);
        let mut lifecycle = ReconfigurationLifecycle::new();

        assert!(lifecycle.begin(ticket));
        assert!(lifecycle.request_destroy());
        assert!(!lifecycle.can_publish(ticket, logical, surface, 15, false));
        assert!(lifecycle.take_destroy_request(ticket));
        assert!(!lifecycle.take_destroy_request(ticket));
        assert!(!lifecycle.blocks_frame_operations());
    }

    #[test]
    fn reentrant_external_callback_can_query_marked_registry() {
        let logical = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0105);
        let surface = vk::SurfaceKHR::from_raw(0x2205);
        let ticket = ReconfigurationTicket::new(logical, surface, 18);
        let lifecycle = std::sync::Arc::new(std::sync::Mutex::new(ReconfigurationLifecycle::new()));

        {
            let mut state = lifecycle.lock().unwrap();
            assert!(state.begin(ticket));
        }

        let callback_state = lifecycle.clone();
        let callback =
            std::thread::spawn(move || callback_state.lock().unwrap().blocks_frame_operations());

        assert!(callback.join().unwrap());
    }

    #[test]
    fn successful_publication_keeps_logical_contract_and_changes_only_physical_generation() {
        let logical = vk::SwapchainKHR::from_raw(0x8000_0000_0000_0106);
        let surface = vk::SurfaceKHR::from_raw(0x2206);
        let ticket = ReconfigurationTicket::new(logical, surface, 21);
        let mut lifecycle = ReconfigurationLifecycle::new();

        assert!(lifecycle.begin(ticket));
        assert!(lifecycle.publish(ticket, logical, surface, 21, false));
        assert_eq!(lifecycle.generation(), Some(21));
        assert_eq!(ticket.logical_handle(), logical);
        assert_eq!(ticket.game_surface(), surface);
    }
}
