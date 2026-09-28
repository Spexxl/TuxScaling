use super::SwapchainRuntime;
use ash::vk;
use tuxscaling_config::{ProtectionMode, Upscaler};
use tuxscaling_overlay::OverlayFrame;
use tuxscaling_temporal::GuidanceView;
use tuxscaling_upscaler::protection::{
    MAX_PROTECTION_REGIONS, ProtectionFrame, ProtectionOutputs, ProtectionRenderer,
    ProtectionSettings,
};
use tuxscaling_upscaler::{BackendImage, content_viewport};

impl SwapchainRuntime {
    pub(super) fn queue_protection_requests(&mut self, frame: &OverlayFrame) {
        if let Some(mode) = frame.requested_protection_mode {
            self.temporal.request_protection_mode(mode);
            self.requested_config.protection_mode = mode;
            self.diagnostics.requested_protection_mode = Some(mode);
        }
        if let Some(regions) = &frame.requested_protection_regions
            && regions.len() <= MAX_PROTECTION_REGIONS
        {
            self.temporal.request_protection_regions(regions.clone());
            self.requested_config.protection_regions = regions.clone();
            self.diagnostics.requested_protection_regions = Some(regions.clone());
        }
    }

    pub(super) unsafe fn record_protection_frame(
        &mut self,
        command: vk::CommandBuffer,
        slot: usize,
        guidance: Option<GuidanceView>,
        valid: bool,
    ) -> Option<ProtectionOutputs> {
        if self.temporal.active_upscaler != Upscaler::Fsr314
            || self.temporal.config.protection_mode == ProtectionMode::Disabled
        {
            return None;
        }
        let inputs = self.temporal.upscaler.as_ref()?.protection_inputs(slot)?;
        let capture = self.temporal.capture.as_ref()?;
        let settings = match ProtectionSettings::new(
            self.temporal.config.protection_mode,
            &self.temporal.config.protection_regions,
        ) {
            Ok(settings) => settings,
            Err(error) => {
                self.diagnostics.state = format!("Protection unavailable: {error}");
                return None;
            }
        };
        if self.temporal.protection.is_none() {
            let memory = unsafe {
                self.instance
                    .get_physical_device_memory_properties(self.physical)
            };
            match unsafe {
                ProtectionRenderer::new(
                    &self.device,
                    &memory,
                    self.temporal.resolution.game_extent,
                    self.temporal.resolution.output_extent,
                    self.info.format,
                    self.output_images.len(),
                )
            } {
                Ok(renderer) => self.temporal.protection = Some(renderer),
                Err(error) => {
                    self.diagnostics.state = format!("Protection unavailable: {error}");
                    return None;
                }
            }
        }
        let source = BackendImage {
            image: capture.source.color.handle,
            view: capture.source.color.view,
            format: capture.source.color.format,
            extent: capture.source.color.extent,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let output = BackendImage {
            image: self.output_images[slot],
            view: self.output_views[slot],
            format: self.info.format,
            extent: self.temporal.resolution.output_extent,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };
        let reset = !valid || guidance.is_some_and(|view| view.requires_history_reset);
        let frame = ProtectionFrame {
            command,
            slot,
            frame_id: guidance.map_or(self.temporal.history.frame_id.saturating_add(1), |view| {
                view.motion.metadata.frame_id
            }),
            source,
            output,
            guidance: inputs,
            viewport: content_viewport(source.extent, output.extent),
            settings,
            elapsed: self.temporal.pending_timing.validated,
            reset,
        };
        match unsafe {
            self.temporal
                .protection
                .as_mut()
                .expect("initialized above")
                .record(frame)
        } {
            Ok(outputs) => Some(outputs),
            Err(error) => {
                self.diagnostics.state = format!("Protection unavailable: {error}");
                None
            }
        }
    }
}
