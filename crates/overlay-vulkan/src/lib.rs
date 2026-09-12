#![allow(clippy::missing_safety_doc)]
use ash::vk;
use egui_ash_renderer::{Options, Renderer};
use tuxscaling_input::{InputRoute, PointerViewport, X11Input};
use tuxscaling_overlay::{FrameDiagnostics, OverlayFrame};

pub const CRATE_NAME: &str = "tuxscaling-overlay-vulkan";

#[derive(Clone, Copy)]
pub struct SwapchainInfo {
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub extent: vk::Extent2D,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InputRouteConfig {
    pub event_window: u64,
    pub game_window: u64,
    pub output_extent: [u32; 2],
    pub game_extent: [u32; 2],
    pub viewport: PointerViewport,
}

impl InputRouteConfig {
    pub const fn new(
        event_window: u64,
        game_window: u64,
        output_extent: [u32; 2],
        game_extent: [u32; 2],
        viewport: PointerViewport,
    ) -> Self {
        Self {
            event_window,
            game_window,
            output_extent,
            game_extent,
            viewport,
        }
    }

    pub fn aspect_fit(
        event_window: u64,
        game_window: u64,
        output_extent: [u32; 2],
        game_extent: [u32; 2],
    ) -> Option<Self> {
        if output_extent.contains(&0) || game_extent.contains(&0) {
            return None;
        }
        let input_aspect = game_extent[0] as f32 / game_extent[1] as f32;
        let output_aspect = output_extent[0] as f32 / output_extent[1] as f32;
        let extent = if output_aspect > input_aspect {
            [
                output_extent[0] as f32 * input_aspect / output_aspect,
                output_extent[1] as f32,
            ]
        } else {
            [
                output_extent[0] as f32,
                output_extent[1] as f32 * output_aspect / input_aspect,
            ]
        };
        Some(Self::new(
            event_window,
            game_window,
            output_extent,
            game_extent,
            PointerViewport {
                offset: [
                    (output_extent[0] as f32 - extent[0]) * 0.5,
                    (output_extent[1] as f32 - extent[1]) * 0.5,
                ],
                extent,
            },
        ))
    }

    pub fn with_output_extent(self, output_extent: [u32; 2]) -> Option<Self> {
        Self::aspect_fit(
            self.event_window,
            self.game_window,
            output_extent,
            self.game_extent,
        )
    }

    fn route(self) -> InputRoute {
        InputRoute::new(
            self.event_window,
            self.game_window,
            self.output_extent,
            self.game_extent,
            self.viewport,
        )
    }
}

struct Slot {
    view: vk::ImageView,
    framebuffer: vk::Framebuffer,
    renderer: Option<Renderer>,
    free: Vec<egui::TextureId>,
    pending_textures: Vec<(egui::TextureId, egui::epaint::ImageDelta)>,
}

pub struct OverlayRenderer {
    device: ash::Device,
    info: SwapchainInfo,
    render_pass: vk::RenderPass,
    slots: Vec<Slot>,
    input: Option<X11Input>,
    visible: bool,
    context: egui::Context,
}

pub fn is_srgb_framebuffer(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB | vk::Format::A8B8G8R8_SRGB_PACK32
    )
}

impl OverlayRenderer {
    pub unsafe fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        info: SwapchainInfo,
        images: &[vk::Image],
        input_route: Option<InputRouteConfig>,
    ) -> Result<Self, vk::Result> {
        let mut result = Self {
            device: device.clone(),
            info,
            render_pass: vk::RenderPass::null(),
            slots: Vec::new(),
            input: input_route.and_then(|route| {
                X11Input::connect(route.route())
                    .map_err(|error| eprintln!("TuxScaling input: {error}"))
                    .ok()
            }),
            visible: false,
            context: egui::Context::default(),
        };
        let attachments = [vk::AttachmentDescription::default()
            .format(info.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let colors = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&colors)];
        result.render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(&subpasses),
                None,
            )
        }?;
        for image in images {
            result.slots.push(Slot {
                view: vk::ImageView::null(),
                framebuffer: vk::Framebuffer::null(),
                renderer: None,
                free: Vec::new(),
                pending_textures: Vec::new(),
            });
            let slot = result.slots.last_mut().unwrap();
            slot.view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(*image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(info.format)
                        .subresource_range(tuxscaling_vulkan::color_range()),
                    None,
                )
            }?;
            slot.framebuffer = unsafe {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(result.render_pass)
                        .attachments(&[slot.view])
                        .width(info.extent.width)
                        .height(info.extent.height)
                        .layers(1),
                    None,
                )
            }?;
            slot.renderer = Some(
                Renderer::with_default_allocator(
                    instance,
                    physical,
                    device.clone(),
                    result.render_pass,
                    Options {
                        in_flight_frames: 1,
                        srgb_framebuffer: is_srgb_framebuffer(info.format),
                        ..Default::default()
                    },
                )
                .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?,
            );
        }
        Ok(result)
    }

    pub fn prepare(
        &mut self,
        queue: vk::Queue,
        pool: vk::CommandPool,
        index: usize,
        diagnostics: &mut FrameDiagnostics,
    ) -> Result<OverlayFrame, vk::Result> {
        let slot = self
            .slots
            .get_mut(index)
            .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;
        let renderer = slot
            .renderer
            .as_mut()
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        renderer
            .free_textures(&slot.free)
            .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
        slot.free.clear();
        let input = self
            .input
            .as_mut()
            .map_or_else(Default::default, |input| input.poll());
        if input.toggle_overlay {
            self.visible = !self.visible;
        }
        let frame = tuxscaling_overlay::render_diagnostics(
            &self.context,
            [self.info.extent.width, self.info.extent.height],
            diagnostics,
            &input.events,
            self.visible,
        );
        for slot in &mut self.slots {
            slot.pending_textures
                .extend(frame.textures_delta.set.iter().cloned());
        }
        let slot = &mut self.slots[index];
        if !slot.pending_textures.is_empty() {
            slot.renderer
                .as_mut()
                .unwrap()
                .set_textures(queue, pool, &slot.pending_textures)
                .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
            slot.pending_textures.clear();
        }
        slot.free.clone_from(&frame.textures_delta.free);
        Ok(frame)
    }

    pub unsafe fn record(
        &mut self,
        command: vk::CommandBuffer,
        index: usize,
        frame: &OverlayFrame,
    ) -> Result<(), vk::Result> {
        let slot = self
            .slots
            .get_mut(index)
            .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;
        let info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(slot.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.info.extent,
            });
        unsafe {
            self.device
                .cmd_begin_render_pass(command, &info, vk::SubpassContents::INLINE);
        }
        let result = slot
            .renderer
            .as_mut()
            .unwrap()
            .cmd_draw(
                command,
                self.info.extent,
                frame.pixels_per_point,
                &frame.primitives,
            )
            .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED);
        unsafe {
            self.device.cmd_end_render_pass(command);
        }
        result
    }
}
impl Drop for OverlayRenderer {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            drop(slot.renderer.take());
            unsafe {
                self.device.destroy_framebuffer(slot.framebuffer, None);
                self.device.destroy_image_view(slot.view, None);
            }
        }
        unsafe {
            self.device.destroy_render_pass(self.render_pass, None);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identifies_srgb_swapchain_formats() {
        assert!(is_srgb_framebuffer(vk::Format::B8G8R8A8_SRGB));
        assert!(!is_srgb_framebuffer(vk::Format::R16G16B16A16_SFLOAT));
    }

    #[test]
    fn input_route_config_maps_native_output_to_logical_game_viewport() {
        let route = InputRouteConfig::aspect_fit(0x900, 0x400, [1920, 1200], [1280, 720])
            .expect("valid extents should produce a route");

        assert_eq!(route.event_window, 0x900);
        assert_eq!(route.game_window, 0x400);
        assert_eq!(route.viewport.offset, [0.0, 60.0]);
        assert_eq!(route.viewport.extent, [1920.0, 1080.0]);
        assert_eq!(route.with_output_extent([0, 1200]), None);
    }
}
