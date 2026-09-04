#![allow(clippy::missing_safety_doc)]
use ash::vk;
use egui_ash_renderer::{Options, Renderer};
use tuxscaling_overlay::{FrameDiagnostics, OverlayFrame};

pub const CRATE_NAME: &str = "tuxscaling-overlay-vulkan";

#[derive(Clone, Copy)]
pub struct SwapchainInfo {
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

struct Slot {
    view: vk::ImageView,
    framebuffer: vk::Framebuffer,
    renderer: Option<Renderer>,
    context: egui::Context,
    free: Vec<egui::TextureId>,
}

pub struct OverlayRenderer {
    device: ash::Device,
    info: SwapchainInfo,
    render_pass: vk::RenderPass,
    slots: Vec<Slot>,
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
    ) -> Result<Self, vk::Result> {
        let mut result = Self {
            device: device.clone(),
            info,
            render_pass: vk::RenderPass::null(),
            slots: Vec::new(),
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
                context: egui::Context::default(),
                free: Vec::new(),
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
        diagnostics: &FrameDiagnostics,
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
        let frame = tuxscaling_overlay::render_diagnostics(
            &slot.context,
            [self.info.extent.width, self.info.extent.height],
            diagnostics,
        );
        if !frame.textures_delta.set.is_empty() {
            renderer
                .set_textures(queue, pool, &frame.textures_delta.set)
                .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
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
}
