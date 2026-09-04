use ash::vk;
use egui_ash_renderer::{Options as RendererOptions, Renderer};

#[derive(Clone, Copy)]
pub struct SwapchainInfo {
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

#[derive(Clone, Copy)]
pub struct FrameSubmission {
    pub command_buffer: vk::CommandBuffer,
    pub render_complete: vk::Semaphore,
    pub fence: vk::Fence,
}

struct FrameSlot {
    command_buffer: vk::CommandBuffer,
    render_complete: vk::Semaphore,
    fence: vk::Fence,
}

pub struct OverlaySwapchain {
    info: SwapchainInfo,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    renderer: Renderer,
    context: egui::Context,
    command_pool: vk::CommandPool,
    frames: Vec<FrameSlot>,
    enabled: bool,
}

pub const CRATE_NAME: &str = "tuxscaling-overlay-vulkan";

pub fn is_srgb_framebuffer(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::R8_SRGB
            | vk::Format::R8G8_SRGB
            | vk::Format::R8G8B8_SRGB
            | vk::Format::B8G8R8_SRGB
            | vk::Format::R8G8B8A8_SRGB
            | vk::Format::B8G8R8A8_SRGB
            | vk::Format::A8B8G8R8_SRGB_PACK32
    )
}

impl OverlaySwapchain {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        queue_family_index: u32,
        info: SwapchainInfo,
        images: Vec<vk::Image>,
    ) -> Result<Self, vk::Result> {
        let attachment = vk::AttachmentDescription::default()
            .format(info.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_reference = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_reference));
        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&attachment))
            .subpasses(std::slice::from_ref(&subpass));
        let render_pass = unsafe { device.create_render_pass(&render_pass_info, None) }?;

        let mut image_views = Vec::with_capacity(images.len());
        let mut framebuffers = Vec::with_capacity(images.len());
        for image in &images {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(info.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let view = unsafe { device.create_image_view(&view_info, None) }?;
            let framebuffer_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(std::slice::from_ref(&view))
                .width(info.extent.width)
                .height(info.extent.height)
                .layers(1);
            let framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None) }?;
            image_views.push(view);
            framebuffers.push(framebuffer);
        }

        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }?;
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(images.len() as u32);
        let command_buffers = unsafe { device.allocate_command_buffers(&allocate_info) }?;
        let mut frames = Vec::with_capacity(images.len());
        for command_buffer in command_buffers {
            let render_complete =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
            let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            let fence = unsafe { device.create_fence(&fence_info, None) }?;
            frames.push(FrameSlot {
                command_buffer,
                render_complete,
                fence,
            });
        }

        let renderer = Renderer::with_default_allocator(
            instance,
            physical_device,
            device.clone(),
            render_pass,
            RendererOptions {
                in_flight_frames: images.len().max(1),
                srgb_framebuffer: is_srgb_framebuffer(info.format),
                ..Default::default()
            },
        )
        .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;

        Ok(Self {
            info,
            images,
            image_views,
            render_pass,
            framebuffers,
            renderer,
            context: egui::Context::default(),
            command_pool,
            frames,
            enabled: true,
        })
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn prepare_frame(
        &mut self,
        device: &ash::Device,
        queue: vk::Queue,
        image_index: u32,
    ) -> Result<FrameSubmission, vk::Result> {
        if !self.enabled {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        let image_index = image_index as usize;
        let image = *self
            .images
            .get(image_index)
            .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;
        let framebuffer = *self
            .framebuffers
            .get(image_index)
            .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;
        let frame_slot = self
            .frames
            .get(image_index)
            .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;

        unsafe { device.wait_for_fences(std::slice::from_ref(&frame_slot.fence), true, u64::MAX) }?;
        unsafe { device.reset_fences(std::slice::from_ref(&frame_slot.fence)) }?;
        unsafe {
            device.reset_command_buffer(
                frame_slot.command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )
        }?;

        let frame = tuxscaling_overlay::render_smoke_frame(
            &self.context,
            [self.info.extent.width, self.info.extent.height],
            1.0,
        );
        if !frame.textures_delta.set.is_empty() {
            self.renderer
                .set_textures(queue, self.command_pool, &frame.textures_delta.set)
                .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
        }

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(frame_slot.command_buffer, &begin_info) }?;

        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let to_color = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_READ)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(image)
            .subresource_range(range);
        unsafe {
            device.cmd_pipeline_barrier(
                frame_slot.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_color),
            );
        }

        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.info.extent,
            });
        unsafe {
            device.cmd_begin_render_pass(
                frame_slot.command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
        }
        self.renderer
            .cmd_draw(
                frame_slot.command_buffer,
                self.info.extent,
                frame.pixels_per_point,
                &frame.primitives,
            )
            .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
        unsafe { device.cmd_end_render_pass(frame_slot.command_buffer) };

        let to_present = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(image)
            .subresource_range(range);
        unsafe {
            device.cmd_pipeline_barrier(
                frame_slot.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_present),
            );
            device.end_command_buffer(frame_slot.command_buffer)?;
        }

        Ok(FrameSubmission {
            command_buffer: frame_slot.command_buffer,
            render_complete: frame_slot.render_complete,
            fence: frame_slot.fence,
        })
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn destroy(self, device: &ash::Device) {
        let Self {
            image_views,
            render_pass,
            framebuffers,
            renderer,
            command_pool,
            frames,
            ..
        } = self;
        drop(renderer);
        for frame in frames {
            unsafe {
                device.destroy_fence(frame.fence, None);
                device.destroy_semaphore(frame.render_complete, None);
            }
        }
        unsafe { device.destroy_command_pool(command_pool, None) };
        for framebuffer in framebuffers {
            unsafe { device.destroy_framebuffer(framebuffer, None) };
        }
        for image_view in image_views {
            unsafe { device.destroy_image_view(image_view, None) };
        }
        unsafe { device.destroy_render_pass(render_pass, None) };
    }
}

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::is_srgb_framebuffer;

    #[test]
    fn identifies_srgb_swapchain_formats() {
        assert!(is_srgb_framebuffer(vk::Format::B8G8R8A8_SRGB));
        assert!(!is_srgb_framebuffer(vk::Format::R16G16B16A16_SFLOAT));
    }
}
