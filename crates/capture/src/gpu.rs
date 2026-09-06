#![allow(clippy::missing_safety_doc)]
use ash::vk;
use tuxscaling_vulkan::{Image, image_barrier};

pub struct CapturedFrame {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub frame_id: u64,
    pub timestamp: std::time::Duration,
    pub generation: u64,
}
pub fn supported_format(format: vk::Format, space: vk::ColorSpaceKHR) -> bool {
    space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        && matches!(
            format,
            vk::Format::R8G8B8A8_UNORM
                | vk::Format::B8G8R8A8_UNORM
                | vk::Format::R8G8B8A8_SRGB
                | vk::Format::B8G8R8A8_SRGB
                | vk::Format::R16G16B16A16_SFLOAT
        )
}

pub fn requires_scaling(source: vk::Extent2D, destination: vk::Extent2D) -> bool {
    source != destination
}

pub struct Capture {
    pub color: Image,
    pub previous: Image,
    initialized: bool,
}
impl Capture {
    pub unsafe fn new(
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Result<Self, vk::Result> {
        let usage = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED;
        Ok(Self {
            color: unsafe { Image::new(device, memory, extent, format, usage) }?,
            previous: unsafe { Image::new(device, memory, extent, format, usage) }?,
            initialized: false,
        })
    }
    pub unsafe fn record(
        &mut self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        source: vk::Image,
    ) {
        unsafe {
            self.record_from(device, command, source, vk::ImageLayout::PRESENT_SRC_KHR);
        }
    }

    pub unsafe fn record_from(
        &mut self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        source: vk::Image,
        layout: vk::ImageLayout,
    ) {
        let extent = self.color.extent;
        unsafe {
            self.record_scaled_from(
                device,
                command,
                source,
                extent,
                layout,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        }
    }

    pub unsafe fn record_scaled_from(
        &mut self,
        device: &ash::Device,
        command: vk::CommandBuffer,
        source: vk::Image,
        source_extent: vk::Extent2D,
        layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) {
        unsafe {
            if self.initialized {
                image_barrier(
                    device,
                    command,
                    self.color.handle,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
                image_barrier(
                    device,
                    command,
                    self.previous.handle,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                let layers = vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1);
                device.cmd_copy_image(
                    command,
                    self.color.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.previous.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::ImageCopy::default()
                        .src_subresource(layers)
                        .dst_subresource(layers)
                        .extent(vk::Extent3D {
                            width: self.color.extent.width,
                            height: self.color.extent.height,
                            depth: 1,
                        })],
                );
                image_barrier(
                    device,
                    command,
                    self.color.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
                image_barrier(
                    device,
                    command,
                    self.previous.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            } else {
                image_barrier(
                    device,
                    command,
                    self.previous.handle,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                device.cmd_clear_color_image(
                    command,
                    self.previous.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                    &[tuxscaling_vulkan::color_range()],
                );
                image_barrier(
                    device,
                    command,
                    self.previous.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }
            image_barrier(
                device,
                command,
                source,
                layout,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            image_barrier(
                device,
                command,
                self.color.handle,
                if self.initialized {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                } else {
                    vk::ImageLayout::UNDEFINED
                },
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            let layers = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1);
            if requires_scaling(source_extent, self.color.extent) {
                device.cmd_blit_image(
                    command,
                    source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.color.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::ImageBlit::default()
                        .src_subresource(layers)
                        .dst_subresource(layers)
                        .src_offsets([
                            vk::Offset3D::default(),
                            vk::Offset3D {
                                x: source_extent.width as i32,
                                y: source_extent.height as i32,
                                z: 1,
                            },
                        ])
                        .dst_offsets([
                            vk::Offset3D::default(),
                            vk::Offset3D {
                                x: self.color.extent.width as i32,
                                y: self.color.extent.height as i32,
                                z: 1,
                            },
                        ])],
                    vk::Filter::LINEAR,
                );
            } else {
                let region = vk::ImageCopy::default()
                    .src_subresource(layers)
                    .dst_subresource(layers)
                    .extent(vk::Extent3D {
                        width: self.color.extent.width,
                        height: self.color.extent.height,
                        depth: 1,
                    });
                device.cmd_copy_image(
                    command,
                    source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.color.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            }
            image_barrier(
                device,
                command,
                self.color.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            image_barrier(
                device,
                command,
                source,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                final_layout,
            );
        }
        self.initialized = true;
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_when_capture_requires_scaling() {
        let source = vk::Extent2D {
            width: 1280,
            height: 720,
        };

        assert!(!requires_scaling(source, source));
        assert!(requires_scaling(
            source,
            vk::Extent2D {
                width: 960,
                height: 540,
            }
        ));
    }
    #[test]
    fn rejects_hdr_and_accepts_sdr() {
        assert!(supported_format(
            vk::Format::B8G8R8A8_SRGB,
            vk::ColorSpaceKHR::SRGB_NONLINEAR
        ));
        assert!(supported_format(
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ColorSpaceKHR::SRGB_NONLINEAR
        ));
        assert!(!supported_format(
            vk::Format::R8G8B8A8_UNORM,
            vk::ColorSpaceKHR::HDR10_ST2084_EXT
        ));
    }
}
