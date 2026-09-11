#![allow(clippy::missing_safety_doc)]
use ash::vk;

pub fn memory_type(
    p: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32, vk::Result> {
    (0..p.memory_type_count)
        .find(|&i| {
            bits & (1 << i) != 0 && p.memory_types[i as usize].property_flags.contains(flags)
        })
        .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ImageCreateOptions<'a> {
    pub image_flags: vk::ImageCreateFlags,
    pub view_formats: &'a [vk::Format],
    pub queue_family_indices: &'a [u32],
}

pub struct Image {
    device: ash::Device,
    pub handle: vk::Image,
    pub view: vk::ImageView,
    memory: vk::DeviceMemory,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
}
impl Image {
    pub unsafe fn new(
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self, vk::Result> {
        unsafe {
            Self::with_options(
                device,
                memory,
                extent,
                format,
                usage,
                &ImageCreateOptions::default(),
            )
        }
    }

    pub unsafe fn with_sharing(
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        queue_families: &[u32],
    ) -> Result<Self, vk::Result> {
        unsafe {
            Self::with_options(
                device,
                memory,
                extent,
                format,
                usage,
                &ImageCreateOptions {
                    queue_family_indices: queue_families,
                    ..Default::default()
                },
            )
        }
    }

    pub unsafe fn with_options(
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        options: &ImageCreateOptions<'_>,
    ) -> Result<Self, vk::Result> {
        let mut image = Self {
            device: device.clone(),
            handle: vk::Image::null(),
            view: vk::ImageView::null(),
            memory: vk::DeviceMemory::null(),
            extent,
            format,
        };
        image.handle = with_image_create_info(format, extent, usage, options, |info| unsafe {
            device.create_image(info, None)
        })?;
        let r = unsafe { device.get_image_memory_requirements(image.handle) };
        image.memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(r.size)
                    .memory_type_index(memory_type(
                        memory,
                        r.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )?),
                None,
            )
        }?;
        unsafe { device.bind_image_memory(image.handle, image.memory, 0) }?;
        image.view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image.handle)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(color_range()),
                None,
            )
        }?;
        Ok(image)
    }
}

fn with_image_create_info<R>(
    format: vk::Format,
    extent: vk::Extent2D,
    usage: vk::ImageUsageFlags,
    options: &ImageCreateOptions<'_>,
    invoke: impl FnOnce(&vk::ImageCreateInfo<'_>) -> R,
) -> R {
    let mut format_list = (!options.view_formats.is_empty())
        .then(|| vk::ImageFormatListCreateInfo::default().view_formats(options.view_formats));
    let mut info = vk::ImageCreateInfo::default()
        .flags(options.image_flags)
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(if options.queue_family_indices.is_empty() {
            vk::SharingMode::EXCLUSIVE
        } else {
            vk::SharingMode::CONCURRENT
        })
        .queue_family_indices(options.queue_family_indices);
    if let Some(format_list) = format_list.as_mut() {
        info.p_next = (format_list as *mut vk::ImageFormatListCreateInfo<'_>).cast();
    }
    invoke(&info)
}
impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.handle, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

pub struct Buffer {
    device: ash::Device,
    pub handle: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
}
impl Buffer {
    pub unsafe fn new(
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        size: u64,
        usage: vk::BufferUsageFlags,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<Self, vk::Result> {
        let mut buffer = Self {
            device: device.clone(),
            handle: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size,
        };
        buffer.handle = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default().size(size).usage(usage),
                None,
            )
        }?;
        let r = unsafe { device.get_buffer_memory_requirements(buffer.handle) };
        buffer.memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(r.size)
                    .memory_type_index(memory_type(memory, r.memory_type_bits, flags)?),
                None,
            )
        }?;
        unsafe { device.bind_buffer_memory(buffer.handle, buffer.memory, 0) }?;
        Ok(buffer)
    }
    pub unsafe fn write(&self, bytes: &[u8]) -> Result<(), vk::Result> {
        if bytes.len() as u64 > self.size {
            return Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
        }
        let p = unsafe {
            self.device
                .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
        }?;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.cast(), bytes.len());
            self.device.unmap_memory(self.memory);
        }
        Ok(())
    }
    pub unsafe fn read(&self, bytes: &mut [u8]) -> Result<(), vk::Result> {
        if bytes.len() as u64 > self.size {
            return Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
        }
        let p = unsafe {
            self.device
                .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
        }?;
        unsafe {
            std::ptr::copy_nonoverlapping(p.cast(), bytes.as_mut_ptr(), bytes.len());
            self.device.unmap_memory(self.memory);
        }
        Ok(())
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.handle, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
pub fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

pub unsafe fn compute_memory_barrier(device: &ash::Device, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    unsafe {
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

pub unsafe fn transfer_memory_barrier(device: &ash::Device, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
    unsafe {
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

pub unsafe fn image_barrier(
    device: &ash::Device,
    command: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
) {
    let b = vk::ImageMemoryBarrier::default()
        .image(image)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .src_access_mask(if old == vk::ImageLayout::UNDEFINED {
            vk::AccessFlags::empty()
        } else {
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
        })
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .subresource_range(color_range());
    unsafe {
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[b],
        );
    }
}
pub unsafe fn memory_barrier(device: &ash::Device, command: vk::CommandBuffer) {
    let b = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ)
        .dst_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ);
    unsafe {
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[b],
            &[],
            &[],
        );
    }
}
