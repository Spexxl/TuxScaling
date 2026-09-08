use ash::vk;

pub struct Gpu {
    _entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub memory: vk::PhysicalDeviceMemoryProperties,
    pub queue: vk::Queue,
    pub pool: vk::CommandPool,
}
impl Gpu {
    pub unsafe fn new() -> Self {
        let entry = unsafe { ash::Entry::load() }.unwrap();
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
        }
        .unwrap();
        let physical = unsafe { instance.enumerate_physical_devices() }.unwrap()[0];
        let physical_features = unsafe { instance.get_physical_device_features(physical) };
        let features = vk::PhysicalDeviceFeatures {
            shader_int16: physical_features.shader_int16,
            shader_storage_image_write_without_format: physical_features
                .shader_storage_image_write_without_format,
            ..Default::default()
        };
        let mut supported_vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut supported_features2 =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut supported_vulkan12);
        unsafe { instance.get_physical_device_features2(physical, &mut supported_features2) };
        let mut enabled_vulkan12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(supported_vulkan12.shader_float16 == vk::TRUE);
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let family = families
            .iter()
            .position(|f| {
                f.queue_flags
                    .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS)
            })
            .unwrap() as u32;
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&[1.0])];
        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_features(&features)
                    .push_next(&mut enabled_vulkan12),
                None,
            )
        }
        .unwrap();
        let queue = unsafe { device.get_device_queue(family, 0) };
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .unwrap();
        Self {
            _entry: entry,
            instance,
            device,
            memory,
            queue,
            pool,
        }
    }
    pub unsafe fn submit(&self, record: impl FnOnce(vk::CommandBuffer)) {
        let command = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .unwrap()[0];
        unsafe {
            self.device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
        }
        .unwrap();
        record(command);
        unsafe { self.device.end_command_buffer(command) }.unwrap();
        let fence = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .unwrap();
        unsafe {
            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[command])],
                    fence,
                )
                .unwrap();
            self.device
                .wait_for_fences(&[fence], true, 30_000_000_000)
                .unwrap();
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.pool, &[command]);
        }
    }
}
impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
