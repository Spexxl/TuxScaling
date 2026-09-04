use super::*;
use ash::vk;
use ash::vk::Handle;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

static FAIL: AtomicUsize = AtomicUsize::new(0);
static EVENTS: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

unsafe extern "system" fn create_image(
    _: vk::Device,
    _: *const vk::ImageCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Image,
) -> vk::Result {
    if FAIL.load(Ordering::SeqCst) == 1 {
        return vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
    }
    unsafe {
        *out = vk::Image::from_raw(1);
    }
    EVENTS.lock().unwrap().push("image");
    vk::Result::SUCCESS
}
unsafe extern "system" fn requirements(
    _: vk::Device,
    _: vk::Image,
    out: *mut vk::MemoryRequirements,
) {
    unsafe {
        *out = vk::MemoryRequirements {
            size: 4096,
            alignment: 256,
            memory_type_bits: 1,
        };
    }
}
unsafe extern "system" fn allocate(
    _: vk::Device,
    _: *const vk::MemoryAllocateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::DeviceMemory,
) -> vk::Result {
    if FAIL.load(Ordering::SeqCst) == 2 {
        return vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
    }
    unsafe {
        *out = vk::DeviceMemory::from_raw(2);
    }
    EVENTS.lock().unwrap().push("memory");
    vk::Result::SUCCESS
}
unsafe extern "system" fn bind(
    _: vk::Device,
    _: vk::Image,
    _: vk::DeviceMemory,
    _: u64,
) -> vk::Result {
    if FAIL.load(Ordering::SeqCst) == 3 {
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
    } else {
        vk::Result::SUCCESS
    }
}
unsafe extern "system" fn create_view(
    _: vk::Device,
    _: *const vk::ImageViewCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::ImageView,
) -> vk::Result {
    if FAIL.load(Ordering::SeqCst) == 4 {
        return vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
    }
    unsafe {
        *out = vk::ImageView::from_raw(3);
    }
    EVENTS.lock().unwrap().push("view");
    vk::Result::SUCCESS
}
unsafe extern "system" fn destroy_image(
    _: vk::Device,
    image: vk::Image,
    _: *const vk::AllocationCallbacks<'_>,
) {
    if image != vk::Image::null() {
        EVENTS.lock().unwrap().push("destroy image");
    }
}
unsafe extern "system" fn destroy_view(
    _: vk::Device,
    view: vk::ImageView,
    _: *const vk::AllocationCallbacks<'_>,
) {
    if view != vk::ImageView::null() {
        EVENTS.lock().unwrap().push("destroy view");
    }
}
unsafe extern "system" fn free(
    _: vk::Device,
    memory: vk::DeviceMemory,
    _: *const vk::AllocationCallbacks<'_>,
) {
    if memory != vk::DeviceMemory::null() {
        EVENTS.lock().unwrap().push("free memory");
    }
}
#[test]
fn partial_image_allocations_are_rolled_back() {
    let device = unsafe {
        ash::Device::load_with(
            |name| {
                match name.to_bytes() {
                    b"vkCreateImage" => create_image as *const (),
                    b"vkGetImageMemoryRequirements" => requirements as *const (),
                    b"vkAllocateMemory" => allocate as *const (),
                    b"vkBindImageMemory" => bind as *const (),
                    b"vkCreateImageView" => create_view as *const (),
                    b"vkDestroyImage" => destroy_image as *const (),
                    b"vkDestroyImageView" => destroy_view as *const (),
                    b"vkFreeMemory" => free as *const (),
                    _ => std::ptr::null(),
                }
                .cast()
            },
            vk::Device::from_raw(1),
        )
    };
    let mut memory = vk::PhysicalDeviceMemoryProperties {
        memory_type_count: 1,
        ..Default::default()
    };
    memory.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    for fail in 1..=5 {
        FAIL.store(fail, Ordering::SeqCst);
        EVENTS.lock().unwrap().clear();
        let image = unsafe {
            Image::new(
                &device,
                &memory,
                vk::Extent2D {
                    width: 32,
                    height: 32,
                },
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED,
            )
        };
        assert_eq!(image.is_err(), fail < 5);
        drop(image);
        let events = EVENTS.lock().unwrap();
        let expected: &[&str] = match fail {
            1 => &[],
            2 => &["image", "destroy image"],
            3 | 4 => &["image", "memory", "destroy image", "free memory"],
            _ => &[
                "image",
                "memory",
                "view",
                "destroy view",
                "destroy image",
                "free memory",
            ],
        };
        assert_eq!(&*events, expected, "failure stage {fail}");
    }
}
