use super::*;
use ash::vk;
use ash::vk::Handle;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

static FAIL: AtomicUsize = AtomicUsize::new(0);
static EVENTS: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
static OBSERVED_IMAGE_CREATE: Mutex<Option<ImageCreateSnapshot>> = Mutex::new(None);

#[derive(Debug, PartialEq, Eq)]
struct ImageCreateSnapshot {
    flags: vk::ImageCreateFlags,
    sharing_mode: vk::SharingMode,
    queue_family_indices: Vec<u32>,
    view_formats: Vec<vk::Format>,
}

unsafe extern "system" fn create_image(
    _: vk::Device,
    info: *const vk::ImageCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Image,
) -> vk::Result {
    if FAIL.load(Ordering::SeqCst) == 1 {
        return vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
    }
    let info = unsafe { &*info };
    let mut view_formats = Vec::new();
    let mut next = info.p_next;
    while !next.is_null() {
        let header = unsafe { &*next.cast::<vk::BaseInStructure<'_>>() };
        if header.s_type == vk::StructureType::IMAGE_FORMAT_LIST_CREATE_INFO {
            let list = unsafe { &*next.cast::<vk::ImageFormatListCreateInfo<'_>>() };
            if !list.p_view_formats.is_null() {
                view_formats = unsafe {
                    std::slice::from_raw_parts(list.p_view_formats, list.view_format_count as usize)
                }
                .to_vec();
            }
        }
        next = header.p_next.cast();
    }
    let queue_family_indices = if info.p_queue_family_indices.is_null() {
        Vec::new()
    } else {
        unsafe {
            std::slice::from_raw_parts(
                info.p_queue_family_indices,
                info.queue_family_index_count as usize,
            )
        }
        .to_vec()
    };
    *OBSERVED_IMAGE_CREATE.lock().unwrap() = Some(ImageCreateSnapshot {
        flags: info.flags,
        sharing_mode: info.sharing_mode,
        queue_family_indices,
        view_formats,
    });
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

fn test_device() -> ash::Device {
    unsafe {
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
    }
}

fn test_memory() -> vk::PhysicalDeviceMemoryProperties {
    let mut memory = vk::PhysicalDeviceMemoryProperties {
        memory_type_count: 1,
        ..Default::default()
    };
    memory.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    memory
}

#[test]
fn image_create_options_preserve_default_and_queue_sharing_contracts() {
    let device = test_device();
    let memory = test_memory();
    let extent = vk::Extent2D {
        width: 32,
        height: 32,
    };

    unsafe {
        Image::with_options(
            &device,
            &memory,
            extent,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
            &ImageCreateOptions::default(),
        )
        .unwrap();
    }
    let default_snapshot = OBSERVED_IMAGE_CREATE.lock().unwrap().take().unwrap();
    assert_eq!(default_snapshot.flags, vk::ImageCreateFlags::empty());
    assert_eq!(default_snapshot.sharing_mode, vk::SharingMode::EXCLUSIVE);
    assert!(default_snapshot.queue_family_indices.is_empty());
    assert!(default_snapshot.view_formats.is_empty());

    let queue_families = [2, 5];
    unsafe {
        Image::with_options(
            &device,
            &memory,
            extent,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
            &ImageCreateOptions {
                queue_family_indices: &queue_families,
                ..Default::default()
            },
        )
        .unwrap();
    }
    let sharing_snapshot = OBSERVED_IMAGE_CREATE.lock().unwrap().take().unwrap();
    assert_eq!(sharing_snapshot.flags, vk::ImageCreateFlags::empty());
    assert_eq!(sharing_snapshot.sharing_mode, vk::SharingMode::CONCURRENT);
    assert_eq!(sharing_snapshot.queue_family_indices, queue_families);
    assert!(sharing_snapshot.view_formats.is_empty());
}

#[test]
fn mutable_image_create_options_attach_only_the_owned_format_list() {
    let device = test_device();
    let memory = test_memory();
    let formats = [vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB];

    unsafe {
        Image::with_options(
            &device,
            &memory,
            vk::Extent2D {
                width: 32,
                height: 32,
            },
            formats[0],
            vk::ImageUsageFlags::SAMPLED,
            &ImageCreateOptions {
                image_flags: vk::ImageCreateFlags::MUTABLE_FORMAT,
                view_formats: &formats,
                ..Default::default()
            },
        )
        .unwrap();
    }

    let snapshot = OBSERVED_IMAGE_CREATE.lock().unwrap().take().unwrap();
    assert_eq!(snapshot.flags, vk::ImageCreateFlags::MUTABLE_FORMAT);
    assert_eq!(snapshot.view_formats, formats);
    assert!(
        !snapshot
            .flags
            .contains(vk::ImageCreateFlags::SPARSE_BINDING)
    );
}

#[test]
fn normal_image_creation_never_forwards_swapchain_only_contract_bits() {
    let device = test_device();
    let memory = test_memory();

    unsafe {
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
        .unwrap();
    }

    let snapshot = OBSERVED_IMAGE_CREATE.lock().unwrap().take().unwrap();
    assert!(snapshot.flags.is_empty());
    assert!(snapshot.view_formats.is_empty());
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
