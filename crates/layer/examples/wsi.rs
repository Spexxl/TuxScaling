#![allow(clippy::missing_safety_doc)]
use ash::vk;
use std::{
    ffi::{c_char, c_int, c_uint, c_ulong, c_void},
    time::{Duration, Instant},
};
use tuxscaling_vulkan::image_barrier;
use x11rb::{
    connection::Connection,
    protocol::xproto::{ChangeWindowAttributesAux, ConnectionExt},
};

fn game_extent() -> vk::Extent2D {
    match std::env::var("TUXSCALING_TEST_SCENARIO").as_deref() {
        Ok("upscale") => vk::Extent2D {
            width: 1280,
            height: 720,
        },
        Ok("native") => vk::Extent2D {
            width: 1920,
            height: 1080,
        },
        Ok("aspect") => vk::Extent2D {
            width: 1024,
            height: 768,
        },
        _ => vk::Extent2D {
            width: 400,
            height: 300,
        },
    }
}

#[link(name = "X11")]
unsafe extern "C" {
    fn XOpenDisplay(name: *const c_char) -> *mut c_void;
    fn XDefaultRootWindow(display: *mut c_void) -> c_ulong;
    fn XCreateSimpleWindow(
        display: *mut c_void,
        parent: c_ulong,
        x: c_int,
        y: c_int,
        width: c_uint,
        height: c_uint,
        border: c_uint,
        border_pixel: c_ulong,
        background: c_ulong,
    ) -> c_ulong;
    fn XStoreName(display: *mut c_void, window: c_ulong, name: *const c_char) -> c_int;
    fn XInternAtom(display: *mut c_void, name: *const c_char, only_if_exists: c_int) -> c_ulong;
    fn XChangeProperty(
        display: *mut c_void,
        window: c_ulong,
        property: c_ulong,
        property_type: c_ulong,
        format: c_int,
        mode: c_int,
        data: *const u8,
        element_count: c_int,
    ) -> c_int;
    fn XMapWindow(display: *mut c_void, window: c_ulong) -> c_int;
    fn XResizeWindow(display: *mut c_void, window: c_ulong, width: c_uint, height: c_uint)
    -> c_int;
    fn XDestroyWindow(display: *mut c_void, window: c_ulong) -> c_int;
    fn XFlush(display: *mut c_void) -> c_int;
    fn XCloseDisplay(display: *mut c_void) -> c_int;
}

unsafe fn mark_fullscreen(display: *mut c_void, window: c_ulong) {
    let state = unsafe { XInternAtom(display, c"_NET_WM_STATE".as_ptr(), 0) };
    let fullscreen = unsafe { XInternAtom(display, c"_NET_WM_STATE_FULLSCREEN".as_ptr(), 0) };
    let atom_type = unsafe { XInternAtom(display, c"ATOM".as_ptr(), 0) };
    unsafe {
        XChangeProperty(
            display,
            window,
            state,
            atom_type,
            32,
            0,
            (&fullscreen as *const c_ulong).cast(),
            1,
        );
    }
}

fn parse_frame_limit(value: Option<&str>) -> Option<u32> {
    value
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn parses_positive_frame_limits() {
        assert_eq!(super::parse_frame_limit(Some("780")), Some(780));
        assert_eq!(super::parse_frame_limit(Some("0")), None);
        assert_eq!(super::parse_frame_limit(Some("invalid")), None);
        assert_eq!(super::parse_frame_limit(None), None);
    }
}

struct Chain {
    surface: vk::SurfaceKHR,
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    ready: Vec<vk::Semaphore>,
    acquired: vk::Semaphore,
    fence: vk::Fence,
    command: vk::CommandBuffer,
}
unsafe fn replace(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    device: &ash::Device,
    surfaces: &ash::khr::surface::Instance,
    swapchains: &ash::khr::swapchain::Device,
    chain: &mut Chain,
    extent: vk::Extent2D,
) {
    unsafe {
        device.device_wait_idle().unwrap();
        let caps = surfaces
            .get_physical_device_surface_capabilities(physical, chain.surface)
            .unwrap();
        let formats = surfaces
            .get_physical_device_surface_formats(physical, chain.surface)
            .unwrap();
        let format = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap();
        let extent = if caps.current_extent.width == u32::MAX
            || std::env::var("TUXSCALING_TEST_FORCE_VIRTUAL")
                .ok()
                .as_deref()
                == Some("1")
        {
            extent
        } else {
            caps.current_extent
        };
        let count = (caps.min_image_count + 1).min(if caps.max_image_count == 0 {
            u32::MAX
        } else {
            caps.max_image_count
        });
        let alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::INHERIT,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        ]
        .into_iter()
        .find(|f| caps.supported_composite_alpha.contains(*f))
        .unwrap();
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(chain.surface)
            .min_image_count(count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(alpha)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(chain.handle);
        let new = swapchains.create_swapchain(&info, None).unwrap();
        for s in chain.ready.drain(..) {
            device.destroy_semaphore(s, None);
        }
        swapchains.destroy_swapchain(chain.handle, None);
        chain.handle = new;
        chain.images = swapchains.get_swapchain_images(new).unwrap();
        let mut count = 0;
        assert_eq!(
            (swapchains.fp().get_swapchain_images_khr)(
                device.handle(),
                new,
                &mut count,
                std::ptr::null_mut()
            ),
            vk::Result::SUCCESS
        );
        assert_eq!(count as usize, chain.images.len());
        if count > 1 {
            let mut first = vk::Image::null();
            count = 1;
            assert_eq!(
                (swapchains.fp().get_swapchain_images_khr)(
                    device.handle(),
                    new,
                    &mut count,
                    &mut first
                ),
                vk::Result::INCOMPLETE
            );
            assert_eq!(first, chain.images[0]);
            assert_eq!(count, 1);
        }
        chain.ready = chain
            .images
            .iter()
            .map(|_| {
                device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .unwrap()
            })
            .collect();
        let _ = instance;
    }
}
fn main() {
    unsafe {
        run();
    }
}
unsafe fn run() {
    unsafe {
        let display = XOpenDisplay(std::ptr::null());
        assert!(!display.is_null(), "an X11 or Xwayland display is required");
        let windows = [0, 1].map(|i| {
            let w = XCreateSimpleWindow(
                display,
                XDefaultRootWindow(display),
                20 + i * 440,
                40,
                game_extent().width,
                game_extent().height,
                0,
                0,
                0,
            );
            XStoreName(display, w, c"TuxScaling WSI validation".as_ptr());
            XFlush(display);
            if std::env::var("TUXSCALING_TEST_SCENARIO").is_ok() {
                let (connection, _) = x11rb::connect(None).unwrap();
                connection
                    .change_window_attributes(
                        w as u32,
                        &ChangeWindowAttributesAux::new().override_redirect(1),
                    )
                    .unwrap()
                    .check()
                    .unwrap();
                connection.flush().unwrap();
            }
            XMapWindow(display, w);
            mark_fullscreen(display, w);
            w
        });
        XFlush(display);
        let entry = ash::Entry::load().unwrap();
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
        let extensions = [
            ash::khr::surface::NAME.as_ptr(),
            ash::khr::xlib_surface::NAME.as_ptr(),
        ];
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app)
                    .enabled_extension_names(&extensions),
                None,
            )
            .unwrap();
        let xlib = ash::khr::xlib_surface::Instance::new(&entry, &instance);
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let surfaces = windows.map(|w| {
            xlib.create_xlib_surface(
                &vk::XlibSurfaceCreateInfoKHR::default()
                    .dpy(display.cast())
                    .window(w),
                None,
            )
            .unwrap()
        });
        let physical = instance.enumerate_physical_devices().unwrap()[0];
        let families = instance.get_physical_device_queue_family_properties(physical);
        let family = families
            .iter()
            .enumerate()
            .find(|(i, f)| {
                f.queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                    && surfaces.iter().all(|s| {
                        surface_loader
                            .get_physical_device_surface_support(physical, *i as u32, *s)
                            .unwrap()
                    })
            })
            .unwrap()
            .0 as u32;
        let n = families[family as usize].queue_count.min(2) as usize;
        let priorities = [1.0, 1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities[..n])];
        let device = instance
            .create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&[ash::khr::swapchain::NAME.as_ptr()]),
                None,
            )
            .unwrap();
        let queue0 = device.get_device_queue(family, 0);
        let queue1 = device.get_device_queue2(
            &vk::DeviceQueueInfo2::default()
                .queue_family_index(family)
                .queue_index((n - 1) as u32),
        );
        let queue_handles = [queue0, queue1];
        let swapchains = ash::khr::swapchain::Device::new(&instance, &device);
        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .unwrap();
        let commands = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .command_buffer_count(2),
            )
            .unwrap();
        let mut chains = surfaces
            .into_iter()
            .enumerate()
            .map(|(i, surface)| Chain {
                surface,
                handle: vk::SwapchainKHR::null(),
                images: Vec::new(),
                ready: Vec::new(),
                acquired: device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .unwrap(),
                fence: device
                    .create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                    .unwrap(),
                command: commands[i],
            })
            .collect::<Vec<_>>();
        for chain in &mut chains {
            replace(
                &instance,
                physical,
                &device,
                &surface_loader,
                &swapchains,
                chain,
                game_extent(),
            );
        }
        let start = Instant::now();
        if std::env::var("TUXSCALING_TEST_SCENARIO").is_ok() {
            let display = tuxscaling_display::X11Display::connect().unwrap();
            for window in windows {
                let rect = display.window_rect(window).unwrap();
                let expected = if std::env::var("TUXSCALING_TEST_FORCE_RESIZE_FAILURE").as_deref()
                    == Ok("1")
                {
                    game_extent()
                } else {
                    vk::Extent2D {
                        width: 1920,
                        height: 1080,
                    }
                };
                assert_eq!(
                    [rect.width, rect.height],
                    [expected.width, expected.height],
                    "physical presentation extent"
                );
            }
        }
        let seconds = std::env::var("TUXSCALING_TEST_SECONDS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(8);
        let frame_limit =
            parse_frame_limit(std::env::var("TUXSCALING_TEST_FRAMES").ok().as_deref());
        let mut frame = 0u32;
        let mut grouped = 0;
        let mut resizes = 0;
        let resize_interval = std::env::var("TUXSCALING_TEST_RESIZE_INTERVAL")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(4);
        let present_proc = entry
            .get_instance_proc_addr(instance.handle(), c"vkQueuePresentKHR".as_ptr())
            .unwrap();
        let present: vk::PFN_vkQueuePresentKHR = std::mem::transmute(present_proc);
        while frame_limit.is_some_and(|limit| frame < limit)
            || frame_limit.is_none() && start.elapsed() < Duration::from_secs(seconds)
        {
            if resize_interval > 0 && frame > 0 && frame.is_multiple_of(resize_interval) {
                let extent = if resizes % 2 == 0 {
                    vk::Extent2D {
                        width: 480,
                        height: 320,
                    }
                } else {
                    vk::Extent2D {
                        width: 400,
                        height: 300,
                    }
                };
                for w in windows {
                    XResizeWindow(display, w, extent.width, extent.height);
                }
                XFlush(display);
                for chain in &mut chains {
                    replace(
                        &instance,
                        physical,
                        &device,
                        &surface_loader,
                        &swapchains,
                        chain,
                        extent,
                    );
                }
                resizes += 1;
            }
            let mut indices = [0u32; 2];
            let mut signals = [vk::Semaphore::null(); 2];
            for (i, chain) in chains.iter_mut().enumerate() {
                device
                    .wait_for_fences(&[chain.fence], true, 10_000_000_000)
                    .unwrap();
                let acquired = if frame.is_multiple_of(2) {
                    swapchains.acquire_next_image2(
                        &vk::AcquireNextImageInfoKHR::default()
                            .swapchain(chain.handle)
                            .timeout(u64::MAX)
                            .semaphore(chain.acquired)
                            .device_mask(1),
                    )
                } else {
                    swapchains.acquire_next_image(
                        chain.handle,
                        u64::MAX,
                        chain.acquired,
                        vk::Fence::null(),
                    )
                };
                let (index, _) = match acquired {
                    Ok(v) => v,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        replace(
                            &instance,
                            physical,
                            &device,
                            &surface_loader,
                            &swapchains,
                            chain,
                            vk::Extent2D {
                                width: 400,
                                height: 300,
                            },
                        );
                        swapchains
                            .acquire_next_image(
                                chain.handle,
                                u64::MAX,
                                chain.acquired,
                                vk::Fence::null(),
                            )
                            .unwrap()
                    }
                    Err(e) => panic!("{e:?}"),
                };
                indices[i] = index;
                signals[i] = chain.ready[index as usize];
                device.reset_fences(&[chain.fence]).unwrap();
                device
                    .reset_command_buffer(chain.command, vk::CommandBufferResetFlags::empty())
                    .unwrap();
                device
                    .begin_command_buffer(chain.command, &vk::CommandBufferBeginInfo::default())
                    .unwrap();
                image_barrier(
                    &device,
                    chain.command,
                    chain.images[index as usize],
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                let color = vk::ClearColorValue {
                    float32: [(frame % 60) as f32 / 60.0, 0.15 + i as f32 * 0.2, 0.2, 1.0],
                };
                device.cmd_clear_color_image(
                    chain.command,
                    chain.images[index as usize],
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &color,
                    &[tuxscaling_vulkan::color_range()],
                );
                image_barrier(
                    &device,
                    chain.command,
                    chain.images[index as usize],
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                );
                device.end_command_buffer(chain.command).unwrap();
                device
                    .queue_submit(
                        queue_handles[i],
                        &[vk::SubmitInfo::default()
                            .wait_semaphores(&[chain.acquired])
                            .wait_dst_stage_mask(&[vk::PipelineStageFlags::ALL_COMMANDS])
                            .command_buffers(&[chain.command])
                            .signal_semaphores(&[signals[i]])],
                        chain.fence,
                    )
                    .unwrap();
            }
            if frame.is_multiple_of(3) {
                let handles = [chains[0].handle, chains[1].handle];
                let mut results = [vk::Result::SUCCESS; 2];
                let info = vk::PresentInfoKHR::default()
                    .wait_semaphores(&signals)
                    .swapchains(&handles)
                    .image_indices(&indices)
                    .results(&mut results);
                let result = present(queue0, &info);
                assert!(result == vk::Result::SUCCESS || result == vk::Result::SUBOPTIMAL_KHR);
                grouped += 1;
            } else {
                for i in 0..2 {
                    let info = vk::PresentInfoKHR::default()
                        .wait_semaphores(&signals[i..i + 1])
                        .swapchains(std::slice::from_ref(&chains[i].handle))
                        .image_indices(&indices[i..i + 1]);
                    let result = present(queue_handles[i], &info);
                    assert!(result == vk::Result::SUCCESS || result == vk::Result::SUBOPTIMAL_KHR);
                }
            }
            frame += 1;
        }
        device.device_wait_idle().unwrap();
        for chain in chains {
            for s in chain.ready {
                device.destroy_semaphore(s, None);
            }
            device.destroy_semaphore(chain.acquired, None);
            device.destroy_fence(chain.fence, None);
            swapchains.destroy_swapchain(chain.handle, None);
            surface_loader.destroy_surface(chain.surface, None);
        }
        device.destroy_command_pool(pool, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
        for w in windows {
            XDestroyWindow(display, w);
        }
        XCloseDisplay(display);
        eprintln!(
            "WSI test complete: frames={frame}, grouped presents={grouped}, resize cycles={resizes}, queues={n}"
        );
        assert!(grouped > 0);
        if resize_interval > 0 {
            assert!(resizes > 0);
        }
    }
}
