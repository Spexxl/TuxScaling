#![allow(clippy::missing_safety_doc)]
use ash::vk;
use std::{
    ffi::{c_char, c_int, c_uint, c_ulong, c_void},
    time::{Duration, Instant},
};
use tuxscaling_vulkan::image_barrier;
use x11rb::{
    connection::Connection,
    protocol::{
        randr::ConnectionExt as RandrConnectionExt,
        xproto::{ChangeWindowAttributesAux, ConnectionExt},
    },
};

fn game_extent() -> vk::Extent2D {
    game_extent_for(std::env::var("TUXSCALING_TEST_SCENARIO").ok().as_deref())
}

fn game_extent_for(scenario: Option<&str>) -> vk::Extent2D {
    match scenario {
        Some(
            "upscale" | "windowed_promote" | "already_borderless" | "monitor_origin"
            | "promotion_failure" | "temporal_failure" | "guidance_resolve",
        ) => vk::Extent2D {
            width: 1280,
            height: 720,
        },
        Some("native" | "native_aa") => vk::Extent2D {
            width: 1920,
            height: 1080,
        },
        Some("aspect") => vk::Extent2D {
            width: 1024,
            height: 768,
        },
        _ => vk::Extent2D {
            width: 400,
            height: 300,
        },
    }
}

fn starts_borderless() -> bool {
    matches!(
        std::env::var("TUXSCALING_TEST_SCENARIO").as_deref(),
        Ok("already_borderless")
    )
}

fn uses_negative_monitor_origin() -> bool {
    matches!(
        std::env::var("TUXSCALING_TEST_SCENARIO").as_deref(),
        Ok("monitor_origin")
    )
}

fn native_monitor_rect() -> Option<(i32, i32, u32, u32)> {
    let (connection, screen) = x11rb::connect(None).ok()?;
    let root = connection.setup().roots.get(screen)?.root;
    let reply = connection
        .randr_get_monitors(root, true)
        .ok()?
        .reply()
        .ok()?;
    let monitor = reply.monitors.first()?;
    Some((
        i32::from(monitor.x),
        i32::from(monitor.y),
        u32::from(monitor.width),
        u32::from(monitor.height),
    ))
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
    fn XMapWindow(display: *mut c_void, window: c_ulong) -> c_int;
    fn XResizeWindow(display: *mut c_void, window: c_ulong, width: c_uint, height: c_uint)
    -> c_int;
    fn XDestroyWindow(display: *mut c_void, window: c_ulong) -> c_int;
    fn XFlush(display: *mut c_void) -> c_int;
    fn XCloseDisplay(display: *mut c_void) -> c_int;
}

fn parse_frame_limit(value: Option<&str>) -> Option<u32> {
    value
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
}

fn single_window(value: Option<&str>) -> bool {
    value == Some("1")
}

fn requires_grouped_presents(window_count: usize) -> bool {
    window_count > 1
}

fn queue_index_for_swapchain(swapchain: usize, queue_count: usize) -> usize {
    swapchain.min(queue_count.saturating_sub(1))
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

    #[test]
    fn parses_single_window_benchmark_mode() {
        assert!(super::single_window(Some("1")));
        assert!(!super::single_window(Some("0")));
        assert!(!super::single_window(None));
    }

    #[test]
    fn skips_grouped_present_requirement_for_one_swapchain() {
        assert!(!super::requires_grouped_presents(1));
        assert!(super::requires_grouped_presents(2));
    }

    #[test]
    fn shares_the_last_available_queue_between_swapchains() {
        assert_eq!(super::queue_index_for_swapchain(0, 1), 0);
        assert_eq!(super::queue_index_for_swapchain(1, 1), 0);
        assert_eq!(super::queue_index_for_swapchain(1, 2), 1);
    }

    #[test]
    fn recognizes_native_output_window_scenarios() {
        for scenario in [
            "windowed_promote",
            "already_borderless",
            "native_aa",
            "aspect",
            "resize",
            "monitor_origin",
            "promotion_failure",
            "temporal_failure",
            "guidance_resolve",
        ] {
            assert!(
                super::game_extent_for(Some(scenario)).width > 0,
                "{scenario}"
            );
        }
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
        let promotion_failure_recreate = std::env::var("TUXSCALING_TEST_SCENARIO").ok().as_deref()
            == Some("promotion_failure")
            && std::env::var("TUXSCALING_TEST_FORCE_RESIZE_FAILURE")
                .ok()
                .as_deref()
                == Some("1")
            && chain.handle != vk::SwapchainKHR::null();
        let native_aa_direct =
            std::env::var("TUXSCALING_TEST_SCENARIO").ok().as_deref() == Some("native_aa");
        let extent = if caps.current_extent.width == u32::MAX
            || (std::env::var("TUXSCALING_TEST_FORCE_VIRTUAL")
                .ok()
                .as_deref()
                == Some("1")
                && !promotion_failure_recreate
                && !native_aa_direct)
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
        let window_count: usize = if single_window(
            std::env::var("TUXSCALING_TEST_SINGLE_WINDOW")
                .ok()
                .as_deref(),
        ) {
            1
        } else {
            2
        };
        let scenario_active = std::env::var("TUXSCALING_TEST_SCENARIO").is_ok();
        let borderless_monitor = starts_borderless().then(native_monitor_rect).flatten();
        let initial_extent =
            borderless_monitor.map_or_else(game_extent, |(_, _, width, height)| vk::Extent2D {
                width,
                height,
            });
        let windows = (0..window_count)
            .map(|i| {
                let w = XCreateSimpleWindow(
                    display,
                    XDefaultRootWindow(display),
                    if let Some((x, _y, _, _)) = borderless_monitor {
                        x + i as c_int * 440
                    } else if uses_negative_monitor_origin() {
                        -50 + i as c_int * 440
                    } else {
                        20 + i as c_int * 440
                    },
                    borderless_monitor.map_or(40, |(_, y, _, _)| y),
                    initial_extent.width,
                    initial_extent.height,
                    0,
                    0,
                    0,
                );
                XStoreName(display, w, c"TuxScaling WSI validation".as_ptr());
                XFlush(display);
                if scenario_active {
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
                w
            })
            .collect::<Vec<_>>();
        XFlush(display);
        let display_probe = tuxscaling_display::X11Display::connect().unwrap();
        let original_windows = windows
            .iter()
            .map(|&window| {
                (
                    window,
                    display_probe.window_rect(window).unwrap(),
                    display_probe.is_fullscreen(window),
                )
            })
            .collect::<Vec<_>>();
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
        let surfaces = windows
            .iter()
            .map(|&w| {
                xlib.create_xlib_surface(
                    &vk::XlibSurfaceCreateInfoKHR::default()
                        .dpy(display.cast())
                        .window(w),
                    None,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
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
        let n = families[family as usize]
            .queue_count
            .min(window_count as u32) as usize;
        let priorities = vec![1.0; n];
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
        let queue_handles = (0..window_count)
            .map(|swapchain| {
                device.get_device_queue2(
                    &vk::DeviceQueueInfo2::default()
                        .queue_family_index(family)
                        .queue_index(queue_index_for_swapchain(swapchain, n) as u32),
                )
            })
            .collect::<Vec<_>>();
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
                    .command_buffer_count(window_count as u32),
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
        if scenario_active {
            let display = tuxscaling_display::X11Display::connect().unwrap();
            for &window in &windows {
                let rect = display.window_rect(window).unwrap();
                let expected = if std::env::var("TUXSCALING_TEST_SCENARIO").ok().as_deref()
                    == Some("native_aa")
                {
                    game_extent()
                } else {
                    let monitor = display_probe.monitor_for_window(window).unwrap();
                    vk::Extent2D {
                        width: monitor.rect.width,
                        height: monitor.rect.height,
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
            let promotion_failure_finished =
                std::env::var("TUXSCALING_TEST_SCENARIO").ok().as_deref()
                    == Some("promotion_failure")
                    && resizes > 0;
            if resize_interval > 0
                && frame > 0
                && frame.is_multiple_of(resize_interval)
                && !promotion_failure_finished
            {
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
                for &window in &windows {
                    XResizeWindow(display, window, extent.width, extent.height);
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
            let mut indices = vec![0u32; chains.len()];
            let mut signals = vec![vk::Semaphore::null(); chains.len()];
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
            if chains.len() > 1 && frame.is_multiple_of(3) {
                let handles = chains.iter().map(|chain| chain.handle).collect::<Vec<_>>();
                let mut results = vec![vk::Result::SUCCESS; chains.len()];
                let info = vk::PresentInfoKHR::default()
                    .wait_semaphores(&signals)
                    .swapchains(&handles)
                    .image_indices(&indices)
                    .results(&mut results);
                let result = present(queue0, &info);
                assert!(result == vk::Result::SUCCESS || result == vk::Result::SUBOPTIMAL_KHR);
                grouped += 1;
            } else {
                for i in 0..chains.len() {
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
        XFlush(display);
        if scenario_active {
            let restored = tuxscaling_display::X11Display::connect().unwrap();
            for (window, original, fullscreen) in &original_windows {
                let current = restored.window_rect(*window).unwrap();
                assert_eq!(current, *original, "window geometry was not restored");
                assert_eq!(restored.is_fullscreen(*window), *fullscreen);
            }
        }
        device.destroy_command_pool(pool, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
        for window in windows {
            XDestroyWindow(display, window);
        }
        XCloseDisplay(display);
        eprintln!(
            "WSI test complete: frames={frame}, grouped presents={grouped}, resize cycles={resizes}, queues={n}"
        );
        if requires_grouped_presents(window_count) {
            assert!(grouped > 0);
        }
        if resize_interval > 0 {
            assert!(resizes > 0);
        }
    }
}
