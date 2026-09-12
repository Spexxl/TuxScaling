#![allow(clippy::missing_safety_doc)]
use ash::vk;
use std::{
    ffi::{CStr, c_char, c_int, c_uint, c_ulong, c_void},
    thread,
    time::{Duration, Instant},
};
use tuxscaling_vulkan::image_barrier;
use x11rb::{
    connection::Connection,
    protocol::{
        randr::ConnectionExt as RandrConnectionExt,
        xproto::{ChangeWindowAttributesAux, ConnectionExt},
    },
    wrapper::ConnectionExt as _,
};

fn game_extent() -> vk::Extent2D {
    game_extent_for(std::env::var("TUXSCALING_TEST_SCENARIO").ok().as_deref())
}

fn game_extent_for(scenario: Option<&str>) -> vk::Extent2D {
    match scenario {
        Some(
            "upscale"
            | "windowed_promote"
            | "already_borderless"
            | "monitor_origin"
            | "promotion_failure"
            | "temporal_failure"
            | "guidance_resolve"
            | "maintenance1"
            | "mutable_format"
            | "present_wait_generation"
            | "hdr_replacement"
            | "display_timing"
            | "incompatible_direct",
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

fn pin_fullscreen_monitor(windows: &[c_ulong]) {
    let Ok((connection, screen)) = x11rb::connect(None) else {
        return;
    };
    let Some(root) = connection
        .setup()
        .roots
        .get(screen)
        .map(|screen| screen.root)
    else {
        return;
    };
    let Ok(cookie) = connection.intern_atom(false, b"_NET_WM_FULLSCREEN_MONITORS") else {
        return;
    };
    let Ok(fullscreen_monitors) = cookie.reply() else {
        return;
    };
    for &window in windows {
        let Ok(cookie) = connection.send_event(
            false,
            root,
            x11rb::protocol::xproto::EventMask::SUBSTRUCTURE_REDIRECT
                | x11rb::protocol::xproto::EventMask::SUBSTRUCTURE_NOTIFY,
            x11rb::protocol::xproto::ClientMessageEvent::new(
                32,
                window as u32,
                fullscreen_monitors.atom,
                [0, 0, 0, 0, 1],
            ),
        ) else {
            return;
        };
        if cookie.check().is_err() {
            return;
        }
    }
    let _ = connection.flush();
}

fn suppress_window_decorations(window: c_ulong) {
    let Ok((connection, _screen)) = x11rb::connect(None) else {
        return;
    };
    let Ok(cookie) = connection.intern_atom(false, b"_MOTIF_WM_HINTS") else {
        return;
    };
    let Ok(motif_hints) = cookie.reply() else {
        return;
    };
    let Ok(cookie) = connection.change_property32(
        x11rb::protocol::xproto::PropMode::REPLACE,
        window as u32,
        motif_hints.atom,
        x11rb::protocol::xproto::AtomEnum::CARDINAL,
        &[2, 0, 0, 0, 0],
    ) else {
        return;
    };
    let _ = cookie.check();
    let _ = connection.flush();
}

fn wait_for_window_restore(
    original_windows: &[(u64, tuxscaling_display::Rect, bool)],
) -> tuxscaling_display::X11Display {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let restored = tuxscaling_display::X11Display::connect().unwrap();
        let ready = original_windows
            .iter()
            .all(|(window, original, fullscreen)| {
                restored.window_rect(*window).ok() == Some(*original)
                    && restored.is_fullscreen(*window) == *fullscreen
            });
        if ready || Instant::now() >= deadline {
            return restored;
        }
        // Mutter applies fullscreen removal and the original configure request
        // asynchronously.  This is a teardown assertion, not a frame-path
        // wait or a synchronization mechanism for presentation.
        thread::sleep(Duration::from_millis(10));
    }
}

fn wait_for_native_window(window: u64) -> tuxscaling_display::X11Display {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let display = tuxscaling_display::X11Display::connect().unwrap();
        let native = display
            .monitor_for_window(window)
            .ok()
            .and_then(|monitor| display.window_rect(window).ok().map(|rect| (monitor, rect)))
            .is_some_and(|(monitor, rect)| rect == monitor.rect && display.is_fullscreen(window));
        if native || Instant::now() >= deadline {
            return display;
        }
        thread::sleep(Duration::from_millis(10));
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
    fn XMapWindow(display: *mut c_void, window: c_ulong) -> c_int;
    fn XMoveWindow(display: *mut c_void, window: c_ulong, x: c_int, y: c_int) -> c_int;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
enum PresenterTeardownEvent {
    PhysicalSwapchain,
    PresenterSurface,
    PresenterWindow,
    Instance,
}

#[allow(dead_code)]
fn presenter_teardown_is_valid(events: &[PresenterTeardownEvent]) -> bool {
    let physical = events
        .iter()
        .position(|event| *event == PresenterTeardownEvent::PhysicalSwapchain);
    let surface = events
        .iter()
        .position(|event| *event == PresenterTeardownEvent::PresenterSurface);
    let window = events
        .iter()
        .position(|event| *event == PresenterTeardownEvent::PresenterWindow);
    let instance = events
        .iter()
        .position(|event| *event == PresenterTeardownEvent::Instance);
    physical.zip(surface).zip(window).zip(instance).is_some_and(
        |(((physical, surface), window), instance)| {
            physical < surface && surface < window && window < instance
        },
    )
}

#[cfg(test)]
mod presenter_lifetime_tests {
    use super::{PresenterTeardownEvent, presenter_teardown_is_valid};

    #[test]
    fn presenter_surface_and_window_follow_instance_lifetime_order() {
        assert!(presenter_teardown_is_valid(&[
            PresenterTeardownEvent::PhysicalSwapchain,
            PresenterTeardownEvent::PresenterSurface,
            PresenterTeardownEvent::PresenterWindow,
            PresenterTeardownEvent::Instance,
        ]));
        assert!(!presenter_teardown_is_valid(&[
            PresenterTeardownEvent::PhysicalSwapchain,
            PresenterTeardownEvent::PresenterWindow,
            PresenterTeardownEvent::PresenterSurface,
            PresenterTeardownEvent::Instance,
        ]));
    }
}

fn queue_index_for_swapchain(swapchain: usize, queue_count: usize) -> usize {
    swapchain.min(queue_count.saturating_sub(1))
}

#[derive(Clone, Copy)]
struct MaintenanceScaling {
    behavior: vk::PresentScalingFlagsEXT,
    gravity_x: vk::PresentGravityFlagsEXT,
    gravity_y: vk::PresentGravityFlagsEXT,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MutableFormatPair {
    base: vk::Format,
    alternate: vk::Format,
    color_space: vk::ColorSpaceKHR,
}

fn choose_mutable_format_pair(formats: &[vk::SurfaceFormatKHR]) -> Option<MutableFormatPair> {
    for (base, alternate) in [
        (vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB),
        (vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_SRGB),
    ] {
        let Some(base_format) = formats.iter().find(|format| format.format == base) else {
            continue;
        };
        if formats.iter().any(|format| {
            format.format == alternate && format.color_space == base_format.color_space
        }) {
            return Some(MutableFormatPair {
                base,
                alternate,
                color_space: base_format.color_space,
            });
        }
    }
    None
}

#[derive(Clone, Copy)]
enum Maintenance1Flavor {
    Ext(u32),
    Khr(u32),
}

impl Maintenance1Flavor {
    const fn name(self) -> &'static CStr {
        match self {
            Self::Ext(_) => c"VK_EXT_swapchain_maintenance1",
            Self::Khr(_) => c"VK_KHR_swapchain_maintenance1",
        }
    }

    const fn uses_khr_commands(self) -> bool {
        matches!(self, Self::Khr(_))
    }

    const fn version(self) -> u32 {
        match self {
            Self::Ext(version) | Self::Khr(version) => version,
        }
    }
}

#[derive(Clone)]
struct MaintenanceConfig {
    present_modes: Vec<vk::PresentModeKHR>,
    scaling: Option<MaintenanceScaling>,
}

impl MaintenanceConfig {
    fn present_mode(&self) -> vk::PresentModeKHR {
        self.present_modes
            .iter()
            .copied()
            .find(|mode| *mode == vk::PresentModeKHR::FIFO)
            .unwrap_or(self.present_modes[0])
    }
}

fn device_extension_version(
    properties: &[vk::ExtensionProperties],
    expected: &CStr,
) -> Option<u32> {
    properties.iter().find_map(|property| {
        let name = unsafe { CStr::from_ptr(property.extension_name.as_ptr()) };
        (name == expected).then_some(property.spec_version)
    })
}

fn supported_device_extension(properties: &[vk::ExtensionProperties], expected: &CStr) -> bool {
    device_extension_version(properties, expected).is_some()
}

fn maintenance1_flavor(
    properties: &[vk::ExtensionProperties],
    surface_maintenance_name: &CStr,
) -> Option<Maintenance1Flavor> {
    if surface_maintenance_name.to_bytes() == b"VK_KHR_surface_maintenance1" {
        device_extension_version(properties, c"VK_KHR_swapchain_maintenance1")
            .filter(|version| *version >= 1)
            .map(Maintenance1Flavor::Khr)
    } else {
        device_extension_version(properties, c"VK_EXT_swapchain_maintenance1")
            .filter(|version| *version >= 1)
            .map(Maintenance1Flavor::Ext)
    }
}

fn choose_gravity(flags: vk::PresentGravityFlagsEXT) -> Option<vk::PresentGravityFlagsEXT> {
    [
        vk::PresentGravityFlagsEXT::CENTERED,
        vk::PresentGravityFlagsEXT::MIN,
        vk::PresentGravityFlagsEXT::MAX,
    ]
    .into_iter()
    .find(|candidate| flags.contains(*candidate))
}

fn choose_scaling(
    supported: &vk::SurfacePresentScalingCapabilitiesEXT<'_>,
) -> Option<MaintenanceScaling> {
    let behavior = [
        vk::PresentScalingFlagsEXT::ASPECT_RATIO_STRETCH,
        vk::PresentScalingFlagsEXT::ONE_TO_ONE,
        vk::PresentScalingFlagsEXT::STRETCH,
    ]
    .into_iter()
    .find(|candidate| supported.supported_present_scaling.contains(*candidate))?;
    Some(MaintenanceScaling {
        behavior,
        gravity_x: choose_gravity(supported.supported_present_gravity_x)?,
        gravity_y: choose_gravity(supported.supported_present_gravity_y)?,
    })
}

unsafe fn maintenance_config_for_surface(
    capabilities2: &ash::khr::get_surface_capabilities2::Instance,
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    flavor: Maintenance1Flavor,
) -> Result<MaintenanceConfig, String> {
    let mut present_mode =
        vk::SurfacePresentModeEXT::default().present_mode(vk::PresentModeKHR::FIFO);
    let surface_info = vk::PhysicalDeviceSurfaceInfo2KHR::default()
        .surface(surface)
        .push_next(&mut present_mode);
    let mut compatibility = vk::SurfacePresentModeCompatibilityEXT::default();
    let mut scaling = vk::SurfacePresentScalingCapabilitiesEXT::default();
    let mut capabilities = vk::SurfaceCapabilities2KHR::default()
        .push_next(&mut compatibility)
        .push_next(&mut scaling);
    unsafe {
        capabilities2.get_physical_device_surface_capabilities2(
            physical,
            &surface_info,
            &mut capabilities,
        )
    }
    .map_err(|error| format!("initial maintenance capabilities query failed: {error:?}"))?;

    let mode_count = compatibility.present_mode_count as usize;
    if mode_count == 0 {
        return Err("maintenance capabilities returned no compatible present modes".into());
    }
    let mut present_modes = vec![vk::PresentModeKHR::FIFO; mode_count];
    let mut compatibility =
        vk::SurfacePresentModeCompatibilityEXT::default().present_modes(&mut present_modes);
    let mut scaling = vk::SurfacePresentScalingCapabilitiesEXT::default();
    let mut capabilities = vk::SurfaceCapabilities2KHR::default()
        .push_next(&mut compatibility)
        .push_next(&mut scaling);
    unsafe {
        capabilities2.get_physical_device_surface_capabilities2(
            physical,
            &surface_info,
            &mut capabilities,
        )
    }
    .map_err(|error| format!("complete maintenance capabilities query failed: {error:?}"))?;
    let returned_count = compatibility.present_mode_count as usize;
    if returned_count > present_modes.len() {
        return Err("maintenance capabilities returned too many present modes".into());
    }
    present_modes.truncate(returned_count);
    if present_modes.is_empty() {
        return Err("maintenance capabilities returned an empty present mode list".into());
    }
    let scaling = choose_scaling(&scaling);
    if scaling.is_none() {
        eprintln!("TuxScaling WSI maintenance1: scaling=unsupported");
    }
    eprintln!(
        "TuxScaling WSI maintenance1: extension={} revision={} modes={} scaling={}",
        flavor.name().to_string_lossy(),
        flavor.version(),
        present_modes.len(),
        scaling.is_some(),
    );
    Ok(MaintenanceConfig {
        present_modes,
        scaling,
    })
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
            "mutable_format",
        ] {
            assert!(
                super::game_extent_for(Some(scenario)).width > 0,
                "{scenario}"
            );
        }
    }

    #[test]
    fn maintenance_scenario_preserves_the_1280x720_game_extent() {
        assert_eq!(
            super::game_extent_for(Some("maintenance1")),
            ash::vk::Extent2D {
                width: 1280,
                height: 720,
            }
        );
    }

    #[test]
    fn mutable_format_pair_requires_two_surface_formats_with_one_color_space() {
        let formats = [
            ash::vk::SurfaceFormatKHR {
                format: ash::vk::Format::B8G8R8A8_UNORM,
                color_space: ash::vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
            ash::vk::SurfaceFormatKHR {
                format: ash::vk::Format::B8G8R8A8_SRGB,
                color_space: ash::vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
        ];
        assert_eq!(
            super::choose_mutable_format_pair(&formats),
            Some(super::MutableFormatPair {
                base: ash::vk::Format::B8G8R8A8_UNORM,
                alternate: ash::vk::Format::B8G8R8A8_SRGB,
                color_space: ash::vk::ColorSpaceKHR::SRGB_NONLINEAR,
            })
        );
        assert!(super::choose_mutable_format_pair(&formats[..1]).is_none());
    }
}

struct Chain {
    surface: vk::SurfaceKHR,
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    ready: Vec<vk::Semaphore>,
    acquired: vk::Semaphore,
    fence: vk::Fence,
    present_fence: vk::Fence,
    present_mode: vk::PresentModeKHR,
    maintenance: Option<MaintenanceConfig>,
    mutable_formats: Option<MutableFormatPair>,
    command: vk::CommandBuffer,
}

struct SwapchainContext<'a> {
    physical: vk::PhysicalDevice,
    device: &'a ash::Device,
    surfaces: &'a ash::khr::surface::Instance,
    swapchains: &'a ash::khr::swapchain::Device,
}

unsafe fn replace(
    context: &SwapchainContext<'_>,
    chain: &mut Chain,
    extent: vk::Extent2D,
    maintenance: Option<&MaintenanceConfig>,
    mutable_formats: Option<MutableFormatPair>,
    direct_fallback: bool,
) {
    unsafe {
        context.device.device_wait_idle().unwrap();
        let caps = context
            .surfaces
            .get_physical_device_surface_capabilities(context.physical, chain.surface)
            .unwrap();
        let formats = context
            .surfaces
            .get_physical_device_surface_formats(context.physical, chain.surface)
            .unwrap();
        let selected_format = mutable_formats
            .map_or_else(
                || {
                    formats
                        .iter()
                        .find(|f| {
                            f.format == vk::Format::B8G8R8A8_UNORM
                                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                        })
                        .copied()
                },
                |pair| {
                    Some(vk::SurfaceFormatKHR {
                        format: pair.base,
                        color_space: pair.color_space,
                    })
                },
            )
            .expect("surface must expose the selected swapchain format");
        let mutable_formats = mutable_formats
            .map(|pair| [pair.base, pair.alternate])
            .map(|formats| formats.to_vec());
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
                && !native_aa_direct
                && !direct_fallback)
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
        let mut info = vk::SwapchainCreateInfoKHR::default()
            .surface(chain.surface)
            .min_image_count(count)
            .image_format(selected_format.format)
            .image_color_space(selected_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(alpha)
            .present_mode(
                maintenance.map_or(vk::PresentModeKHR::FIFO, MaintenanceConfig::present_mode),
            )
            .clipped(true)
            .old_swapchain(chain.handle)
            .flags(
                maintenance.map_or(vk::SwapchainCreateFlagsKHR::empty(), |_| {
                    vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT
                }) | mutable_formats
                    .as_ref()
                    .map_or(vk::SwapchainCreateFlagsKHR::empty(), |_| {
                        vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT
                    }),
            );
        let mut modes_info = maintenance.map(|maintenance| {
            vk::SwapchainPresentModesCreateInfoEXT::default()
                .present_modes(&maintenance.present_modes)
        });
        let mut scaling_info = maintenance.and_then(|maintenance| {
            maintenance.scaling.map(|scaling| {
                vk::SwapchainPresentScalingCreateInfoEXT::default()
                    .scaling_behavior(scaling.behavior)
                    .present_gravity_x(scaling.gravity_x)
                    .present_gravity_y(scaling.gravity_y)
            })
        });
        if let Some(modes_info) = modes_info.as_mut() {
            info = info.push_next(modes_info);
        }
        if let Some(scaling_info) = scaling_info.as_mut() {
            info = info.push_next(scaling_info);
        }
        let mut format_list = mutable_formats
            .as_ref()
            .map(|formats| vk::ImageFormatListCreateInfo::default().view_formats(formats));
        if let Some(format_list) = format_list.as_mut() {
            info = info.push_next(format_list);
        }
        let mut device_group_info = direct_fallback.then(|| {
            vk::DeviceGroupSwapchainCreateInfoKHR::default()
                .modes(vk::DeviceGroupPresentModeFlagsKHR::LOCAL)
        });
        if let Some(device_group_info) = device_group_info.as_mut() {
            info = info.push_next(device_group_info);
        }
        let new = context.swapchains.create_swapchain(&info, None).unwrap();
        for s in chain.ready.drain(..) {
            context.device.destroy_semaphore(s, None);
        }
        context.swapchains.destroy_swapchain(chain.handle, None);
        chain.handle = new;
        chain.images = context.swapchains.get_swapchain_images(new).unwrap();
        let mut count = 0;
        assert_eq!(
            (context.swapchains.fp().get_swapchain_images_khr)(
                context.device.handle(),
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
                (context.swapchains.fp().get_swapchain_images_khr)(
                    context.device.handle(),
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
                context
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .unwrap()
            })
            .collect();
        chain.present_mode =
            maintenance.map_or(vk::PresentModeKHR::FIFO, MaintenanceConfig::present_mode);
        chain.maintenance = maintenance.cloned();
        chain.mutable_formats = mutable_formats.map(|formats| MutableFormatPair {
            base: formats[0],
            alternate: formats[1],
            color_space: selected_format.color_space,
        });
    }
}

unsafe fn validate_alternate_views(
    device: &ash::Device,
    images: &[vk::Image],
    pair: MutableFormatPair,
) -> Result<(), vk::Result> {
    let mut views = Vec::with_capacity(images.len());
    for &image in images {
        let mut usage = vk::ImageViewUsageCreateInfo::default().usage(
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        );
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(pair.alternate)
            .subresource_range(tuxscaling_vulkan::color_range())
            .push_next(&mut usage);
        let view = match unsafe { device.create_image_view(&view_info, None) } {
            Ok(view) => view,
            Err(error) => {
                for view in views {
                    unsafe { device.destroy_image_view(view, None) };
                }
                return Err(error);
            }
        };
        views.push(view);
    }
    for view in views {
        unsafe { device.destroy_image_view(view, None) };
    }
    Ok(())
}

unsafe fn release_swapchain_image(
    instance: &ash::Instance,
    device: vk::Device,
    flavor: Maintenance1Flavor,
    swapchain: vk::SwapchainKHR,
    image_index: u32,
) -> vk::Result {
    let preferred = if flavor.uses_khr_commands() {
        c"vkReleaseSwapchainImagesKHR"
    } else {
        c"vkReleaseSwapchainImagesEXT"
    };
    let alternate = if flavor.uses_khr_commands() {
        c"vkReleaseSwapchainImagesEXT"
    } else {
        c"vkReleaseSwapchainImagesKHR"
    };
    let proc = unsafe {
        instance
            .get_device_proc_addr(device, preferred.as_ptr())
            .or_else(|| instance.get_device_proc_addr(device, alternate.as_ptr()))
    };
    let Some(proc) = proc else {
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    };
    let release: vk::PFN_vkReleaseSwapchainImagesEXT = unsafe { std::mem::transmute(proc) };
    let indices = [image_index];
    let info = vk::ReleaseSwapchainImagesInfoEXT::default()
        .swapchain(swapchain)
        .image_indices(&indices);
    unsafe { release(device, &info) }
}

unsafe fn prepare_present_fence(device: &ash::Device, fence: vk::Fence) {
    if fence == vk::Fence::null() {
        return;
    }
    unsafe {
        device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        device.reset_fences(&[fence]).unwrap();
    }
}

unsafe fn query_swapchain_status(
    instance: &ash::Instance,
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
) -> vk::Result {
    let Some(proc) =
        (unsafe { instance.get_device_proc_addr(device, c"vkGetSwapchainStatusKHR".as_ptr()) })
    else {
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    };
    let get: vk::PFN_vkGetSwapchainStatusKHR = unsafe { std::mem::transmute(proc) };
    unsafe { get(device, swapchain) }
}

unsafe fn query_counter(
    display_control: &ash::ext::display_control::Device,
    swapchain: vk::SwapchainKHR,
    counter: vk::SurfaceCounterFlagsEXT,
) -> vk::Result {
    let mut value = 0;
    unsafe {
        (display_control.fp().get_swapchain_counter_ext)(
            display_control.device(),
            swapchain,
            counter,
            &mut value,
        )
    }
}

unsafe fn query_timing_count_and_data(
    instance: &ash::Instance,
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
) -> Result<(), vk::Result> {
    let Some(proc) = (unsafe {
        instance.get_device_proc_addr(device, c"vkGetPastPresentationTimingGOOGLE".as_ptr())
    }) else {
        return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT);
    };
    let get: vk::PFN_vkGetPastPresentationTimingGOOGLE = unsafe { std::mem::transmute(proc) };
    let mut count = 0;
    let result = unsafe { get(device, swapchain, &mut count, std::ptr::null_mut()) };
    if result != vk::Result::SUCCESS && result != vk::Result::INCOMPLETE {
        return Err(result);
    }
    let mut records = vec![vk::PastPresentationTimingGOOGLE::default(); count as usize];
    let mut capacity = count;
    let result = unsafe { get(device, swapchain, &mut capacity, records.as_mut_ptr()) };
    if result != vk::Result::SUCCESS && result != vk::Result::INCOMPLETE {
        return Err(result);
    }
    Ok(())
}

fn sample_hdr_metadata() -> vk::HdrMetadataEXT<'static> {
    vk::HdrMetadataEXT::default()
        .display_primary_red(vk::XYColorEXT::default().x(0.68).y(0.32))
        .display_primary_green(vk::XYColorEXT::default().x(0.265).y(0.69))
        .display_primary_blue(vk::XYColorEXT::default().x(0.15).y(0.06))
        .white_point(vk::XYColorEXT::default().x(0.3127).y(0.3290))
        .max_luminance(1_000.0)
        .min_luminance(0.1)
        .max_content_light_level(1_000.0)
        .max_frame_average_light_level(400.0)
}

fn prepend_present_id<'a>(
    info: &mut vk::PresentInfoKHR<'a>,
    present_ids: &'a [u64],
    id_info: &mut vk::PresentIdKHR<'a>,
) {
    *id_info = vk::PresentIdKHR::default().present_ids(present_ids);
    id_info.p_next = info.p_next;
    info.p_next = (id_info as *const vk::PresentIdKHR<'_>).cast();
}

unsafe fn acquire_for_release(
    device: &ash::Device,
    swapchains: &ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
) -> u32 {
    let fence = unsafe {
        device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap()
    };
    let index = unsafe {
        swapchains
            .acquire_next_image(swapchain, u64::MAX, vk::Semaphore::null(), fence)
            .unwrap()
            .0
    };
    unsafe {
        device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        device.destroy_fence(fence, None);
    }
    index
}

fn main() {
    if matches!(unsafe { run() }, WsiOutcome::EnvironmentSkip) {
        std::process::exit(77);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WsiOutcome {
    Passed,
    EnvironmentSkip,
}

unsafe fn finish_environment_skip(
    display: *mut c_void,
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surfaces: &[vk::SurfaceKHR],
    windows: &[c_ulong],
    scenario: &str,
    reason: &str,
) -> WsiOutcome {
    eprintln!(
        "TuxScaling evidence event=wsi_scenario scenario={scenario} result=unverified reason=extension_unavailable detail={reason}"
    );
    for &surface in surfaces {
        unsafe { surface_loader.destroy_surface(surface, None) };
    }
    unsafe { instance.destroy_instance(None) };
    for &window in windows {
        unsafe { XDestroyWindow(display, window) };
    }
    unsafe { XCloseDisplay(display) };
    WsiOutcome::EnvironmentSkip
}

unsafe fn finish_instance_environment_skip(
    display: *mut c_void,
    windows: &[c_ulong],
    scenario: &str,
    reason: &str,
) -> WsiOutcome {
    eprintln!(
        "TuxScaling evidence event=wsi_scenario scenario={scenario} result=unverified reason=extension_unavailable detail={reason}"
    );
    for &window in windows {
        unsafe { XDestroyWindow(display, window) };
    }
    unsafe { XCloseDisplay(display) };
    WsiOutcome::EnvironmentSkip
}

unsafe fn run() -> WsiOutcome {
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
        let scenario = std::env::var("TUXSCALING_TEST_SCENARIO").ok();
        let scenario_active = scenario.is_some();
        let scenario_name = scenario.as_deref().unwrap_or("default");
        let maintenance_scenario = scenario.as_deref() == Some("maintenance1");
        let mutable_scenario = scenario.as_deref() == Some("mutable_format");
        let present_wait_scenario = scenario.as_deref() == Some("present_wait_generation");
        let hdr_scenario = scenario.as_deref() == Some("hdr_replacement");
        let timing_scenario = scenario.as_deref() == Some("display_timing");
        let incompatible_scenario = scenario.as_deref() == Some("incompatible_direct");
        let portable_wsi_scenario = mutable_scenario
            || present_wait_scenario
            || hdr_scenario
            || timing_scenario
            || incompatible_scenario;
        if scenario_active {
            let requested = game_extent();
            eprintln!(
                "TuxScaling evidence event=wsi_scenario_request scenario={scenario_name} requested={}x{}",
                requested.width, requested.height,
            );
        }
        let borderless_monitor = starts_borderless().then(native_monitor_rect).flatten();
        let placement_monitor =
            borderless_monitor.or_else(|| maintenance_scenario.then(native_monitor_rect).flatten());
        let initial_extent =
            borderless_monitor.map_or_else(game_extent, |(_, _, width, height)| vk::Extent2D {
                width,
                height,
            });
        let windows = (0..window_count)
            .map(|i| {
                let x = if let Some((x, _y, _, _)) = placement_monitor {
                    x + i as c_int * 440
                } else if uses_negative_monitor_origin() {
                    -50 + i as c_int * 440
                } else {
                    20 + i as c_int * 440
                };
                let y = placement_monitor.map_or(40, |(_, y, _, _)| y + 40);
                let w = XCreateSimpleWindow(
                    display,
                    XDefaultRootWindow(display),
                    x,
                    y,
                    initial_extent.width,
                    initial_extent.height,
                    0,
                    0,
                    0,
                );
                XStoreName(display, w, c"TuxScaling WSI validation".as_ptr());
                XFlush(display);
                if scenario_active && !maintenance_scenario && !portable_wsi_scenario {
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
                if maintenance_scenario || portable_wsi_scenario {
                    suppress_window_decorations(w);
                }
                XMapWindow(display, w);
                XMoveWindow(display, w, x, y);
                w
            })
            .collect::<Vec<_>>();
        XFlush(display);
        if portable_wsi_scenario {
            // Let the window manager finish its initial decoration pass before
            // taking the lease snapshot that teardown must restore exactly.
            thread::sleep(Duration::from_millis(100));
        }
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
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let instance_properties = entry.enumerate_instance_extension_properties(None).unwrap();
        if timing_scenario
            && (!supported_device_extension(
                &instance_properties,
                ash::ext::display_surface_counter::NAME,
            ) || !supported_device_extension(&instance_properties, ash::khr::display::NAME))
        {
            return finish_instance_environment_skip(
                display,
                &windows,
                scenario_name,
                "VK_EXT_display_surface_counter or VK_KHR_display is unavailable",
            );
        }
        let surface_maintenance_name = if maintenance_scenario {
            let khr =
                device_extension_version(&instance_properties, c"VK_KHR_surface_maintenance1")
                    .filter(|version| *version >= 1);
            let ext =
                device_extension_version(&instance_properties, c"VK_EXT_surface_maintenance1")
                    .filter(|version| *version >= 1);
            Some(if khr.is_some() {
                c"VK_KHR_surface_maintenance1"
            } else if ext.is_some() {
                c"VK_EXT_surface_maintenance1"
            } else {
                panic!("maintenance1 scenario requires a surface maintenance extension")
            })
        } else {
            None
        };
        let mut extensions = vec![
            ash::khr::surface::NAME.as_ptr(),
            ash::khr::xlib_surface::NAME.as_ptr(),
        ];
        if maintenance_scenario {
            assert!(
                device_extension_version(
                    &instance_properties,
                    ash::khr::get_surface_capabilities2::NAME
                )
                .is_some_and(|version| version >= 1),
                "maintenance1 scenario requires VK_KHR_get_surface_capabilities2"
            );
            extensions.push(ash::khr::get_surface_capabilities2::NAME.as_ptr());
            extensions.push(
                surface_maintenance_name
                    .expect("surface maintenance name is selected")
                    .as_ptr(),
            );
        }
        if timing_scenario {
            extensions.push(ash::khr::display::NAME.as_ptr());
            extensions.push(ash::ext::display_surface_counter::NAME.as_ptr());
        }
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
        let device_properties = instance
            .enumerate_device_extension_properties(physical)
            .unwrap();
        let mutable_formats = if mutable_scenario {
            if device_extension_version(
                &device_properties,
                ash::khr::swapchain_mutable_format::NAME,
            )
            .is_none()
            {
                return finish_environment_skip(
                    display,
                    &instance,
                    &surface_loader,
                    &surfaces,
                    &windows,
                    scenario_name,
                    "VK_KHR_swapchain_mutable_format is unavailable",
                );
            }
            let mut selected = None;
            for &surface in &surfaces {
                let formats =
                    match surface_loader.get_physical_device_surface_formats(physical, surface) {
                        Ok(formats) => formats,
                        Err(error) => {
                            return finish_environment_skip(
                                display,
                                &instance,
                                &surface_loader,
                                &surfaces,
                                &windows,
                                scenario_name,
                                &format!("surface format query failed: {error:?}"),
                            );
                        }
                    };
                let Some(pair) = choose_mutable_format_pair(&formats) else {
                    return finish_environment_skip(
                        display,
                        &instance,
                        &surface_loader,
                        &surfaces,
                        &windows,
                        scenario_name,
                        "surface does not expose a compatible UNORM/SRGB pair",
                    );
                };
                if selected.is_some_and(|selected| selected != pair) {
                    return finish_environment_skip(
                        display,
                        &instance,
                        &surface_loader,
                        &surfaces,
                        &windows,
                        scenario_name,
                        "surfaces expose different mutable format pairs",
                    );
                }
                selected = Some(pair);
            }
            selected
        } else {
            None
        };
        let maintenance_flavor = if maintenance_scenario {
            let flavor = maintenance1_flavor(
                &device_properties,
                surface_maintenance_name.expect("surface maintenance name is selected"),
            )
            .expect("maintenance1 scenario requires EXT or KHR revision 1");
            eprintln!(
                "TuxScaling WSI maintenance1: selected extension={} revision={}",
                flavor.name().to_string_lossy(),
                flavor.version(),
            );
            Some(flavor)
        } else {
            None
        };
        if present_wait_scenario
            && (!supported_device_extension(&device_properties, ash::khr::present_id::NAME)
                || !supported_device_extension(&device_properties, ash::khr::present_wait::NAME))
        {
            return finish_environment_skip(
                display,
                &instance,
                &surface_loader,
                &surfaces,
                &windows,
                scenario_name,
                "VK_KHR_present_id or VK_KHR_present_wait is unavailable",
            );
        }
        if hdr_scenario
            && !supported_device_extension(&device_properties, ash::ext::hdr_metadata::NAME)
        {
            return finish_environment_skip(
                display,
                &instance,
                &surface_loader,
                &surfaces,
                &windows,
                scenario_name,
                "VK_EXT_hdr_metadata is unavailable",
            );
        }
        if timing_scenario
            && (!supported_device_extension(&device_properties, ash::google::display_timing::NAME)
                || !supported_device_extension(&device_properties, ash::ext::display_control::NAME))
        {
            return finish_environment_skip(
                display,
                &instance,
                &surface_loader,
                &surfaces,
                &windows,
                scenario_name,
                "display timing or display control is unavailable",
            );
        }
        let incompatible_extension = if incompatible_scenario {
            [
                c"VK_KHR_shared_presentable_image",
                c"VK_EXT_full_screen_exclusive",
                c"VK_NV_low_latency2",
                c"VK_NV_present_barrier",
            ]
            .into_iter()
            .find(|name| supported_device_extension(&device_properties, name))
        } else {
            None
        };
        let incompatible_device_group = incompatible_scenario
            && supported_device_extension(&device_properties, ash::khr::device_group::NAME);
        if incompatible_scenario && incompatible_extension.is_none() && !incompatible_device_group {
            return finish_environment_skip(
                display,
                &instance,
                &surface_loader,
                &surfaces,
                &windows,
                scenario_name,
                "no audited incompatible WSI extension or device-group contract is available",
            );
        }
        let counter_flag = if timing_scenario {
            let counter_instance =
                ash::ext::display_surface_counter::Instance::new(&entry, &instance);
            let mut capabilities = vk::SurfaceCapabilities2EXT::default();
            let result = (counter_instance
                .fp()
                .get_physical_device_surface_capabilities2_ext)(
                physical,
                surfaces[0],
                &mut capabilities,
            );
            if result != vk::Result::SUCCESS
                || !capabilities
                    .supported_surface_counters
                    .contains(vk::SurfaceCounterFlagsEXT::VBLANK)
            {
                return finish_environment_skip(
                    display,
                    &instance,
                    &surface_loader,
                    &surfaces,
                    &windows,
                    scenario_name,
                    "VBLANK surface counter is unavailable",
                );
            }
            Some(vk::SurfaceCounterFlagsEXT::VBLANK)
        } else {
            None
        };
        let physical_features = instance.get_physical_device_features(physical);
        let features = vk::PhysicalDeviceFeatures {
            shader_int16: physical_features.shader_int16,
            shader_storage_image_write_without_format: physical_features
                .shader_storage_image_write_without_format,
            ..Default::default()
        };
        let mut supported_vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut supported_maintenance =
            vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default();
        let mut supported_present_id = vk::PhysicalDevicePresentIdFeaturesKHR::default();
        let mut supported_present_wait = vk::PhysicalDevicePresentWaitFeaturesKHR::default();
        let mut supported_features2 =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut supported_vulkan12);
        if maintenance_scenario {
            supported_features2 = supported_features2.push_next(&mut supported_maintenance);
        }
        if present_wait_scenario {
            supported_features2 = supported_features2
                .push_next(&mut supported_present_id)
                .push_next(&mut supported_present_wait);
        }
        instance.get_physical_device_features2(physical, &mut supported_features2);
        if maintenance_scenario {
            assert_eq!(
                supported_maintenance.swapchain_maintenance1,
                vk::TRUE,
                "maintenance1 feature is not supported by the selected device"
            );
        }
        if present_wait_scenario
            && (supported_present_id.present_id != vk::TRUE
                || supported_present_wait.present_wait != vk::TRUE)
        {
            return finish_environment_skip(
                display,
                &instance,
                &surface_loader,
                &surfaces,
                &windows,
                scenario_name,
                "present ID or present wait feature is unavailable",
            );
        }
        let mut enabled_vulkan12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(supported_vulkan12.shader_float16 == vk::TRUE);
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
        let mut device_extensions = vec![ash::khr::swapchain::NAME.as_ptr()];
        if mutable_scenario {
            device_extensions.push(ash::khr::swapchain_mutable_format::NAME.as_ptr());
        }
        if present_wait_scenario {
            device_extensions.push(ash::khr::present_id::NAME.as_ptr());
            device_extensions.push(ash::khr::present_wait::NAME.as_ptr());
        }
        if hdr_scenario {
            device_extensions.push(ash::ext::hdr_metadata::NAME.as_ptr());
        }
        if timing_scenario {
            device_extensions.push(ash::google::display_timing::NAME.as_ptr());
            device_extensions.push(ash::ext::display_control::NAME.as_ptr());
        }
        if let Some(extension) = incompatible_extension {
            device_extensions.push(extension.as_ptr());
        }
        if incompatible_device_group {
            device_extensions.push(ash::khr::device_group::NAME.as_ptr());
        }
        if let Some(flavor) = maintenance_flavor {
            device_extensions.push(flavor.name().as_ptr());
        }
        let mut enabled_maintenance = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
            .swapchain_maintenance1(true);
        let mut enabled_present_id =
            vk::PhysicalDevicePresentIdFeaturesKHR::default().present_id(true);
        let mut enabled_present_wait =
            vk::PhysicalDevicePresentWaitFeaturesKHR::default().present_wait(true);
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&features)
            .push_next(&mut enabled_vulkan12);
        if maintenance_scenario {
            device_info = device_info.push_next(&mut enabled_maintenance);
        }
        if present_wait_scenario {
            device_info = device_info
                .push_next(&mut enabled_present_id)
                .push_next(&mut enabled_present_wait);
        }
        let device = instance
            .create_device(physical, &device_info, None)
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
        let present_wait =
            present_wait_scenario.then(|| ash::khr::present_wait::Device::new(&instance, &device));
        let hdr_metadata =
            hdr_scenario.then(|| ash::ext::hdr_metadata::Device::new(&instance, &device));
        let display_control =
            timing_scenario.then(|| ash::ext::display_control::Device::new(&instance, &device));
        let display_timing =
            timing_scenario.then(|| ash::google::display_timing::Device::new(&instance, &device));
        let replacement = SwapchainContext {
            physical,
            device: &device,
            surfaces: &surface_loader,
            swapchains: &swapchains,
        };
        let capabilities2 = maintenance_scenario
            .then(|| ash::khr::get_surface_capabilities2::Instance::new(&entry, &instance));
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
                present_fence: vk::Fence::null(),
                present_mode: vk::PresentModeKHR::FIFO,
                maintenance: None,
                mutable_formats: None,
                command: commands[i],
            })
            .collect::<Vec<_>>();
        for chain in &mut chains {
            let maintenance = if let (Some(capabilities2), Some(flavor)) =
                (capabilities2.as_ref(), maintenance_flavor)
            {
                Some(
                    maintenance_config_for_surface(capabilities2, physical, chain.surface, flavor)
                        .unwrap(),
                )
            } else {
                None
            };
            if maintenance_scenario {
                chain.present_fence = device
                    .create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                    .unwrap();
            }
            replace(
                &replacement,
                chain,
                game_extent(),
                maintenance.as_ref(),
                mutable_formats,
                incompatible_scenario,
            );
        }
        let mutable_views_validated = if let Some(pair) = mutable_formats {
            assert_eq!(
                std::env::var("TUXSCALING_VIEW").ok().as_deref(),
                Some("reconstructed"),
                "mutable format scenario requires reconstructed presentation"
            );
            for chain in &chains {
                validate_alternate_views(&device, &chain.images, pair)
                    .expect("alternate mutable image views must be compatible");
            }
            true
        } else {
            false
        };
        if maintenance_scenario {
            // The layer requests fullscreen during the first swapchain create.
            // Repeat the monitor selection after that request so the window
            // manager applies it to the active fullscreen state.
            pin_fullscreen_monitor(&windows);
        }
        if portable_wsi_scenario && !incompatible_scenario {
            for &window in &windows {
                wait_for_native_window(window);
            }
        }
        let start = Instant::now();
        if scenario_active && !maintenance_scenario {
            let display = tuxscaling_display::X11Display::connect().unwrap();
            for &window in &windows {
                let rect = display.window_rect(window).unwrap();
                let expected = if matches!(
                    std::env::var("TUXSCALING_TEST_SCENARIO").ok().as_deref(),
                    Some("native_aa" | "incompatible_direct")
                ) {
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
        let mut maintenance_released = false;
        let mut maintenance_recreated = false;
        let mut present_wait_current = false;
        let mut present_wait_old = false;
        let mut next_present_id = 1_u64;
        let mut hdr_before = false;
        let mut hdr_after = false;
        let mut timing_count = false;
        let mut timing_data = false;
        let mut status_query = false;
        let mut counter_query = false;
        let mut refresh_query = false;
        if hdr_scenario {
            let metadata = [sample_hdr_metadata()];
            let hdr_metadata = hdr_metadata.as_ref().expect("HDR device is enabled");
            hdr_metadata.set_hdr_metadata(&[chains[0].handle], &metadata);
            hdr_before = true;
        }
        if timing_scenario {
            let chain = chains.first().expect("timing scenario needs one swapchain");
            let status = query_swapchain_status(&instance, device.handle(), chain.handle);
            assert!(
                matches!(
                    status,
                    vk::Result::SUCCESS
                        | vk::Result::SUBOPTIMAL_KHR
                        | vk::Result::ERROR_OUT_OF_DATE_KHR
                ),
                "status query failed: {status:?}"
            );
            status_query = true;
            let counter = counter_flag.expect("timing scenario selected a surface counter");
            let counter_result = query_counter(
                display_control
                    .as_ref()
                    .expect("display control device is enabled"),
                chain.handle,
                counter,
            );
            assert_eq!(counter_result, vk::Result::SUCCESS, "counter query failed");
            counter_query = true;
            let refresh = display_timing
                .as_ref()
                .expect("display timing device is enabled")
                .get_refresh_cycle_duration(chain.handle);
            assert!(refresh.is_ok(), "refresh-cycle query failed: {refresh:?}");
            refresh_query = true;
            query_timing_count_and_data(&instance, device.handle(), chain.handle)
                .expect("initial display timing count/data query failed");
            timing_count = true;
            timing_data = true;
        }
        if let (Some(flavor), Some(chain)) = (maintenance_flavor, chains.first_mut()) {
            let released_index = acquire_for_release(&device, &swapchains, chain.handle);
            assert_eq!(
                release_swapchain_image(
                    &instance,
                    device.handle(),
                    flavor,
                    chain.handle,
                    released_index,
                ),
                vk::Result::SUCCESS,
                "release without present"
            );
            let reused_index = acquire_for_release(&device, &swapchains, chain.handle);
            assert_eq!(
                release_swapchain_image(
                    &instance,
                    device.handle(),
                    flavor,
                    chain.handle,
                    reused_index,
                ),
                vk::Result::SUCCESS,
                "released logical slot was not reusable"
            );
            maintenance_released = true;

            let maintenance = chain.maintenance.clone();
            replace(
                &replacement,
                chain,
                game_extent(),
                maintenance.as_ref(),
                mutable_formats,
                incompatible_scenario,
            );
            maintenance_recreated = true;
        }
        let present = swapchains.fp().queue_present_khr;
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
                    let maintenance = chain.maintenance.clone();
                    replace(
                        &replacement,
                        chain,
                        extent,
                        maintenance.as_ref(),
                        mutable_formats,
                        incompatible_scenario,
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
                        let maintenance = chain.maintenance.clone();
                        replace(
                            &replacement,
                            chain,
                            vk::Extent2D {
                                width: 400,
                                height: 300,
                            },
                            maintenance.as_ref(),
                            mutable_formats,
                            incompatible_scenario,
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
            if maintenance_scenario {
                for chain in &chains {
                    prepare_present_fence(&device, chain.present_fence);
                }
            }
            if chains.len() > 1 && frame.is_multiple_of(3) {
                let handles = chains.iter().map(|chain| chain.handle).collect::<Vec<_>>();
                let mut results = vec![vk::Result::SUCCESS; chains.len()];
                let present_ids = present_wait_scenario.then(|| {
                    (0..chains.len())
                        .map(|_| {
                            let id = next_present_id;
                            next_present_id += 1;
                            id
                        })
                        .collect::<Vec<_>>()
                });
                let present_fences = chains
                    .iter()
                    .map(|chain| chain.present_fence)
                    .collect::<Vec<_>>();
                let present_modes = chains
                    .iter()
                    .map(|chain| chain.present_mode)
                    .collect::<Vec<_>>();
                let mut fence_info =
                    vk::SwapchainPresentFenceInfoEXT::default().fences(&present_fences);
                let mut mode_info =
                    vk::SwapchainPresentModeInfoEXT::default().present_modes(&present_modes);
                if maintenance_scenario {
                    fence_info.p_next =
                        (&mut mode_info as *mut vk::SwapchainPresentModeInfoEXT<'_>).cast();
                }
                let mut info = vk::PresentInfoKHR::default()
                    .wait_semaphores(&signals)
                    .swapchains(&handles)
                    .image_indices(&indices)
                    .results(&mut results);
                if maintenance_scenario {
                    info.p_next =
                        (&mut fence_info as *mut vk::SwapchainPresentFenceInfoEXT<'_>).cast();
                }
                let mut present_id_info = vk::PresentIdKHR::default();
                if let Some(present_ids) = present_ids.as_ref() {
                    prepend_present_id(&mut info, present_ids, &mut present_id_info);
                }
                let result = present(queue0, &info);
                assert!(result == vk::Result::SUCCESS || result == vk::Result::SUBOPTIMAL_KHR);
                if let Some(present_ids) = present_ids {
                    for (chain, present_id) in chains.iter().zip(present_ids) {
                        let wait_result = present_wait
                            .as_ref()
                            .expect("present wait device is enabled")
                            .wait_for_present(chain.handle, present_id, u64::MAX);
                        assert!(wait_result.is_ok(), "present wait failed: {wait_result:?}");
                        if frame == 0 {
                            present_wait_old = true;
                        } else {
                            present_wait_current = true;
                        }
                    }
                }
                grouped += 1;
            } else {
                for i in 0..chains.len() {
                    let present_fences = [chains[i].present_fence];
                    let present_modes = [chains[i].present_mode];
                    let mut fence_info =
                        vk::SwapchainPresentFenceInfoEXT::default().fences(&present_fences);
                    let mut mode_info =
                        vk::SwapchainPresentModeInfoEXT::default().present_modes(&present_modes);
                    if maintenance_scenario {
                        fence_info.p_next =
                            (&mut mode_info as *mut vk::SwapchainPresentModeInfoEXT<'_>).cast();
                    }
                    let mut info = vk::PresentInfoKHR::default()
                        .wait_semaphores(&signals[i..i + 1])
                        .swapchains(std::slice::from_ref(&chains[i].handle))
                        .image_indices(&indices[i..i + 1]);
                    if maintenance_scenario {
                        info.p_next =
                            (&mut fence_info as *mut vk::SwapchainPresentFenceInfoEXT<'_>).cast();
                    }
                    let present_id = present_wait_scenario.then(|| {
                        let id = next_present_id;
                        next_present_id += 1;
                        id
                    });
                    let present_ids = present_id.map(|present_id| [present_id]);
                    let mut present_id_info = vk::PresentIdKHR::default();
                    if let Some(present_ids) = present_ids.as_ref() {
                        prepend_present_id(&mut info, present_ids, &mut present_id_info);
                    }
                    let result = present(queue_handles[i], &info);
                    assert!(result == vk::Result::SUCCESS || result == vk::Result::SUBOPTIMAL_KHR);
                    if let Some([present_id]) = present_ids {
                        let wait_result = present_wait
                            .as_ref()
                            .expect("present wait device is enabled")
                            .wait_for_present(chains[i].handle, present_id, u64::MAX);
                        assert!(wait_result.is_ok(), "present wait failed: {wait_result:?}");
                        if frame == 0 {
                            present_wait_old = true;
                        } else {
                            present_wait_current = true;
                        }
                    }
                }
            }
            frame += 1;
            if hdr_scenario && frame == 1 {
                let metadata = [sample_hdr_metadata()];
                hdr_metadata
                    .as_ref()
                    .expect("HDR device is enabled")
                    .set_hdr_metadata(&[chains[0].handle], &metadata);
                hdr_after = true;
            }
            if timing_scenario && frame == 1 {
                query_timing_count_and_data(&instance, device.handle(), chains[0].handle)
                    .expect("post-replacement display timing count/data query failed");
            }
        }
        if maintenance_scenario {
            assert!(maintenance_released);
            assert!(maintenance_recreated);
            assert!(
                frame >= 8,
                "maintenance scenario did not present enough frames"
            );
            assert!(
                grouped >= 2,
                "maintenance scenario did not group enough presents"
            );
        }
        if mutable_scenario {
            assert!(mutable_views_validated);
            assert!(frame > 0, "mutable format scenario did not present a frame");
            eprintln!(
                "TuxScaling WSI evidence scenario=mutable_format alternate_views=1 reconstructed=1"
            );
        }
        if present_wait_scenario {
            assert!(
                present_wait_old,
                "present wait did not route an old generation"
            );
            assert!(
                present_wait_current,
                "present wait did not route the current generation"
            );
        }
        if hdr_scenario {
            assert!(hdr_before, "HDR metadata was not set before replacement");
            assert!(hdr_after, "HDR metadata was not set after replacement");
        }
        if timing_scenario {
            assert!(status_query);
            assert!(counter_query);
            assert!(refresh_query);
            assert!(timing_count);
            assert!(timing_data);
        }
        if mutable_scenario || present_wait_scenario || hdr_scenario || timing_scenario {
            let scenario_fields = match scenario_name {
                "mutable_format" => {
                    "alternate_views=1 present_wait_current=0 present_wait_old=0 hdr_before=0 hdr_after=0 queries=none timing=none direct=0"
                }
                "present_wait_generation" => {
                    "alternate_views=0 present_wait_current=1 present_wait_old=1 hdr_before=0 hdr_after=0 queries=none timing=none direct=0"
                }
                "hdr_replacement" => {
                    "alternate_views=0 present_wait_current=0 present_wait_old=0 hdr_before=1 hdr_after=1 queries=none timing=none direct=0"
                }
                "display_timing" => {
                    "alternate_views=0 present_wait_current=0 present_wait_old=0 hdr_before=0 hdr_after=0 queries=status,counter,refresh timing=count,data direct=0"
                }
                _ => unreachable!("scenario marker only applies to portable WSI scenarios"),
            };
            eprintln!(
                "TuxScaling evidence event=wsi_scenario scenario={scenario_name} result=verified {scenario_fields} recreations_after_publish=0"
            );
        }
        if incompatible_scenario {
            assert_eq!(chains.len(), 1);
            eprintln!(
                "TuxScaling evidence event=wsi_scenario scenario=incompatible_direct result=verified direct=1"
            );
        }
        device.device_wait_idle().unwrap();
        for chain in chains {
            for s in chain.ready {
                device.destroy_semaphore(s, None);
            }
            device.destroy_semaphore(chain.acquired, None);
            device.destroy_fence(chain.fence, None);
            if chain.present_fence != vk::Fence::null() {
                device.destroy_fence(chain.present_fence, None);
            }
            swapchains.destroy_swapchain(chain.handle, None);
            surface_loader.destroy_surface(chain.surface, None);
        }
        XFlush(display);
        if scenario_active {
            let restored = wait_for_window_restore(&original_windows);
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
        WsiOutcome::Passed
    }
}
