use ash::vk;
use std::ffi::CStr;

pub(crate) const EXT_SWAPCHAIN_MAINTENANCE1_NAME: &CStr = c"VK_EXT_swapchain_maintenance1";
pub(crate) const KHR_SWAPCHAIN_MAINTENANCE1_NAME: &CStr = c"VK_KHR_swapchain_maintenance1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Maintenance1Flavor {
    Ext,
    Khr,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Maintenance1Support {
    pub(crate) enabled: bool,
    pub(crate) flavor: Option<Maintenance1Flavor>,
}

unsafe fn extension_enabled(info: &vk::DeviceCreateInfo<'_>, expected: &CStr) -> bool {
    if info.enabled_extension_count == 0 || info.pp_enabled_extension_names.is_null() {
        return false;
    }
    let names = unsafe {
        std::slice::from_raw_parts(
            info.pp_enabled_extension_names,
            info.enabled_extension_count as usize,
        )
    };
    names.iter().any(|name| {
        !name.is_null() && unsafe { CStr::from_ptr(*name) }.to_bytes() == expected.to_bytes()
    })
}

pub(crate) unsafe fn maintenance1_support(info: &vk::DeviceCreateInfo<'_>) -> Maintenance1Support {
    let mut next = info.p_next;
    let mut feature_enabled = false;
    while !next.is_null() {
        let header = unsafe { &*next.cast::<vk::BaseInStructure<'_>>() };
        if header.s_type == vk::StructureType::PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT
        {
            feature_enabled = unsafe {
                (*next.cast::<vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT<'_>>())
                    .swapchain_maintenance1
                    != 0
            };
            break;
        }
        next = header.p_next.cast();
    }
    if !feature_enabled {
        return Maintenance1Support::default();
    }

    let ext_enabled = unsafe { extension_enabled(info, EXT_SWAPCHAIN_MAINTENANCE1_NAME) };
    let khr_enabled = unsafe { extension_enabled(info, KHR_SWAPCHAIN_MAINTENANCE1_NAME) };
    let flavor = if khr_enabled {
        Some(Maintenance1Flavor::Khr)
    } else if ext_enabled {
        Some(Maintenance1Flavor::Ext)
    } else {
        None
    };
    Maintenance1Support {
        enabled: flavor.is_some(),
        flavor,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PresentScaling {
    pub(crate) behavior: vk::PresentScalingFlagsEXT,
    pub(crate) gravity_x: vk::PresentGravityFlagsEXT,
    pub(crate) gravity_y: vk::PresentGravityFlagsEXT,
}

#[cfg(test)]
mod tests {
    use super::{Maintenance1Flavor, Maintenance1Support, maintenance1_support};
    use ash::vk;
    use ash::vk::Handle;

    #[test]
    fn with_create_info_rebuilds_owned_nodes_for_the_downstream_callback() {
        let modes = [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX];
        let mut mode_info = vk::SwapchainPresentModesCreateInfoEXT::default().present_modes(&modes);
        let mut scaling = vk::SwapchainPresentScalingCreateInfoEXT::default()
            .scaling_behavior(vk::PresentScalingFlagsEXT::ASPECT_RATIO_STRETCH)
            .present_gravity_x(vk::PresentGravityFlagsEXT::CENTERED)
            .present_gravity_y(vk::PresentGravityFlagsEXT::CENTERED);
        mode_info.p_next =
            (&mut scaling as *mut vk::SwapchainPresentScalingCreateInfoEXT<'_>).cast();
        let info = vk::SwapchainCreateInfoKHR::default()
            .image_extent(vk::Extent2D {
                width: 1280,
                height: 720,
            })
            .push_next(&mut mode_info);
        let template = crate::state::SwapchainTemplate::from_create_info(&info).unwrap();

        let recorded = template.with_create_info(
            vk::SurfaceKHR::from_raw(1),
            vk::Extent2D {
                width: 3440,
                height: 1440,
            },
            vk::SwapchainKHR::from_raw(2),
            |physical_info| {
                let head = physical_info.p_next;
                assert!(!head.is_null());
                assert_eq!(physical_info.image_extent.width, 3440);
                assert_eq!(physical_info.image_extent.height, 1440);
                physical_info.old_swapchain
            },
        );

        assert_eq!(recorded, vk::SwapchainKHR::from_raw(2));
    }

    #[test]
    fn enabled_ext_feature_is_detected_without_disabling_overlay() {
        let mut feature = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
            .swapchain_maintenance1(true);
        let names = [
            c"VK_KHR_swapchain".as_ptr(),
            c"VK_EXT_swapchain_maintenance1".as_ptr(),
        ];
        let info = vk::DeviceCreateInfo::default()
            .enabled_extension_names(&names)
            .push_next(&mut feature);

        assert_eq!(
            unsafe { maintenance1_support(&info) },
            Maintenance1Support {
                enabled: true,
                flavor: Some(Maintenance1Flavor::Ext),
            }
        );
    }

    #[test]
    fn feature_bit_without_an_extension_name_is_not_enabled() {
        let mut feature = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
            .swapchain_maintenance1(true);
        let info = vk::DeviceCreateInfo::default().push_next(&mut feature);

        assert_eq!(
            unsafe { maintenance1_support(&info) },
            Maintenance1Support::default()
        );
    }

    #[test]
    fn khr_name_is_preferred_when_both_aliases_are_enabled() {
        let mut feature = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
            .swapchain_maintenance1(true);
        let names = [
            c"VK_KHR_swapchain".as_ptr(),
            c"VK_EXT_swapchain_maintenance1".as_ptr(),
            c"VK_KHR_swapchain_maintenance1".as_ptr(),
        ];
        let info = vk::DeviceCreateInfo::default()
            .enabled_extension_names(&names)
            .push_next(&mut feature);

        assert_eq!(
            unsafe { maintenance1_support(&info) },
            Maintenance1Support {
                enabled: true,
                flavor: Some(Maintenance1Flavor::Khr),
            }
        );
    }

    #[test]
    fn ordinary_device_without_maintenance_remains_disabled() {
        let info = vk::DeviceCreateInfo::default();

        assert_eq!(
            unsafe { maintenance1_support(&info) },
            Maintenance1Support::default()
        );
    }

    #[test]
    fn disabled_feature_bit_does_not_enable_an_extension_alias() {
        let mut feature = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default();
        let names = [c"VK_EXT_swapchain_maintenance1".as_ptr()];
        let info = vk::DeviceCreateInfo::default()
            .enabled_extension_names(&names)
            .push_next(&mut feature);

        assert_eq!(
            unsafe { maintenance1_support(&info) },
            Maintenance1Support::default()
        );
    }
}
