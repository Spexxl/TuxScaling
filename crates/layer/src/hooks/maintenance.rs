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
pub(crate) enum MaintenanceCreateError {
    DuplicateStructure,
    NullArray,
    EmptyPresentModes,
    UnknownStructure,
    UnsupportedFlags,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PresentScaling {
    pub(crate) behavior: vk::PresentScalingFlagsEXT,
    pub(crate) gravity_x: vk::PresentGravityFlagsEXT,
    pub(crate) gravity_y: vk::PresentGravityFlagsEXT,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SwapchainMaintenanceTemplate {
    pub(crate) present_modes: Option<Vec<vk::PresentModeKHR>>,
    pub(crate) scaling: Option<PresentScaling>,
}

impl SwapchainMaintenanceTemplate {
    pub(crate) unsafe fn parse(
        mut next: *const std::ffi::c_void,
    ) -> Result<Self, MaintenanceCreateError> {
        let mut template = Self::default();
        while !next.is_null() {
            let header = unsafe { &*next.cast::<vk::BaseInStructure<'_>>() };
            match header.s_type {
                vk::StructureType::SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT => {
                    if template.present_modes.is_some() {
                        return Err(MaintenanceCreateError::DuplicateStructure);
                    }
                    let info =
                        unsafe { &*next.cast::<vk::SwapchainPresentModesCreateInfoEXT<'_>>() };
                    if info.present_mode_count == 0 {
                        return Err(MaintenanceCreateError::EmptyPresentModes);
                    }
                    if info.p_present_modes.is_null() {
                        return Err(MaintenanceCreateError::NullArray);
                    }
                    let modes = unsafe {
                        std::slice::from_raw_parts(
                            info.p_present_modes,
                            info.present_mode_count as usize,
                        )
                    }
                    .to_vec();
                    template.present_modes = Some(modes);
                }
                vk::StructureType::SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_EXT => {
                    if template.scaling.is_some() {
                        return Err(MaintenanceCreateError::DuplicateStructure);
                    }
                    let info =
                        unsafe { &*next.cast::<vk::SwapchainPresentScalingCreateInfoEXT<'_>>() };
                    template.scaling = Some(PresentScaling {
                        behavior: info.scaling_behavior,
                        gravity_x: info.present_gravity_x,
                        gravity_y: info.present_gravity_y,
                    });
                }
                _ => return Err(MaintenanceCreateError::UnknownStructure),
            }
            next = header.p_next.cast();
        }
        Ok(template)
    }
}

pub(crate) fn supported_swapchain_flags(flags: vk::SwapchainCreateFlagsKHR) -> bool {
    flags.is_empty() || flags == vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT
}

pub(crate) fn maintenance_create_supported(info: &vk::SwapchainCreateInfoKHR<'_>) -> bool {
    supported_swapchain_flags(info.flags)
        && unsafe { SwapchainMaintenanceTemplate::parse(info.p_next) }.is_ok()
}

#[cfg(test)]
mod tests {
    use super::{
        Maintenance1Flavor, Maintenance1Support, SwapchainMaintenanceTemplate,
        maintenance1_support, supported_swapchain_flags,
    };
    use ash::vk;
    use ash::vk::Handle;

    #[test]
    fn maintenance_create_chain_is_copied_without_retaining_application_pointers() {
        let modes = [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX];
        let mut mode_info = vk::SwapchainPresentModesCreateInfoEXT::default().present_modes(&modes);
        let mut scaling = vk::SwapchainPresentScalingCreateInfoEXT::default()
            .scaling_behavior(vk::PresentScalingFlagsEXT::ASPECT_RATIO_STRETCH)
            .present_gravity_x(vk::PresentGravityFlagsEXT::CENTERED)
            .present_gravity_y(vk::PresentGravityFlagsEXT::CENTERED);
        mode_info.p_next =
            (&mut scaling as *mut vk::SwapchainPresentScalingCreateInfoEXT<'_>).cast();

        let parsed = unsafe {
            SwapchainMaintenanceTemplate::parse(
                (&mode_info as *const vk::SwapchainPresentModesCreateInfoEXT<'_>).cast(),
            )
        }
        .unwrap();

        assert_eq!(parsed.present_modes.as_deref(), Some(modes.as_slice()));
        assert_eq!(
            parsed.scaling.unwrap().gravity_x,
            vk::PresentGravityFlagsEXT::CENTERED
        );
        assert_ne!(
            parsed.present_modes.as_ref().unwrap().as_ptr(),
            modes.as_ptr()
        );
    }

    #[test]
    fn maintenance_create_chain_accepts_null_and_reverse_node_order() {
        assert_eq!(
            unsafe { SwapchainMaintenanceTemplate::parse(std::ptr::null()) }.unwrap(),
            SwapchainMaintenanceTemplate::default()
        );

        let modes = [vk::PresentModeKHR::FIFO];
        let mut mode_info = vk::SwapchainPresentModesCreateInfoEXT::default().present_modes(&modes);
        let scaling = vk::SwapchainPresentScalingCreateInfoEXT {
            p_next: (&mut mode_info as *mut vk::SwapchainPresentModesCreateInfoEXT<'_>).cast(),
            ..Default::default()
        };

        let parsed = unsafe {
            SwapchainMaintenanceTemplate::parse(
                (&scaling as *const vk::SwapchainPresentScalingCreateInfoEXT<'_>).cast(),
            )
        }
        .unwrap();

        assert_eq!(parsed.present_modes.as_deref(), Some(modes.as_slice()));
        assert!(parsed.scaling.is_some());
    }

    #[test]
    fn maintenance_create_chain_rejects_duplicates_invalid_arrays_zero_modes_and_unknown_nodes() {
        let modes = [vk::PresentModeKHR::FIFO];
        let mut first = vk::SwapchainPresentModesCreateInfoEXT::default().present_modes(&modes);
        let second = vk::SwapchainPresentModesCreateInfoEXT::default().present_modes(&modes);
        first.p_next = (&second as *const vk::SwapchainPresentModesCreateInfoEXT<'_>).cast();
        assert!(
            unsafe {
                SwapchainMaintenanceTemplate::parse(
                    (&first as *const vk::SwapchainPresentModesCreateInfoEXT<'_>).cast(),
                )
            }
            .is_err()
        );

        let null_modes = vk::SwapchainPresentModesCreateInfoEXT {
            present_mode_count: 1,
            ..Default::default()
        };
        assert!(
            unsafe {
                SwapchainMaintenanceTemplate::parse(
                    (&null_modes as *const vk::SwapchainPresentModesCreateInfoEXT<'_>).cast(),
                )
            }
            .is_err()
        );

        let empty_modes = vk::SwapchainPresentModesCreateInfoEXT::default().present_modes(&[]);
        assert!(
            unsafe {
                SwapchainMaintenanceTemplate::parse(
                    (&empty_modes as *const vk::SwapchainPresentModesCreateInfoEXT<'_>).cast(),
                )
            }
            .is_err()
        );

        let unknown = vk::DeviceGroupSwapchainCreateInfoKHR::default();
        assert!(
            unsafe {
                SwapchainMaintenanceTemplate::parse(
                    (&unknown as *const vk::DeviceGroupSwapchainCreateInfoKHR<'_>).cast(),
                )
            }
            .is_err()
        );
    }

    #[test]
    fn only_deferred_memory_allocation_is_accepted_as_a_nonempty_flag() {
        assert!(supported_swapchain_flags(
            vk::SwapchainCreateFlagsKHR::empty()
        ));
        assert!(supported_swapchain_flags(
            vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT
        ));
        assert!(!supported_swapchain_flags(
            vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT
                | vk::SwapchainCreateFlagsKHR::PROTECTED
        ));
    }

    #[test]
    fn maintenance_create_support_requires_a_known_chain_and_supported_flags() {
        let plain = vk::SwapchainCreateInfoKHR::default();
        assert!(super::maintenance_create_supported(&plain));

        let deferred = vk::SwapchainCreateInfoKHR::default()
            .flags(vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT);
        assert!(super::maintenance_create_supported(&deferred));

        let mut unknown = vk::DeviceGroupSwapchainCreateInfoKHR::default();
        let unknown_info = vk::SwapchainCreateInfoKHR::default().push_next(&mut unknown);
        assert!(!super::maintenance_create_supported(&unknown_info));
    }

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
