#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;
    use std::ffi::CString;

    fn device_info<'a>(
        names: &'a [*const i8],
        p_next: *const std::ffi::c_void,
    ) -> vk::DeviceCreateInfo<'a> {
        vk::DeviceCreateInfo {
            p_next,
            ..vk::DeviceCreateInfo::default().enabled_extension_names(names)
        }
    }

    #[test]
    fn unrelated_device_extensions_do_not_disable_virtualization() {
        let names = [
            c"VK_KHR_swapchain".as_ptr(),
            c"VK_KHR_acceleration_structure".as_ptr(),
            c"VK_EXT_memory_budget".as_ptr(),
        ];
        let info = device_info(&names, std::ptr::null());

        let capabilities =
            unsafe { DeviceWsiCapabilities::from_create_info(&info, vk::API_VERSION_1_3) };

        assert!(capabilities.incompatible.is_none());
    }

    #[test]
    fn broad_dxvk_extension_fixture_is_compatible_when_all_enabled_wsi_contracts_are_translated() {
        let owned_names = include_str!("../../tests/fixtures/dxvk-device-extensions.txt")
            .lines()
            .map(|name| CString::new(name).unwrap())
            .collect::<Vec<_>>();
        let names = owned_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let mut maintenance = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
            .swapchain_maintenance1(true);
        let mut present_wait =
            vk::PhysicalDevicePresentWaitFeaturesKHR::default().present_wait(true);
        let mut present_id = vk::PhysicalDevicePresentIdFeaturesKHR {
            p_next: (&mut maintenance
                as *mut vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT<'_>)
                .cast(),
            ..Default::default()
        }
        .present_id(true);
        present_wait.p_next =
            (&mut present_id as *mut vk::PhysicalDevicePresentIdFeaturesKHR<'_>).cast();
        let info = device_info(
            &names,
            (&present_wait as *const vk::PhysicalDevicePresentWaitFeaturesKHR<'_>).cast(),
        );

        let capabilities =
            unsafe { DeviceWsiCapabilities::from_create_info(&info, vk::API_VERSION_1_3) };

        assert!(capabilities.incompatible.is_none());
        assert!(capabilities.mutable_format);
        assert!(capabilities.incremental_present);
        assert!(capabilities.present_id);
        assert!(capabilities.present_wait);
        assert!(capabilities.hdr_metadata);
        assert!(capabilities.display_timing);
        assert!(capabilities.display_control);
        assert!(capabilities.maintenance1.enabled);
    }

    #[test]
    fn known_untranslated_wsi_extension_selects_direct_presentation() {
        let names = [
            c"VK_KHR_swapchain".as_ptr(),
            c"VK_KHR_shared_presentable_image".as_ptr(),
        ];
        let info = device_info(&names, std::ptr::null());

        let capabilities =
            unsafe { DeviceWsiCapabilities::from_create_info(&info, vk::API_VERSION_1_3) };

        let incompatible = capabilities
            .incompatible
            .expect("incompatible WSI extension");
        assert_eq!(incompatible.name, b"VK_KHR_shared_presentable_image");
        assert_eq!(
            incompatible.reason,
            "shared presentable image semantics are not translated"
        );
    }

    #[test]
    fn malformed_extension_name_array_is_incompatible() {
        let names = [std::ptr::null()];
        let info = device_info(&names, std::ptr::null());

        let capabilities =
            unsafe { DeviceWsiCapabilities::from_create_info(&info, vk::API_VERSION_1_3) };

        let incompatible = capabilities
            .incompatible
            .expect("malformed extension array");
        assert!(incompatible.name.is_empty());
        assert_eq!(
            incompatible.reason,
            "device extension name array is malformed"
        );
    }

    #[test]
    fn maintenance_alias_and_feature_detection_remain_independent_of_other_extensions() {
        let names = [
            c"VK_KHR_swapchain".as_ptr(),
            c"VK_KHR_swapchain_maintenance1".as_ptr(),
            c"VK_EXT_descriptor_indexing".as_ptr(),
        ];
        let mut maintenance = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
            .swapchain_maintenance1(true);
        let info = device_info(
            &names,
            (&mut maintenance as *mut vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT<'_>)
                .cast(),
        );

        let capabilities =
            unsafe { DeviceWsiCapabilities::from_create_info(&info, vk::API_VERSION_1_3) };

        assert!(capabilities.incompatible.is_none());
        assert!(capabilities.maintenance1.enabled);
        assert_eq!(
            capabilities.maintenance1.flavor,
            Some(crate::hooks::maintenance::Maintenance1Flavor::Khr)
        );
    }

    #[test]
    fn present_id_and_wait_require_their_enabled_feature_bits() {
        let names = [
            c"VK_KHR_swapchain".as_ptr(),
            c"VK_KHR_present_id".as_ptr(),
            c"VK_KHR_present_wait".as_ptr(),
        ];
        let disabled = device_info(&names, std::ptr::null());
        let disabled_capabilities =
            unsafe { DeviceWsiCapabilities::from_create_info(&disabled, vk::API_VERSION_1_3) };
        assert!(!disabled_capabilities.present_id);
        assert!(!disabled_capabilities.present_wait);

        let mut present_wait =
            vk::PhysicalDevicePresentWaitFeaturesKHR::default().present_wait(true);
        let mut present_id = vk::PhysicalDevicePresentIdFeaturesKHR::default().present_id(true);
        present_wait.p_next =
            (&mut present_id as *mut vk::PhysicalDevicePresentIdFeaturesKHR<'_>).cast();
        let enabled = device_info(
            &names,
            (&present_wait as *const vk::PhysicalDevicePresentWaitFeaturesKHR<'_>).cast(),
        );
        let enabled_capabilities =
            unsafe { DeviceWsiCapabilities::from_create_info(&enabled, vk::API_VERSION_1_3) };
        assert!(enabled_capabilities.present_id);
        assert!(enabled_capabilities.present_wait);
    }

    #[test]
    fn core_image_format_list_support_uses_the_device_api_version() {
        let names = [c"VK_KHR_swapchain_mutable_format".as_ptr()];
        let info = device_info(&names, std::ptr::null());

        let pre_core =
            unsafe { DeviceWsiCapabilities::from_create_info(&info, vk::API_VERSION_1_1) };
        assert!(!pre_core.mutable_format);

        let core = unsafe { DeviceWsiCapabilities::from_create_info(&info, vk::API_VERSION_1_2) };
        assert!(core.mutable_format);
    }

    #[test]
    fn audited_registry_distinguishes_translated_incompatible_and_unrelated_extensions() {
        assert_eq!(
            extension_support(b"VK_KHR_swapchain"),
            Some(WsiExtensionSupport::Translated)
        );
        assert_eq!(
            extension_support(b"VK_NV_present_barrier"),
            Some(WsiExtensionSupport::Incompatible)
        );
        assert_eq!(extension_support(b"VK_EXT_memory_budget"), None);
    }

    #[test]
    fn every_pinned_swapchain_command_has_an_explicit_decision() {
        let expected = [
            (
                b"vkCreateSwapchainKHR".as_slice(),
                WsiCommandSupport::Translated,
            ),
            (
                b"vkDestroySwapchainKHR".as_slice(),
                WsiCommandSupport::Translated,
            ),
            (
                b"vkGetSwapchainImagesKHR".as_slice(),
                WsiCommandSupport::Translated,
            ),
            (
                b"vkAcquireNextImageKHR".as_slice(),
                WsiCommandSupport::Translated,
            ),
            (
                b"vkAcquireNextImage2KHR".as_slice(),
                WsiCommandSupport::Translated,
            ),
            (
                b"vkQueuePresentKHR".as_slice(),
                WsiCommandSupport::Translated,
            ),
            (
                b"vkCreateSharedSwapchainsKHR".as_slice(),
                WsiCommandSupport::Incompatible,
            ),
            (
                b"vkGetSwapchainCounterEXT".as_slice(),
                WsiCommandSupport::Translated,
            ),
            (
                b"vkGetRefreshCycleDurationGOOGLE".as_slice(),
                WsiCommandSupport::Translated,
            ),
            (
                b"vkGetPastPresentationTimingGOOGLE".as_slice(),
                WsiCommandSupport::Translated,
            ),
            (
                b"vkSetHdrMetadataEXT".as_slice(),
                WsiCommandSupport::Translated,
            ),
            (
                b"vkGetSwapchainStatusKHR".as_slice(),
                WsiCommandSupport::Translated,
            ),
            (
                b"vkWaitForPresentKHR".as_slice(),
                WsiCommandSupport::Translated,
            ),
            (
                b"vkAcquireFullScreenExclusiveModeEXT".as_slice(),
                WsiCommandSupport::Incompatible,
            ),
            (
                b"vkReleaseFullScreenExclusiveModeEXT".as_slice(),
                WsiCommandSupport::Incompatible,
            ),
            (
                b"vkSetLocalDimmingAMD".as_slice(),
                WsiCommandSupport::Incompatible,
            ),
            (
                b"vkSetLatencySleepModeNV".as_slice(),
                WsiCommandSupport::Incompatible,
            ),
            (
                b"vkLatencySleepNV".as_slice(),
                WsiCommandSupport::Incompatible,
            ),
            (
                b"vkSetLatencyMarkerNV".as_slice(),
                WsiCommandSupport::Incompatible,
            ),
            (
                b"vkGetLatencyTimingsNV".as_slice(),
                WsiCommandSupport::Incompatible,
            ),
        ];
        assert_eq!(wsi_command_inventory(), expected.as_slice());
        for (name, decision) in expected {
            assert_eq!(wsi_command_support(name), Some(decision), "{name:?}");
        }
    }

    #[test]
    fn non_substring_swapchain_commands_are_not_forwarded_from_virtual_tokens() {
        assert_eq!(
            wsi_command_support(b"vkSetLocalDimmingAMD"),
            Some(WsiCommandSupport::Incompatible)
        );
        assert_eq!(
            wsi_command_support(b"vkSetLatencyMarkerNV"),
            Some(WsiCommandSupport::Incompatible)
        );
    }

    #[test]
    fn every_untranslated_swapchain_extension_disables_promotion() {
        for name in [
            b"VK_AMD_display_native_hdr".as_slice(),
            b"VK_NV_low_latency2".as_slice(),
            b"VK_KHR_display_swapchain".as_slice(),
            b"VK_EXT_full_screen_exclusive".as_slice(),
        ] {
            let name = CString::new(name).unwrap();
            let names = [name.as_ptr()];
            let info = device_info(&names, std::ptr::null());
            assert!(
                unsafe { DeviceWsiCapabilities::from_create_info(&info, vk::API_VERSION_1_3) }
                    .incompatible
                    .is_some(),
                "{name:?}"
            );
        }
    }
}
use crate::hooks::maintenance::{Maintenance1Support, maintenance1_support};
use ash::vk;
use std::ffi::CStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WsiExtensionSupport {
    Translated,
    Incompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WsiCommandSupport {
    Translated,
    Incompatible,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IncompatibleWsiExtension {
    pub(crate) name: Vec<u8>,
    pub(crate) reason: &'static str,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DeviceWsiCapabilities {
    pub(crate) maintenance1: Maintenance1Support,
    pub(crate) mutable_format: bool,
    pub(crate) incremental_present: bool,
    pub(crate) present_id: bool,
    pub(crate) present_wait: bool,
    pub(crate) hdr_metadata: bool,
    pub(crate) display_timing: bool,
    pub(crate) display_control: bool,
    pub(crate) incompatible: Option<IncompatibleWsiExtension>,
}

fn extension_enabled(info: &vk::DeviceCreateInfo<'_>, expected: &[u8]) -> bool {
    if info.enabled_extension_count == 0 || info.pp_enabled_extension_names.is_null() {
        return false;
    }
    let names = unsafe {
        std::slice::from_raw_parts(
            info.pp_enabled_extension_names,
            info.enabled_extension_count as usize,
        )
    };
    names
        .iter()
        .any(|name| !name.is_null() && unsafe { CStr::from_ptr(*name) }.to_bytes() == expected)
}

fn incompatible_reason(name: &[u8]) -> Option<&'static str> {
    match name {
        b"VK_KHR_display_swapchain" => {
            Some("display swapchain sharing semantics are not translated")
        }
        b"VK_KHR_shared_presentable_image" => {
            Some("shared presentable image semantics are not translated")
        }
        b"VK_EXT_full_screen_exclusive" => {
            Some("external full-screen ownership semantics are not translated")
        }
        b"VK_NV_present_barrier" => Some("present barrier semantics are not translated"),
        b"VK_NV_low_latency2" => Some("swapchain latency markers are not translated"),
        b"VK_AMD_display_native_hdr" => {
            Some("native display HDR ownership semantics are not translated")
        }
        _ => None,
    }
}

const WSI_COMMAND_INVENTORY: &[(&[u8], WsiCommandSupport)] = &[
    (b"vkCreateSwapchainKHR", WsiCommandSupport::Translated),
    (b"vkDestroySwapchainKHR", WsiCommandSupport::Translated),
    (b"vkGetSwapchainImagesKHR", WsiCommandSupport::Translated),
    (b"vkAcquireNextImageKHR", WsiCommandSupport::Translated),
    (b"vkAcquireNextImage2KHR", WsiCommandSupport::Translated),
    (b"vkQueuePresentKHR", WsiCommandSupport::Translated),
    (
        b"vkCreateSharedSwapchainsKHR",
        WsiCommandSupport::Incompatible,
    ),
    (b"vkGetSwapchainCounterEXT", WsiCommandSupport::Translated),
    (
        b"vkGetRefreshCycleDurationGOOGLE",
        WsiCommandSupport::Translated,
    ),
    (
        b"vkGetPastPresentationTimingGOOGLE",
        WsiCommandSupport::Translated,
    ),
    (b"vkSetHdrMetadataEXT", WsiCommandSupport::Translated),
    (b"vkGetSwapchainStatusKHR", WsiCommandSupport::Translated),
    (b"vkWaitForPresentKHR", WsiCommandSupport::Translated),
    (
        b"vkAcquireFullScreenExclusiveModeEXT",
        WsiCommandSupport::Incompatible,
    ),
    (
        b"vkReleaseFullScreenExclusiveModeEXT",
        WsiCommandSupport::Incompatible,
    ),
    (b"vkSetLocalDimmingAMD", WsiCommandSupport::Incompatible),
    (b"vkSetLatencySleepModeNV", WsiCommandSupport::Incompatible),
    (b"vkLatencySleepNV", WsiCommandSupport::Incompatible),
    (b"vkSetLatencyMarkerNV", WsiCommandSupport::Incompatible),
    (b"vkGetLatencyTimingsNV", WsiCommandSupport::Incompatible),
];

#[cfg(test)]
pub(crate) fn wsi_command_inventory() -> &'static [(&'static [u8], WsiCommandSupport)] {
    WSI_COMMAND_INVENTORY
}

pub(crate) fn wsi_command_support(name: &[u8]) -> Option<WsiCommandSupport> {
    WSI_COMMAND_INVENTORY
        .iter()
        .find_map(|(command, support)| (*command == name).then_some(*support))
}

pub(crate) fn extension_support(name: &[u8]) -> Option<WsiExtensionSupport> {
    match name {
        b"VK_KHR_swapchain"
        | b"VK_KHR_swapchain_mutable_format"
        | b"VK_KHR_image_format_list"
        | b"VK_KHR_incremental_present"
        | b"VK_KHR_present_id"
        | b"VK_KHR_present_wait"
        | b"VK_EXT_hdr_metadata"
        | b"VK_GOOGLE_display_timing"
        | b"VK_EXT_display_control"
        | b"VK_EXT_swapchain_colorspace"
        | b"VK_EXT_swapchain_maintenance1"
        | b"VK_KHR_swapchain_maintenance1"
        | b"VK_EXT_surface_maintenance1"
        | b"VK_KHR_surface_maintenance1" => Some(WsiExtensionSupport::Translated),
        name if incompatible_reason(name).is_some() => Some(WsiExtensionSupport::Incompatible),
        _ => None,
    }
}

impl DeviceWsiCapabilities {
    pub(crate) unsafe fn from_create_info(
        info: &vk::DeviceCreateInfo<'_>,
        vulkan_api_version: u32,
    ) -> Self {
        let mut capabilities = Self {
            maintenance1: unsafe { maintenance1_support(info) },
            mutable_format: extension_enabled(
                info,
                vk::KHR_SWAPCHAIN_MUTABLE_FORMAT_NAME.to_bytes(),
            ) && (vulkan_api_version >= vk::API_VERSION_1_2
                || extension_enabled(info, vk::KHR_IMAGE_FORMAT_LIST_NAME.to_bytes())),
            incremental_present: extension_enabled(
                info,
                vk::KHR_INCREMENTAL_PRESENT_NAME.to_bytes(),
            ),
            present_id: extension_enabled(info, vk::KHR_PRESENT_ID_NAME.to_bytes())
                && unsafe {
                    feature_enabled(
                        info,
                        vk::StructureType::PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
                    )
                },
            present_wait: extension_enabled(info, vk::KHR_PRESENT_WAIT_NAME.to_bytes())
                && unsafe {
                    feature_enabled(
                        info,
                        vk::StructureType::PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR,
                    )
                },
            hdr_metadata: extension_enabled(info, vk::EXT_HDR_METADATA_NAME.to_bytes()),
            display_timing: extension_enabled(info, vk::GOOGLE_DISPLAY_TIMING_NAME.to_bytes()),
            display_control: extension_enabled(info, vk::EXT_DISPLAY_CONTROL_NAME.to_bytes()),
            incompatible: None,
        };

        if info.enabled_extension_count != 0 && info.pp_enabled_extension_names.is_null() {
            capabilities.incompatible = Some(IncompatibleWsiExtension {
                name: Vec::new(),
                reason: "device extension name array is malformed",
            });
            return capabilities;
        }

        if info.enabled_extension_count != 0 {
            let names = unsafe {
                std::slice::from_raw_parts(
                    info.pp_enabled_extension_names,
                    info.enabled_extension_count as usize,
                )
            };
            for name in names {
                if name.is_null() {
                    capabilities.incompatible = Some(IncompatibleWsiExtension {
                        name: Vec::new(),
                        reason: "device extension name array is malformed",
                    });
                    break;
                }
                let name = unsafe { CStr::from_ptr(*name) }.to_bytes();
                if matches!(
                    extension_support(name),
                    Some(WsiExtensionSupport::Incompatible)
                ) && let Some(reason) = incompatible_reason(name)
                {
                    capabilities.incompatible = Some(IncompatibleWsiExtension {
                        name: name.to_vec(),
                        reason,
                    });
                    break;
                }
            }
        }

        capabilities
    }
}

unsafe fn feature_enabled(
    info: &vk::DeviceCreateInfo<'_>,
    structure_type: vk::StructureType,
) -> bool {
    let mut next = info.p_next;
    while !next.is_null() {
        let header = unsafe { &*next.cast::<vk::BaseInStructure<'_>>() };
        match (header.s_type, structure_type) {
            (
                vk::StructureType::PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
                vk::StructureType::PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
            ) => {
                return unsafe {
                    (*next.cast::<vk::PhysicalDevicePresentIdFeaturesKHR<'_>>()).present_id != 0
                };
            }
            (
                vk::StructureType::PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR,
                vk::StructureType::PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR,
            ) => {
                return unsafe {
                    (*next.cast::<vk::PhysicalDevicePresentWaitFeaturesKHR<'_>>()).present_wait != 0
                };
            }
            _ => {}
        }
        next = header.p_next.cast();
    }
    false
}
