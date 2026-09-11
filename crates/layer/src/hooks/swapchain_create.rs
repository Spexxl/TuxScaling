#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;

    fn base_info(flags: vk::SwapchainCreateFlagsKHR) -> vk::SwapchainCreateInfoKHR<'static> {
        vk::SwapchainCreateInfoKHR::default()
            .flags(flags)
            .image_format(vk::Format::B8G8R8A8_UNORM)
    }

    #[test]
    fn accepts_mutable_format_with_deferred_allocation() {
        let formats = [vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB];
        let mut list = vk::ImageFormatListCreateInfo::default().view_formats(&formats);
        let info = base_info(
            vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT
                | vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT,
        )
        .push_next(&mut list);

        let chain = unsafe { SwapchainCreateChain::from_create_info(&info) }.unwrap();

        assert_eq!(chain.view_formats.as_deref(), Some(formats.as_slice()));
    }

    #[test]
    fn view_formats_are_owned_and_not_borrowed_from_the_application() {
        let formats = [vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB];
        let mut list = vk::ImageFormatListCreateInfo::default().view_formats(&formats);
        let info = base_info(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT).push_next(&mut list);

        let chain = unsafe { SwapchainCreateChain::from_create_info(&info) }.unwrap();

        assert_eq!(chain.view_formats.as_deref(), Some(formats.as_slice()));
        assert_ne!(
            chain.view_formats.as_ref().unwrap().as_ptr(),
            formats.as_ptr()
        );
    }

    #[test]
    fn rebuilt_nodes_are_valid_when_the_input_order_is_reversed() {
        let formats = [vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB];
        let modes = [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX];
        let mut list = vk::ImageFormatListCreateInfo::default().view_formats(&formats);
        let mut modes_info =
            vk::SwapchainPresentModesCreateInfoEXT::default().present_modes(&modes);
        list.p_next = (&mut modes_info as *mut vk::SwapchainPresentModesCreateInfoEXT<'_>).cast();
        let info = base_info(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT).push_next(&mut list);
        let chain = unsafe { SwapchainCreateChain::from_create_info(&info) }.unwrap();

        let observed = chain.with_p_next(|head| unsafe {
            let first = &*head.cast::<vk::BaseInStructure<'_>>();
            assert_eq!(
                first.s_type,
                vk::StructureType::IMAGE_FORMAT_LIST_CREATE_INFO
            );
            let second = &*first.p_next.cast::<vk::BaseInStructure<'_>>();
            assert_eq!(
                second.s_type,
                vk::StructureType::SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT
            );
            true
        });

        assert!(observed);
    }

    #[test]
    fn mutable_format_requires_the_base_swapchain_format() {
        let formats = [vk::Format::B8G8R8A8_SRGB];
        let mut list = vk::ImageFormatListCreateInfo::default().view_formats(&formats);
        let info = base_info(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT).push_next(&mut list);

        assert_eq!(
            unsafe { SwapchainCreateChain::from_create_info(&info) },
            Err(SwapchainCompatibilityError::MissingBaseFormat {
                base: vk::Format::B8G8R8A8_UNORM,
            })
        );
    }

    #[test]
    fn image_format_list_without_mutable_format_is_preserved() {
        let formats = [vk::Format::B8G8R8A8_UNORM];
        let mut list = vk::ImageFormatListCreateInfo::default().view_formats(&formats);
        let info = base_info(vk::SwapchainCreateFlagsKHR::empty()).push_next(&mut list);

        let chain = unsafe { SwapchainCreateChain::from_create_info(&info) }.unwrap();

        assert_eq!(chain.view_formats.as_deref(), Some(formats.as_slice()));
    }

    #[test]
    fn invalid_flags_and_nodes_return_exact_typed_errors() {
        let protected = base_info(vk::SwapchainCreateFlagsKHR::PROTECTED);
        assert_eq!(
            unsafe { SwapchainCreateChain::from_create_info(&protected) },
            Err(SwapchainCompatibilityError::UnsupportedFlags {
                bits: vk::SwapchainCreateFlagsKHR::PROTECTED.as_raw(),
            })
        );

        let mut group = vk::DeviceGroupSwapchainCreateInfoKHR::default();
        let group_info = base_info(vk::SwapchainCreateFlagsKHR::empty()).push_next(&mut group);
        assert_eq!(
            unsafe { SwapchainCreateChain::from_create_info(&group_info) },
            Err(SwapchainCompatibilityError::UnsupportedPNext {
                structure_type: vk::StructureType::DEVICE_GROUP_SWAPCHAIN_CREATE_INFO_KHR,
            })
        );

        let null_list = vk::ImageFormatListCreateInfo {
            view_format_count: 1,
            ..Default::default()
        };
        let mut null_list = null_list;
        let null_info = base_info(vk::SwapchainCreateFlagsKHR::empty()).push_next(&mut null_list);
        assert_eq!(
            unsafe { SwapchainCreateChain::from_create_info(&null_info) },
            Err(SwapchainCompatibilityError::NullArray {
                structure_type: vk::StructureType::IMAGE_FORMAT_LIST_CREATE_INFO,
            })
        );
    }
}
use super::maintenance::PresentScaling;
use ash::vk;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SwapchainCreateChain {
    pub(crate) present_modes: Option<Vec<vk::PresentModeKHR>>,
    pub(crate) scaling: Option<PresentScaling>,
    pub(crate) view_formats: Option<Vec<vk::Format>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SwapchainCompatibilityError {
    MalformedExtensionNames,
    IncompatibleWsiExtension { name: Vec<u8>, reason: &'static str },
    UnsupportedFlags { bits: u32 },
    UnsupportedPNext { structure_type: vk::StructureType },
    DuplicatePNext { structure_type: vk::StructureType },
    NullArray { structure_type: vk::StructureType },
    EmptyArray { structure_type: vk::StructureType },
    MissingBaseFormat { base: vk::Format },
    UnsupportedImageContract(vk::Result),
}

impl SwapchainCompatibilityError {
    pub(crate) fn from_wsi(
        incompatible: &crate::hooks::wsi_compatibility::IncompatibleWsiExtension,
    ) -> Self {
        if incompatible.name.is_empty() {
            Self::MalformedExtensionNames
        } else {
            Self::IncompatibleWsiExtension {
                name: incompatible.name.clone(),
                reason: incompatible.reason,
            }
        }
    }

    pub(crate) fn reason(&self) -> String {
        match self {
            Self::MalformedExtensionNames => "malformed_extension_names".to_owned(),
            Self::IncompatibleWsiExtension { name, .. } => format!(
                "incompatible_wsi_extension extension={}",
                String::from_utf8_lossy(name)
            ),
            Self::UnsupportedFlags { bits } => format!("unsupported_flags bits=0x{bits:x}"),
            Self::UnsupportedPNext { structure_type } => {
                format!("unsupported_pnext stype={structure_type:?}")
            }
            Self::DuplicatePNext { structure_type } => {
                format!("duplicate_pnext stype={structure_type:?}")
            }
            Self::NullArray { structure_type } => format!("null_array stype={structure_type:?}"),
            Self::EmptyArray { structure_type } => {
                format!("empty_array stype={structure_type:?}")
            }
            Self::MissingBaseFormat { base } => format!("missing_base_format base={base:?}"),
            Self::UnsupportedImageContract(result) => {
                format!("unsupported_image_contract result={result:?}")
            }
        }
    }
}

pub(crate) fn supported_swapchain_flags(flags: vk::SwapchainCreateFlagsKHR) -> bool {
    let supported = vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT
        | vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT;
    flags.as_raw() & !supported.as_raw() == 0
}

impl SwapchainCreateChain {
    pub(crate) unsafe fn from_create_info(
        info: &vk::SwapchainCreateInfoKHR<'_>,
    ) -> Result<Self, SwapchainCompatibilityError> {
        let supported = vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT
            | vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT;
        let unsupported = info.flags.as_raw() & !supported.as_raw();
        if unsupported != 0 {
            return Err(SwapchainCompatibilityError::UnsupportedFlags {
                bits: info.flags.as_raw(),
            });
        }

        let mut chain = Self::default();
        let mut next = info.p_next;
        while !next.is_null() {
            let header = unsafe { &*next.cast::<vk::BaseInStructure<'_>>() };
            match header.s_type {
                vk::StructureType::SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT => {
                    if chain.present_modes.is_some() {
                        return Err(SwapchainCompatibilityError::DuplicatePNext {
                            structure_type: header.s_type,
                        });
                    }
                    let modes =
                        unsafe { &*next.cast::<vk::SwapchainPresentModesCreateInfoEXT<'_>>() };
                    if modes.present_mode_count == 0 {
                        return Err(SwapchainCompatibilityError::EmptyArray {
                            structure_type: header.s_type,
                        });
                    }
                    if modes.p_present_modes.is_null() {
                        return Err(SwapchainCompatibilityError::NullArray {
                            structure_type: header.s_type,
                        });
                    }
                    chain.present_modes = Some(
                        unsafe {
                            std::slice::from_raw_parts(
                                modes.p_present_modes,
                                modes.present_mode_count as usize,
                            )
                        }
                        .to_vec(),
                    );
                }
                vk::StructureType::SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_EXT => {
                    if chain.scaling.is_some() {
                        return Err(SwapchainCompatibilityError::DuplicatePNext {
                            structure_type: header.s_type,
                        });
                    }
                    let scaling =
                        unsafe { &*next.cast::<vk::SwapchainPresentScalingCreateInfoEXT<'_>>() };
                    chain.scaling = Some(PresentScaling {
                        behavior: scaling.scaling_behavior,
                        gravity_x: scaling.present_gravity_x,
                        gravity_y: scaling.present_gravity_y,
                    });
                }
                vk::StructureType::IMAGE_FORMAT_LIST_CREATE_INFO => {
                    if chain.view_formats.is_some() {
                        return Err(SwapchainCompatibilityError::DuplicatePNext {
                            structure_type: header.s_type,
                        });
                    }
                    let formats = unsafe { &*next.cast::<vk::ImageFormatListCreateInfo<'_>>() };
                    if formats.view_format_count == 0 {
                        return Err(SwapchainCompatibilityError::EmptyArray {
                            structure_type: header.s_type,
                        });
                    }
                    if formats.p_view_formats.is_null() {
                        return Err(SwapchainCompatibilityError::NullArray {
                            structure_type: header.s_type,
                        });
                    }
                    chain.view_formats = Some(
                        unsafe {
                            std::slice::from_raw_parts(
                                formats.p_view_formats,
                                formats.view_format_count as usize,
                            )
                        }
                        .to_vec(),
                    );
                }
                _ => {
                    return Err(SwapchainCompatibilityError::UnsupportedPNext {
                        structure_type: header.s_type,
                    });
                }
            }
            next = header.p_next.cast();
        }

        if info
            .flags
            .contains(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT)
            && !chain
                .view_formats
                .as_ref()
                .is_some_and(|formats| formats.contains(&info.image_format))
        {
            return Err(SwapchainCompatibilityError::MissingBaseFormat {
                base: info.image_format,
            });
        }
        Ok(chain)
    }

    pub(crate) fn with_p_next<R>(&self, invoke: impl FnOnce(*const std::ffi::c_void) -> R) -> R {
        let mut scaling = self.scaling.map(|scaling| {
            vk::SwapchainPresentScalingCreateInfoEXT::default()
                .scaling_behavior(scaling.behavior)
                .present_gravity_x(scaling.gravity_x)
                .present_gravity_y(scaling.gravity_y)
        });
        let mut modes = self
            .present_modes
            .as_ref()
            .map(|modes| vk::SwapchainPresentModesCreateInfoEXT::default().present_modes(modes));
        let mut formats = self
            .view_formats
            .as_ref()
            .map(|formats| vk::ImageFormatListCreateInfo::default().view_formats(formats));
        let mut next = std::ptr::null();
        if let Some(scaling) = scaling.as_mut() {
            scaling.p_next = next;
            next = (scaling as *mut vk::SwapchainPresentScalingCreateInfoEXT<'_>).cast();
        }
        if let Some(modes) = modes.as_mut() {
            modes.p_next = next;
            next = (modes as *mut vk::SwapchainPresentModesCreateInfoEXT<'_>).cast();
        }
        if let Some(formats) = formats.as_mut() {
            formats.p_next = next;
            next = (formats as *mut vk::ImageFormatListCreateInfo<'_>).cast();
        }
        invoke(next)
    }
}
