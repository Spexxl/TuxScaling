use ash::vk;
use std::ffi::c_void;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PresentChainError {
    DuplicateStructure,
    CountMismatch,
    NullArray,
    UnknownStructure,
}

impl PresentChainError {
    pub(crate) const fn reason(self) -> &'static str {
        match self {
            Self::DuplicateStructure => "duplicate_structure",
            Self::CountMismatch => "count_mismatch",
            Self::NullArray => "null_array",
            Self::UnknownStructure => "unknown_structure",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct PresentChain<'a> {
    pub(crate) regions: Option<vk::PresentRegionsKHR<'a>>,
    pub(crate) ids: Option<vk::PresentIdKHR<'a>>,
    pub(crate) times: Option<vk::PresentTimesInfoGOOGLE<'a>>,
    pub(crate) fences: Option<vk::SwapchainPresentFenceInfoEXT<'a>>,
    pub(crate) modes: Option<vk::SwapchainPresentModeInfoEXT<'a>>,
}

impl<'a> PresentChain<'a> {
    pub(crate) unsafe fn parse(
        mut next: *const c_void,
        swapchain_count: u32,
    ) -> Result<Self, PresentChainError> {
        let mut chain = Self {
            regions: None,
            ids: None,
            times: None,
            fences: None,
            modes: None,
        };
        while !next.is_null() {
            let header = unsafe { &*next.cast::<vk::BaseInStructure<'_>>() };
            match header.s_type {
                vk::StructureType::PRESENT_REGIONS_KHR => {
                    if chain.regions.is_some() {
                        return Err(PresentChainError::DuplicateStructure);
                    }
                    let regions = unsafe { *next.cast::<vk::PresentRegionsKHR<'a>>() };
                    validate_count(regions.swapchain_count, swapchain_count)?;
                    if regions.swapchain_count != 0 && regions.p_regions.is_null() {
                        return Err(PresentChainError::NullArray);
                    }
                    if !regions.p_regions.is_null() {
                        let regions_slice = unsafe {
                            std::slice::from_raw_parts(
                                regions.p_regions,
                                regions.swapchain_count as usize,
                            )
                        };
                        if regions_slice.iter().any(|region| {
                            region.rectangle_count != 0 && region.p_rectangles.is_null()
                        }) {
                            return Err(PresentChainError::NullArray);
                        }
                    }
                    chain.regions = Some(regions);
                }
                vk::StructureType::PRESENT_ID_KHR => {
                    if chain.ids.is_some() {
                        return Err(PresentChainError::DuplicateStructure);
                    }
                    let ids = unsafe { *next.cast::<vk::PresentIdKHR<'a>>() };
                    validate_count(ids.swapchain_count, swapchain_count)?;
                    if ids.swapchain_count != 0 && ids.p_present_ids.is_null() {
                        return Err(PresentChainError::NullArray);
                    }
                    chain.ids = Some(ids);
                }
                vk::StructureType::PRESENT_TIMES_INFO_GOOGLE => {
                    if chain.times.is_some() {
                        return Err(PresentChainError::DuplicateStructure);
                    }
                    let times = unsafe { *next.cast::<vk::PresentTimesInfoGOOGLE<'a>>() };
                    validate_count(times.swapchain_count, swapchain_count)?;
                    if times.swapchain_count != 0 && times.p_times.is_null() {
                        return Err(PresentChainError::NullArray);
                    }
                    chain.times = Some(times);
                }
                vk::StructureType::SWAPCHAIN_PRESENT_FENCE_INFO_EXT => {
                    if chain.fences.is_some() {
                        return Err(PresentChainError::DuplicateStructure);
                    }
                    let fences = unsafe { *next.cast::<vk::SwapchainPresentFenceInfoEXT<'a>>() };
                    validate_count(fences.swapchain_count, swapchain_count)?;
                    if fences.swapchain_count != 0 && fences.p_fences.is_null() {
                        return Err(PresentChainError::NullArray);
                    }
                    chain.fences = Some(fences);
                }
                vk::StructureType::SWAPCHAIN_PRESENT_MODE_INFO_EXT => {
                    if chain.modes.is_some() {
                        return Err(PresentChainError::DuplicateStructure);
                    }
                    let modes = unsafe { *next.cast::<vk::SwapchainPresentModeInfoEXT<'a>>() };
                    validate_count(modes.swapchain_count, swapchain_count)?;
                    if modes.swapchain_count != 0 && modes.p_present_modes.is_null() {
                        return Err(PresentChainError::NullArray);
                    }
                    chain.modes = Some(modes);
                }
                _ => return Err(PresentChainError::UnknownStructure),
            }
            next = header.p_next.cast();
        }
        Ok(chain)
    }

    #[cfg(test)]
    pub(crate) unsafe fn fences_slice(&self) -> &'a [vk::Fence] {
        let Some(fences) = self.fences else {
            return &[];
        };
        if fences.swapchain_count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(fences.p_fences, fences.swapchain_count as usize) }
        }
    }

    #[cfg(test)]
    pub(crate) unsafe fn mode_slice(&self) -> &'a [vk::PresentModeKHR] {
        let Some(modes) = self.modes else {
            return &[];
        };
        if modes.swapchain_count == 0 {
            &[]
        } else {
            unsafe {
                std::slice::from_raw_parts(modes.p_present_modes, modes.swapchain_count as usize)
            }
        }
    }

    pub(crate) unsafe fn regions_slice(&self) -> &'a [vk::PresentRegionKHR<'a>] {
        let Some(regions) = self.regions else {
            return &[];
        };
        if regions.swapchain_count == 0 {
            &[]
        } else {
            unsafe {
                std::slice::from_raw_parts(regions.p_regions, regions.swapchain_count as usize)
            }
        }
    }

    pub(crate) fn with_rebuilt_chain<R>(
        &self,
        mut modified: vk::PresentInfoKHR<'_>,
        mapped_regions: Option<&MappedPresentRegions>,
        invoke: impl FnOnce(&vk::PresentInfoKHR<'_>) -> R,
    ) -> R {
        if self.is_empty() && mapped_regions.is_none() {
            return invoke(&modified);
        }

        let mut mapped_region_nodes = mapped_regions.map(|mapped| {
            mapped
                .rectangles
                .iter()
                .map(|rectangles| vk::PresentRegionKHR::default().rectangles(rectangles))
                .collect::<Vec<_>>()
        });
        let mut regions = self.regions;
        let mut ids = self.ids;
        let mut times = self.times;
        let mut fences = self.fences;
        let mut modes = self.modes;
        let mut next = std::ptr::null();

        if let Some(modes) = modes.as_mut() {
            modes.p_next = next;
            next = (modes as *const vk::SwapchainPresentModeInfoEXT<'_>).cast();
        }
        if let Some(fences) = fences.as_mut() {
            fences.p_next = next;
            next = (fences as *const vk::SwapchainPresentFenceInfoEXT<'_>).cast();
        }
        if let Some(times) = times.as_mut() {
            times.p_next = next;
            next = (times as *const vk::PresentTimesInfoGOOGLE<'_>).cast();
        }
        if let Some(ids) = ids.as_mut() {
            ids.p_next = next;
            next = (ids as *const vk::PresentIdKHR<'_>).cast();
        }
        if let Some(regions) = regions.as_mut() {
            if let Some(mapped_region_nodes) = mapped_region_nodes.as_mut()
                && mapped_region_nodes.len() == regions.swapchain_count as usize
            {
                regions.swapchain_count = mapped_region_nodes.len() as u32;
                regions.p_regions = mapped_region_nodes.as_ptr();
            }
            regions.p_next = next;
            next = (regions as *const vk::PresentRegionsKHR<'_>).cast();
        }
        modified.p_next = next;
        invoke(&modified)
    }

    fn is_empty(&self) -> bool {
        self.regions.is_none()
            && self.ids.is_none()
            && self.times.is_none()
            && self.fences.is_none()
            && self.modes.is_none()
    }
}

fn validate_count(actual: u32, expected: u32) -> Result<(), PresentChainError> {
    (actual == expected)
        .then_some(())
        .ok_or(PresentChainError::CountMismatch)
}

#[derive(Debug, Clone, Default)]
pub(crate) struct MappedPresentRegions {
    rectangles: Vec<Vec<vk::RectLayerKHR>>,
}

impl MappedPresentRegions {
    pub(crate) fn from_rectangles(rectangles: Vec<Vec<vk::RectLayerKHR>>) -> Self {
        Self { rectangles }
    }
}

#[cfg(test)]
mod tests {
    use super::{MappedPresentRegions, PresentChain};
    use ash::vk;
    use ash::vk::Handle;

    fn structure_types(mut next: *const std::ffi::c_void) -> Vec<vk::StructureType> {
        let mut result = Vec::new();
        while !next.is_null() {
            let header = unsafe { &*next.cast::<vk::BaseInStructure<'_>>() };
            result.push(header.s_type);
            next = header.p_next.cast();
        }
        result
    }

    #[test]
    fn combined_present_chain_preserves_supported_nodes_and_order() {
        let rectangles = [vk::RectLayerKHR::default()];
        let regions = [
            vk::PresentRegionKHR::default().rectangles(&rectangles),
            vk::PresentRegionKHR::default().rectangles(&rectangles),
        ];
        let mut region_info = vk::PresentRegionsKHR::default().regions(&regions);
        let ids = [11, 12];
        let mut id_info = vk::PresentIdKHR::default().present_ids(&ids);
        let times = [
            vk::PresentTimeGOOGLE {
                present_id: 11,
                desired_present_time: 100,
            },
            vk::PresentTimeGOOGLE {
                present_id: 12,
                desired_present_time: 200,
            },
        ];
        let mut times_info = vk::PresentTimesInfoGOOGLE::default().times(&times);
        let fences = [vk::Fence::from_raw(21), vk::Fence::from_raw(22)];
        let mut fence_info = vk::SwapchainPresentFenceInfoEXT::default().fences(&fences);
        let modes = [vk::PresentModeKHR::FIFO; 2];
        let mut mode_info = vk::SwapchainPresentModeInfoEXT::default().present_modes(&modes);

        fence_info.p_next = (&mut mode_info as *mut vk::SwapchainPresentModeInfoEXT<'_>).cast();
        times_info.p_next = (&mut fence_info as *mut vk::SwapchainPresentFenceInfoEXT<'_>).cast();
        id_info.p_next = (&mut times_info as *mut vk::PresentTimesInfoGOOGLE<'_>).cast();
        region_info.p_next = (&mut id_info as *mut vk::PresentIdKHR<'_>).cast();

        let parsed = unsafe {
            PresentChain::parse((&region_info as *const vk::PresentRegionsKHR<'_>).cast(), 2)
        }
        .unwrap();
        assert_eq!(parsed.fences.unwrap().swapchain_count, 2);
        assert_eq!(unsafe { parsed.fences_slice() }, &fences);
        assert_eq!(unsafe { parsed.mode_slice() }, &modes);

        let swapchains = [
            vk::SwapchainKHR::from_raw(31),
            vk::SwapchainKHR::from_raw(32),
        ];
        let indices = [0, 1];
        let modified = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indices);
        let mapped_rectangle = vk::RectLayerKHR {
            offset: vk::Offset2D { x: 3, y: 4 },
            extent: vk::Extent2D {
                width: 5,
                height: 6,
            },
            layer: 7,
        };
        let mapped_regions = MappedPresentRegions::from_rectangles(vec![
            vec![mapped_rectangle],
            vec![mapped_rectangle],
        ]);
        let downstream_result =
            parsed.with_rebuilt_chain(modified, Some(&mapped_regions), |info| {
                let downstream_swapchains = unsafe {
                    std::slice::from_raw_parts(info.p_swapchains, info.swapchain_count as usize)
                };
                let downstream_indices = unsafe {
                    std::slice::from_raw_parts(info.p_image_indices, info.swapchain_count as usize)
                };
                assert_eq!(downstream_swapchains, &swapchains);
                assert_eq!(downstream_indices, &indices);

                let rebuilt = unsafe { PresentChain::parse(info.p_next, info.swapchain_count) };
                let rebuilt = rebuilt.unwrap();
                assert_eq!(unsafe { rebuilt.fences_slice() }, &fences);
                assert_eq!(unsafe { rebuilt.mode_slice() }, &modes);
                let rebuilt_regions = unsafe { rebuilt.regions_slice() };
                assert_eq!(rebuilt_regions.len(), 2);
                for region in rebuilt_regions {
                    assert_eq!(region.rectangle_count, 1);
                    let rectangles = unsafe {
                        std::slice::from_raw_parts(
                            region.p_rectangles,
                            region.rectangle_count as usize,
                        )
                    };
                    assert_eq!(rectangles.len(), 1);
                    assert_eq!(rectangles[0].offset, mapped_rectangle.offset);
                    assert_eq!(rectangles[0].extent, mapped_rectangle.extent);
                    assert_eq!(rectangles[0].layer, mapped_rectangle.layer);
                }

                structure_types(info.p_next)
            });
        assert_eq!(
            downstream_result,
            vec![
                vk::StructureType::PRESENT_REGIONS_KHR,
                vk::StructureType::PRESENT_ID_KHR,
                vk::StructureType::PRESENT_TIMES_INFO_GOOGLE,
                vk::StructureType::SWAPCHAIN_PRESENT_FENCE_INFO_EXT,
                vk::StructureType::SWAPCHAIN_PRESENT_MODE_INFO_EXT,
            ]
        );
    }

    #[test]
    fn reverse_present_chain_order_is_accepted_and_duplicates_are_rejected() {
        let modes = [vk::PresentModeKHR::FIFO];
        let mut mode_info = vk::SwapchainPresentModeInfoEXT::default().present_modes(&modes);
        let regions = [vk::PresentRegionKHR::default()];
        let mut region_info = vk::PresentRegionsKHR::default().regions(&regions);
        mode_info.p_next = (&mut region_info as *mut vk::PresentRegionsKHR<'_>).cast();
        assert!(
            unsafe {
                PresentChain::parse(
                    (&mode_info as *const vk::SwapchainPresentModeInfoEXT<'_>).cast(),
                    1,
                )
            }
            .is_ok()
        );

        let second = vk::SwapchainPresentModeInfoEXT::default().present_modes(&modes);
        mode_info.p_next = (&second as *const vk::SwapchainPresentModeInfoEXT<'_>).cast();
        assert!(
            unsafe {
                PresentChain::parse(
                    (&mode_info as *const vk::SwapchainPresentModeInfoEXT<'_>).cast(),
                    1,
                )
            }
            .is_err()
        );
    }

    #[test]
    fn present_chain_rejects_count_mismatch_null_arrays_and_unknown_nodes() {
        let mismatched = vk::SwapchainPresentFenceInfoEXT {
            swapchain_count: 1,
            ..Default::default()
        };
        assert!(
            unsafe {
                PresentChain::parse(
                    (&mismatched as *const vk::SwapchainPresentFenceInfoEXT<'_>).cast(),
                    2,
                )
            }
            .is_err()
        );

        let null_array = vk::SwapchainPresentModeInfoEXT {
            swapchain_count: 1,
            ..Default::default()
        };
        assert!(
            unsafe {
                PresentChain::parse(
                    (&null_array as *const vk::SwapchainPresentModeInfoEXT<'_>).cast(),
                    1,
                )
            }
            .is_err()
        );

        let unknown = vk::DeviceGroupSwapchainCreateInfoKHR::default();
        assert!(
            unsafe {
                PresentChain::parse(
                    (&unknown as *const vk::DeviceGroupSwapchainCreateInfoKHR<'_>).cast(),
                    1,
                )
            }
            .is_err()
        );
    }
}
