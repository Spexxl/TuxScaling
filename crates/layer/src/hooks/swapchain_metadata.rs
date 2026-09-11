use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OwnedXy {
    pub(crate) x: f32,
    pub(crate) y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OwnedHdrMetadata {
    pub(crate) display_primary_red: OwnedXy,
    pub(crate) display_primary_green: OwnedXy,
    pub(crate) display_primary_blue: OwnedXy,
    pub(crate) white_point: OwnedXy,
    pub(crate) max_luminance: f32,
    pub(crate) min_luminance: f32,
    pub(crate) max_content_light_level: f32,
    pub(crate) max_frame_average_light_level: f32,
}

impl OwnedHdrMetadata {
    pub(crate) fn from_vk(metadata: &vk::HdrMetadataEXT<'_>) -> Self {
        Self {
            display_primary_red: OwnedXy {
                x: metadata.display_primary_red.x,
                y: metadata.display_primary_red.y,
            },
            display_primary_green: OwnedXy {
                x: metadata.display_primary_green.x,
                y: metadata.display_primary_green.y,
            },
            display_primary_blue: OwnedXy {
                x: metadata.display_primary_blue.x,
                y: metadata.display_primary_blue.y,
            },
            white_point: OwnedXy {
                x: metadata.white_point.x,
                y: metadata.white_point.y,
            },
            max_luminance: metadata.max_luminance,
            min_luminance: metadata.min_luminance,
            max_content_light_level: metadata.max_content_light_level,
            max_frame_average_light_level: metadata.max_frame_average_light_level,
        }
    }

    pub(crate) fn to_vk(self) -> vk::HdrMetadataEXT<'static> {
        vk::HdrMetadataEXT::default()
            .display_primary_red(vk::XYColorEXT {
                x: self.display_primary_red.x,
                y: self.display_primary_red.y,
            })
            .display_primary_green(vk::XYColorEXT {
                x: self.display_primary_green.x,
                y: self.display_primary_green.y,
            })
            .display_primary_blue(vk::XYColorEXT {
                x: self.display_primary_blue.x,
                y: self.display_primary_blue.y,
            })
            .white_point(vk::XYColorEXT {
                x: self.white_point.x,
                y: self.white_point.y,
            })
            .max_luminance(self.max_luminance)
            .min_luminance(self.min_luminance)
            .max_content_light_level(self.max_content_light_level)
            .max_frame_average_light_level(self.max_frame_average_light_level)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetadataArrayError {
    NullSwapchains,
    NullMetadata,
    CountMismatch,
}

pub(crate) fn validate_metadata_inputs(
    count: u32,
    swapchains: *const vk::SwapchainKHR,
    metadata: *const vk::HdrMetadataEXT<'_>,
) -> Result<(), MetadataArrayError> {
    if count == 0 {
        return Ok(());
    }
    if swapchains.is_null() {
        return Err(MetadataArrayError::NullSwapchains);
    }
    if metadata.is_null() {
        return Err(MetadataArrayError::NullMetadata);
    }
    Ok(())
}

pub(crate) fn with_metadata_array<R>(
    logical: &[vk::SwapchainKHR],
    physical: &[vk::SwapchainKHR],
    metadata: &[OwnedHdrMetadata],
    invoke: impl FnOnce(&[vk::SwapchainKHR], &[vk::HdrMetadataEXT<'_>]) -> R,
) -> Result<R, MetadataArrayError> {
    if logical.len() != physical.len() || physical.len() != metadata.len() {
        return Err(MetadataArrayError::CountMismatch);
    }
    let rebuilt = metadata
        .iter()
        .copied()
        .map(OwnedHdrMetadata::to_vk)
        .collect::<Vec<_>>();
    Ok(invoke(physical, &rebuilt))
}

#[cfg(test)]
mod tests {
    use super::{MetadataArrayError, OwnedHdrMetadata, with_metadata_array};
    use ash::vk;
    use ash::vk::Handle;

    fn sample_metadata() -> vk::HdrMetadataEXT<'static> {
        vk::HdrMetadataEXT::default()
            .display_primary_red(vk::XYColorEXT { x: 0.64, y: 0.33 })
            .display_primary_green(vk::XYColorEXT { x: 0.3, y: 0.6 })
            .display_primary_blue(vk::XYColorEXT { x: 0.15, y: 0.06 })
            .white_point(vk::XYColorEXT {
                x: 0.3127,
                y: 0.329,
            })
            .max_luminance(1000.0)
            .min_luminance(0.01)
            .max_content_light_level(800.0)
            .max_frame_average_light_level(400.0)
    }

    #[test]
    fn metadata_is_owned_without_retaining_application_pointers() {
        let mut source = sample_metadata();
        source.p_next = 1usize as *const std::ffi::c_void;
        let owned = OwnedHdrMetadata::from_vk(&source);
        let rebuilt = owned.to_vk();

        assert_eq!(rebuilt.p_next, std::ptr::null());
        assert_eq!(rebuilt.display_primary_red.x, source.display_primary_red.x);
        assert_eq!(rebuilt.display_primary_red.y, source.display_primary_red.y);
        assert_eq!(rebuilt.max_luminance, source.max_luminance);
    }

    #[test]
    fn mixed_direct_and_virtual_arrays_preserve_order() {
        let handles = [
            vk::SwapchainKHR::from_raw(11),
            vk::SwapchainKHR::from_raw(12),
            vk::SwapchainKHR::from_raw(13),
        ];
        let physical = [
            vk::SwapchainKHR::from_raw(11),
            vk::SwapchainKHR::from_raw(112),
            vk::SwapchainKHR::from_raw(13),
        ];
        let metadata = [OwnedHdrMetadata::from_vk(&sample_metadata()); 3];
        let observed = with_metadata_array(&handles, &physical, &metadata, |handles, metadata| {
            (
                handles.to_vec(),
                metadata
                    .iter()
                    .map(|entry| entry.max_luminance)
                    .collect::<Vec<_>>(),
            )
        })
        .unwrap();

        assert_eq!(observed.0, physical);
        assert_eq!(observed.1, vec![1000.0; 3]);
    }

    #[test]
    fn metadata_count_and_null_inputs_have_exact_errors() {
        assert_eq!(
            super::validate_metadata_inputs(1, std::ptr::null(), std::ptr::null()),
            Err(MetadataArrayError::NullSwapchains)
        );
        let handle = vk::SwapchainKHR::from_raw(21);
        assert_eq!(
            super::validate_metadata_inputs(1, &handle, std::ptr::null()),
            Err(MetadataArrayError::NullMetadata)
        );
        assert_eq!(
            super::validate_metadata_inputs(0, std::ptr::null(), std::ptr::null()),
            Ok(())
        );
    }

    #[test]
    fn malformed_metadata_array_is_rejected_before_downstream() {
        let handles = [vk::SwapchainKHR::from_raw(31)];
        let metadata: [OwnedHdrMetadata; 0] = [];
        assert_eq!(
            with_metadata_array(&handles, &handles, &metadata, |_, _| ()).unwrap_err(),
            MetadataArrayError::CountMismatch
        );
    }
}
