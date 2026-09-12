#![cfg(feature = "fidelityfx")]

use std::path::Path;
use tuxscaling_upscaler::BackendError;
use tuxscaling_upscaler::fidelityfx::{FfxVersion, FidelityFxLibrary};

#[test]
fn bundled_library_reports_fsr_3_1_4() {
    let library = FidelityFxLibrary::load_bundled().unwrap();
    assert_eq!(library.abi_version(), 2);
    assert_eq!(
        library.version(),
        FfxVersion {
            major: 3,
            minor: 1,
            patch: 4
        }
    );
}

#[test]
fn stale_abi_version_is_rejected_before_dispatch() {
    assert!(!FidelityFxLibrary::is_abi_compatible(1));
    assert!(FidelityFxLibrary::is_abi_compatible(2));
}

#[test]
fn missing_library_is_unavailable_without_panicking() {
    let result = FidelityFxLibrary::load(Some(Path::new(
        "/does/not/exist/libtuxscaling_fidelityfx_vk.so",
    )));
    assert!(matches!(result, Err(BackendError::Unavailable)));
}
