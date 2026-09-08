#![allow(dead_code)]

use super::ffi;
use crate::BackendError;
use libloading::Library;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const EXPECTED_VERSION: ffi::TuxFfxVersion = ffi::TuxFfxVersion {
    major: 3,
    minor: 1,
    patch: 4,
};
const LIBRARY_NAME: &str = "libtuxscaling_fidelityfx_vk.so";

pub type FfxVersion = ffi::TuxFfxVersion;

pub struct FidelityFxLibrary {
    _library: Library,
    version: FfxVersion,
    create_fn: ffi::TuxFfxCreateFn,
    dispatch_fn: ffi::TuxFfxDispatchFn,
    reset_fn: ffi::TuxFfxResetFn,
    destroy_fn: ffi::TuxFfxDestroyFn,
}

impl FidelityFxLibrary {
    pub fn load(explicit: Option<&Path>) -> Result<Arc<Self>, BackendError> {
        if let Some(path) = explicit {
            return Self::load_path(path);
        }

        let mut candidates = Vec::new();
        if let Some(path) = std::env::var_os("TUXSCALING_FIDELITYFX_LIBRARY") {
            candidates.push(PathBuf::from(path));
        }
        if let Some(path) = option_env!("TUXSCALING_BUILT_FIDELITYFX") {
            candidates.push(PathBuf::from(path));
        }
        candidates.extend(layer_adjacent_candidates());

        for path in candidates {
            if let Ok(library) = Self::load_path(&path) {
                return Ok(library);
            }
        }

        Self::load_path(Path::new(LIBRARY_NAME))
    }

    pub fn load_bundled() -> Result<Arc<Self>, BackendError> {
        Self::load(None)
    }

    pub fn version(&self) -> FfxVersion {
        self.version
    }

    fn load_path(path: &Path) -> Result<Arc<Self>, BackendError> {
        let library = unsafe { Library::new(path) }.map_err(|_| BackendError::Unavailable)?;
        let version_fn = load_symbol::<ffi::TuxFfxVersionFn>(&library, b"tux_ffx_version\0")?;
        let create_fn = load_symbol::<ffi::TuxFfxCreateFn>(&library, b"tux_ffx_create\0")?;
        let dispatch_fn = load_symbol::<ffi::TuxFfxDispatchFn>(&library, b"tux_ffx_dispatch\0")?;
        let reset_fn = load_symbol::<ffi::TuxFfxResetFn>(&library, b"tux_ffx_reset\0")?;
        let destroy_fn = load_symbol::<ffi::TuxFfxDestroyFn>(&library, b"tux_ffx_destroy\0")?;
        let version = unsafe { version_fn() };
        if version != EXPECTED_VERSION {
            return Err(BackendError::Unavailable);
        }

        Ok(Arc::new(Self {
            _library: library,
            version,
            create_fn,
            dispatch_fn,
            reset_fn,
            destroy_fn,
        }))
    }
}

pub(crate) struct NativeContext {
    pub(crate) library: Arc<FidelityFxLibrary>,
    pub(crate) handle: std::ptr::NonNull<ffi::TuxFfxContext>,
}

// The context is owned by one backend and is only moved between runtime
// owners; dispatch itself remains serialized by the backend contract.
unsafe impl Send for NativeContext {}

impl NativeContext {
    pub(crate) fn create(
        library: Arc<FidelityFxLibrary>,
        info: ffi::TuxFfxCreateInfo,
    ) -> Result<Self, BackendError> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { (library.create_fn)(&info, &mut handle) };
        if status != ffi::TUX_FFX_OK {
            return Err(map_status(status));
        }
        let handle = std::ptr::NonNull::new(handle).ok_or(BackendError::Internal(
            "FidelityFX returned a null context".into(),
        ))?;
        Ok(Self { library, handle })
    }

    pub(crate) unsafe fn dispatch(
        &mut self,
        info: &ffi::TuxFfxDispatchInfo,
    ) -> Result<(), BackendError> {
        let status = unsafe { (self.library.dispatch_fn)(self.handle.as_ptr(), info) };
        if status == ffi::TUX_FFX_OK {
            Ok(())
        } else {
            Err(map_status(status))
        }
    }

    pub(crate) fn reset(&mut self) -> Result<(), BackendError> {
        let status = unsafe { (self.library.reset_fn)(self.handle.as_ptr()) };
        if status == ffi::TUX_FFX_OK {
            Ok(())
        } else {
            Err(map_status(status))
        }
    }
}

impl Drop for NativeContext {
    fn drop(&mut self) {
        unsafe { (self.library.destroy_fn)(self.handle.as_ptr()) };
    }
}

fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, BackendError> {
    let symbol = unsafe { library.get::<T>(name) }.map_err(|_| BackendError::Unavailable)?;
    Ok(*symbol)
}

fn map_status(status: i32) -> BackendError {
    match status {
        ffi::TUX_FFX_INVALID_ARGUMENT => BackendError::InvalidMetadata("FidelityFX arguments"),
        ffi::TUX_FFX_UNSUPPORTED_VERSION => BackendError::Unavailable,
        ffi::TUX_FFX_CREATE_FAILED | ffi::TUX_FFX_DISPATCH_FAILED => {
            BackendError::Internal(format!("FidelityFX native status {status}"))
        }
        _ => BackendError::Internal(format!("unknown FidelityFX native status {status}")),
    }
}

fn layer_adjacent_candidates() -> Vec<PathBuf> {
    let mut directories = Vec::new();
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            directories.push(parent.to_path_buf());
        }
    }

    #[cfg(target_os = "linux")]
    if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
        for path in maps.lines().filter_map(mapped_library_path) {
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.contains("tuxscaling") && name.ends_with(".so"))
            {
                if let Some(parent) = path.parent() {
                    directories.push(parent.to_path_buf());
                }
            }
        }
    }

    directories
        .into_iter()
        .map(|directory| directory.join(LIBRARY_NAME))
        .collect()
}

#[cfg(target_os = "linux")]
fn mapped_library_path(line: &str) -> Option<PathBuf> {
    let path = line.split_whitespace().last()?;
    if path.starts_with('/') {
        Some(PathBuf::from(path))
    } else {
        None
    }
}
