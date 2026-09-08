#![allow(dead_code)]

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TuxFfxContext {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TuxFfxVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TuxFfxImage {
    pub image: u64,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub usage: u32,
    pub state: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TuxFfxCreateInfo {
    pub physical_device: u64,
    pub device: u64,
    pub get_device_proc_addr: u64,
    pub max_render_width: u32,
    pub max_render_height: u32,
    pub max_output_width: u32,
    pub max_output_height: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TuxFfxDispatchInfo {
    pub command_buffer: u64,
    pub color: TuxFfxImage,
    pub depth: TuxFfxImage,
    pub motion: TuxFfxImage,
    pub exposure: TuxFfxImage,
    pub reactive: TuxFfxImage,
    pub composition: TuxFfxImage,
    pub output: TuxFfxImage,
    pub jitter_x: f32,
    pub jitter_y: f32,
    pub motion_scale_x: f32,
    pub motion_scale_y: f32,
    pub frame_time_ms: f32,
    pub pre_exposure: f32,
    pub camera_near: f32,
    pub camera_far: f32,
    pub camera_fov_y: f32,
    pub view_space_to_meters: f32,
    pub render_width: u32,
    pub render_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub reset: u32,
}

pub type TuxFfxVersionFn = unsafe extern "C" fn() -> TuxFfxVersion;
pub type TuxFfxCreateFn =
    unsafe extern "C" fn(info: *const TuxFfxCreateInfo, context: *mut *mut TuxFfxContext) -> i32;
pub type TuxFfxDispatchFn =
    unsafe extern "C" fn(context: *mut TuxFfxContext, info: *const TuxFfxDispatchInfo) -> i32;
pub type TuxFfxResetFn = unsafe extern "C" fn(context: *mut TuxFfxContext) -> i32;
pub type TuxFfxDestroyFn = unsafe extern "C" fn(context: *mut TuxFfxContext);

pub const TUX_FFX_OK: i32 = 0;
pub const TUX_FFX_INVALID_ARGUMENT: i32 = 1;
pub const TUX_FFX_UNSUPPORTED_VERSION: i32 = 2;
pub const TUX_FFX_CREATE_FAILED: i32 = 3;
pub const TUX_FFX_DISPATCH_FAILED: i32 = 4;

pub const TUX_FFX_CREATE_HIGH_DYNAMIC_RANGE: u32 = 1 << 0;
pub const TUX_FFX_CREATE_DISPLAY_RESOLUTION_MOTION: u32 = 1 << 1;
pub const TUX_FFX_CREATE_MOTION_VECTORS_JITTER_CANCELLATION: u32 = 1 << 2;
pub const TUX_FFX_CREATE_DEPTH_INVERTED: u32 = 1 << 3;
pub const TUX_FFX_CREATE_DEPTH_INFINITE: u32 = 1 << 4;
pub const TUX_FFX_CREATE_AUTO_EXPOSURE: u32 = 1 << 5;
pub const TUX_FFX_CREATE_DYNAMIC_RESOLUTION: u32 = 1 << 6;
pub const TUX_FFX_CREATE_DEBUG_CHECKING: u32 = 1 << 7;
pub const TUX_FFX_CREATE_NON_LINEAR_COLORSPACE: u32 = 1 << 8;

pub const TUX_FFX_IMAGE_USAGE_READ_ONLY: u32 = 0;
pub const TUX_FFX_IMAGE_USAGE_UAV: u32 = 1 << 1;
pub const TUX_FFX_IMAGE_STATE_COMPUTE_READ: u32 = 1 << 2;
pub const TUX_FFX_IMAGE_STATE_UNORDERED_ACCESS: u32 = 1 << 1;

const _: () = {
    assert!(std::mem::size_of::<TuxFfxContext>() == 0);
    assert!(std::mem::size_of::<TuxFfxVersion>() == 12);
    assert!(std::mem::align_of::<TuxFfxVersion>() == 4);
    assert!(std::mem::size_of::<TuxFfxImage>() == 32);
    assert!(std::mem::align_of::<TuxFfxImage>() == 8);
    assert!(std::mem::size_of::<TuxFfxCreateInfo>() == 48);
    assert!(std::mem::align_of::<TuxFfxCreateInfo>() == 8);
    assert!(std::mem::size_of::<TuxFfxDispatchInfo>() == 296);
    assert!(std::mem::align_of::<TuxFfxDispatchInfo>() == 8);
};
