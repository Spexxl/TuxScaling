use libloading::Library;
use std::os::raw::{c_char, c_int, c_long, c_uchar, c_uint, c_ulong};
use thiserror::Error;

pub const CRATE_NAME: &str = "tuxscaling-display";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub const fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    fn intersection_area(self, other: Self) -> u64 {
        let left = self.x.max(other.x);
        let top = self.y.max(other.y);
        let right = (self.x + self.width as i32).min(other.x + other.width as i32);
        let bottom = (self.y + self.height as i32).min(other.y + other.height as i32);
        u64::from((right - left).max(0) as u32) * u64::from((bottom - top).max(0) as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Monitor {
    pub rect: Rect,
}

impl Monitor {
    pub const fn new(rect: Rect) -> Self {
        Self { rect }
    }
}

pub fn select_monitor(window: Rect, monitors: &[Monitor]) -> Option<Monitor> {
    monitors
        .iter()
        .copied()
        .max_by_key(|monitor| window.intersection_area(monitor.rect))
        .filter(|monitor| window.intersection_area(monitor.rect) > 0)
}

#[repr(C)]
struct Display;

#[repr(C)]
struct XrrMonitorInfo {
    _name: c_ulong,
    _primary: c_int,
    _automatic: c_int,
    _output_count: c_int,
    x: c_int,
    y: c_int,
    width: c_int,
    height: c_int,
    _physical_width: c_int,
    _physical_height: c_int,
    _outputs: *mut c_ulong,
}

type OpenDisplay = unsafe extern "C" fn(*const c_char) -> *mut Display;
type CloseDisplay = unsafe extern "C" fn(*mut Display) -> c_int;
type GetGeometry = unsafe extern "C" fn(
    *mut Display,
    c_ulong,
    *mut c_ulong,
    *mut c_int,
    *mut c_int,
    *mut c_uint,
    *mut c_uint,
    *mut c_uint,
    *mut c_uint,
) -> c_int;
type ResizeWindow = unsafe extern "C" fn(*mut Display, c_ulong, c_uint, c_uint) -> c_int;
type Sync = unsafe extern "C" fn(*mut Display, c_int) -> c_int;
type InternAtom = unsafe extern "C" fn(*mut Display, *const c_char, c_int) -> c_ulong;
type GetWindowProperty = unsafe extern "C" fn(
    *mut Display,
    c_ulong,
    c_ulong,
    c_long,
    c_long,
    c_int,
    c_ulong,
    *mut c_ulong,
    *mut c_int,
    *mut c_ulong,
    *mut c_ulong,
    *mut *mut c_uchar,
) -> c_int;
type Free = unsafe extern "C" fn(*mut std::ffi::c_void) -> c_int;
type GetMonitors =
    unsafe extern "C" fn(*mut Display, c_ulong, c_int, *mut c_int) -> *mut XrrMonitorInfo;
type FreeMonitors = unsafe extern "C" fn(*mut XrrMonitorInfo);

#[derive(Debug, Error)]
pub enum DisplayError {
    #[error("X11 runtime is unavailable")]
    Library(#[from] libloading::Error),
    #[error("XOpenDisplay failed")]
    Unavailable,
    #[error("X11 window geometry is unavailable")]
    Geometry,
    #[error("X11 monitor discovery is unavailable")]
    Monitor,
}

pub struct X11Display {
    _x11: Library,
    _randr: Library,
    display: *mut Display,
    close_display: CloseDisplay,
    get_geometry: GetGeometry,
    resize_window: ResizeWindow,
    sync: Sync,
    intern_atom: InternAtom,
    get_window_property: GetWindowProperty,
    free: Free,
    get_monitors: GetMonitors,
    free_monitors: FreeMonitors,
}

impl X11Display {
    pub fn connect() -> Result<Self, DisplayError> {
        let x11 = unsafe { Library::new("libX11.so.6") }?;
        let randr = unsafe { Library::new("libXrandr.so.2") }?;
        let open_display = load::<OpenDisplay>(&x11, b"XOpenDisplay\0")?;
        let display = unsafe { open_display(std::ptr::null()) };
        if display.is_null() {
            return Err(DisplayError::Unavailable);
        }
        Ok(Self {
            close_display: load::<CloseDisplay>(&x11, b"XCloseDisplay\0")?,
            get_geometry: load::<GetGeometry>(&x11, b"XGetGeometry\0")?,
            resize_window: load::<ResizeWindow>(&x11, b"XResizeWindow\0")?,
            sync: load::<Sync>(&x11, b"XSync\0")?,
            intern_atom: load::<InternAtom>(&x11, b"XInternAtom\0")?,
            get_window_property: load::<GetWindowProperty>(&x11, b"XGetWindowProperty\0")?,
            free: load::<Free>(&x11, b"XFree\0")?,
            get_monitors: load::<GetMonitors>(&randr, b"XRRGetMonitors\0")?,
            free_monitors: load::<FreeMonitors>(&randr, b"XRRFreeMonitors\0")?,
            _x11: x11,
            _randr: randr,
            display,
        })
    }

    pub fn window_rect(&self, window: u64) -> Result<Rect, DisplayError> {
        let mut root = 0;
        let mut x = 0;
        let mut y = 0;
        let mut width = 0;
        let mut height = 0;
        let mut border = 0;
        let mut depth = 0;
        let status = unsafe {
            (self.get_geometry)(
                self.display,
                window as c_ulong,
                &mut root,
                &mut x,
                &mut y,
                &mut width,
                &mut height,
                &mut border,
                &mut depth,
            )
        };
        if status == 0 {
            return Err(DisplayError::Geometry);
        }
        Ok(Rect::new(x, y, width, height))
    }

    pub fn monitor_for_window(&self, window: u64) -> Result<Monitor, DisplayError> {
        let window_rect = self.window_rect(window)?;
        let mut count = 0;
        let monitors =
            unsafe { (self.get_monitors)(self.display, window as c_ulong, 1, &mut count) };
        if monitors.is_null() || count <= 0 {
            return Err(DisplayError::Monitor);
        }
        let result = unsafe {
            let monitors = std::slice::from_raw_parts(monitors, count as usize)
                .iter()
                .map(|monitor| {
                    Monitor::new(Rect::new(
                        monitor.x,
                        monitor.y,
                        monitor.width.max(0) as u32,
                        monitor.height.max(0) as u32,
                    ))
                })
                .collect::<Vec<_>>();
            select_monitor(window_rect, &monitors).ok_or(DisplayError::Monitor)
        };
        unsafe { (self.free_monitors)(monitors) };
        result
    }

    pub fn is_fullscreen(&self, window: u64) -> bool {
        let state = unsafe { (self.intern_atom)(self.display, c"_NET_WM_STATE".as_ptr(), 0) };
        let fullscreen =
            unsafe { (self.intern_atom)(self.display, c"_NET_WM_STATE_FULLSCREEN".as_ptr(), 0) };
        if state == 0 || fullscreen == 0 {
            return false;
        }
        let mut actual_type = 0;
        let mut actual_format = 0;
        let mut item_count = 0;
        let mut bytes_after = 0;
        let mut data = std::ptr::null_mut();
        let status = unsafe {
            (self.get_window_property)(
                self.display,
                window as c_ulong,
                state,
                0,
                1024,
                0,
                4,
                &mut actual_type,
                &mut actual_format,
                &mut item_count,
                &mut bytes_after,
                &mut data,
            )
        };
        if status != 0 || data.is_null() || actual_format != 32 {
            return false;
        }
        let atoms =
            unsafe { std::slice::from_raw_parts(data.cast::<c_ulong>(), item_count as usize) };
        let found = atoms.contains(&fullscreen);
        unsafe { (self.free)(data.cast()) };
        found
    }

    pub fn resize_window(&self, window: u64, monitor: Monitor) -> Result<(), DisplayError> {
        let width = monitor.rect.width;
        let height = monitor.rect.height;
        if width == 0 || height == 0 {
            return Err(DisplayError::Monitor);
        }
        unsafe {
            (self.resize_window)(self.display, window as c_ulong, width, height);
            (self.sync)(self.display, 0);
        }
        Ok(())
    }
}

impl Drop for X11Display {
    fn drop(&mut self) {
        if !self.display.is_null() {
            unsafe { (self.close_display)(self.display) };
        }
    }
}

fn load<T: Copy>(library: &Library, name: &[u8]) -> Result<T, libloading::Error> {
    Ok(*unsafe { library.get::<T>(name) }?)
}

#[cfg(test)]
mod tests {
    use super::{Monitor, Rect, select_monitor};

    #[test]
    fn selects_the_monitor_with_the_largest_window_intersection() {
        let monitors = [
            Monitor::new(Rect::new(0, 0, 1920, 1080)),
            Monitor::new(Rect::new(1920, 0, 2560, 1440)),
        ];
        let window = Rect::new(1800, 100, 1200, 800);

        assert_eq!(select_monitor(window, &monitors), Some(monitors[1]));
    }
}
