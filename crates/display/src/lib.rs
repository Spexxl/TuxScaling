use std::fmt::Display as FmtDisplay;

use thiserror::Error;
use x11rb::{
    connection::Connection,
    protocol::{
        randr::ConnectionExt as RandrConnectionExt,
        xproto::{AtomEnum, ConfigureWindowAux, ConnectionExt as XprotoConnectionExt, Window},
    },
    rust_connection::RustConnection,
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct X11Window {
    pub id: u64,
    pub rect: Rect,
    pub fullscreen: bool,
}

impl X11Window {
    pub const fn new(id: u64) -> Self {
        Self {
            id,
            rect: Rect::new(0, 0, 0, 0),
            fullscreen: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayTarget {
    pub window: X11Window,
    pub monitor: Monitor,
}

impl DisplayTarget {
    pub const fn new(window: X11Window, monitor: Monitor) -> Self {
        Self { window, monitor }
    }

    pub const fn output_extent(self) -> [u32; 2] {
        [self.monitor.rect.width, self.monitor.rect.height]
    }
}

pub fn select_monitor(window: Rect, monitors: &[Monitor]) -> Option<Monitor> {
    monitors
        .iter()
        .copied()
        .max_by_key(|monitor| window.intersection_area(monitor.rect))
        .filter(|monitor| window.intersection_area(monitor.rect) > 0)
}

#[derive(Debug, Error)]
pub enum DisplayError {
    #[error("X11 operation failed: {0}")]
    Operation(String),
    #[error("X11 window geometry is unavailable")]
    Geometry,
    #[error("X11 monitor discovery is unavailable")]
    Monitor,
}

fn operation<E: FmtDisplay>(error: E) -> DisplayError {
    DisplayError::Operation(error.to_string())
}

pub struct X11Display {
    connection: RustConnection,
    root: Window,
}

impl X11Display {
    pub fn connect() -> Result<Self, DisplayError> {
        let (connection, screen) = x11rb::connect(None).map_err(operation)?;
        let root = connection
            .setup()
            .roots
            .get(screen)
            .ok_or(DisplayError::Monitor)?
            .root;
        Ok(Self { connection, root })
    }

    pub fn describe_window(&self, window: u64) -> Result<X11Window, DisplayError> {
        Ok(X11Window {
            id: window,
            rect: self.window_rect(window)?,
            fullscreen: self.is_fullscreen(window),
        })
    }

    pub fn target_for_window(&self, window: u64) -> Result<DisplayTarget, DisplayError> {
        let window = self.describe_window(window)?;
        let monitor = self.monitor_for_window(window.id)?;
        Ok(DisplayTarget::new(window, monitor))
    }

    pub fn window_rect(&self, window: u64) -> Result<Rect, DisplayError> {
        let window = window as Window;
        let geometry = self
            .connection
            .get_geometry(window)
            .map_err(operation)?
            .reply()
            .map_err(operation)?;
        let translated = self
            .connection
            .translate_coordinates(window, self.root, 0, 0)
            .map_err(operation)?
            .reply()
            .map_err(operation)?;
        Ok(Rect::new(
            translated.dst_x.into(),
            translated.dst_y.into(),
            geometry.width.into(),
            geometry.height.into(),
        ))
    }

    pub fn monitor_for_window(&self, window: u64) -> Result<Monitor, DisplayError> {
        let window_rect = self.window_rect(window)?;
        let monitors = self
            .connection
            .randr_get_monitors(window as Window, true)
            .ok()
            .and_then(|cookie| cookie.reply().ok())
            .map(|reply| {
                reply
                    .monitors
                    .iter()
                    .map(|monitor| {
                        Monitor::new(Rect::new(
                            monitor.x.into(),
                            monitor.y.into(),
                            monitor.width.into(),
                            monitor.height.into(),
                        ))
                    })
                    .collect::<Vec<_>>()
            })
            .filter(|monitors| !monitors.is_empty())
            .unwrap_or_else(|| {
                self.connection
                    .get_geometry(self.root)
                    .ok()
                    .and_then(|cookie| cookie.reply().ok())
                    .map_or_else(Vec::new, |geometry| {
                        vec![Monitor::new(Rect::new(
                            0,
                            0,
                            geometry.width.into(),
                            geometry.height.into(),
                        ))]
                    })
            });
        select_monitor(window_rect, &monitors).ok_or(DisplayError::Monitor)
    }

    pub fn is_fullscreen(&self, window: u64) -> bool {
        let state = self
            .connection
            .intern_atom(false, b"_NET_WM_STATE")
            .ok()
            .and_then(|cookie| cookie.reply().ok())
            .map(|reply| reply.atom);
        let fullscreen = self
            .connection
            .intern_atom(false, b"_NET_WM_STATE_FULLSCREEN")
            .ok()
            .and_then(|cookie| cookie.reply().ok())
            .map(|reply| reply.atom);
        let (Some(state), Some(fullscreen)) = (state, fullscreen) else {
            return false;
        };
        let Ok(cookie) =
            self.connection
                .get_property(false, window as Window, state, AtomEnum::ATOM, 0, 1024)
        else {
            return false;
        };
        let Ok(property) = cookie.reply() else {
            return false;
        };
        property
            .value32()
            .is_some_and(|mut atoms| atoms.any(|atom| atom == fullscreen))
    }

    pub fn resize_window(&self, window: u64, monitor: Monitor) -> Result<(), DisplayError> {
        let rect = monitor.rect;
        if rect.width == 0 || rect.height == 0 {
            return Err(DisplayError::Monitor);
        }
        self.connection
            .configure_window(
                window as Window,
                &ConfigureWindowAux::new()
                    .x(rect.x)
                    .y(rect.y)
                    .width(rect.width)
                    .height(rect.height),
            )
            .map_err(operation)?;
        self.connection.flush().map_err(operation)
    }
}

#[cfg(test)]
mod tests {
    use super::{DisplayTarget, Monitor, Rect, X11Window, select_monitor};

    #[test]
    fn selects_the_monitor_with_the_largest_window_intersection() {
        let monitors = [
            Monitor::new(Rect::new(0, 0, 1920, 1080)),
            Monitor::new(Rect::new(1920, 0, 2560, 1440)),
        ];
        let window = Rect::new(1800, 100, 1200, 800);

        assert_eq!(select_monitor(window, &monitors), Some(monitors[1]));
    }

    #[test]
    fn domain_types_keep_window_and_output_target_explicit() {
        let window = X11Window::new(42);
        let monitor = Monitor::new(Rect::new(0, 0, 1920, 1080));
        let target = DisplayTarget::new(window, monitor);

        assert_eq!(target.window.id, 42);
        assert_eq!(target.output_extent(), [1920, 1080]);
    }
}
