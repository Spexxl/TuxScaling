use std::{
    fmt::Display as FmtDisplay,
    time::{Duration, Instant},
};

use thiserror::Error;
use x11rb::{
    connection::Connection,
    protocol::{
        randr::ConnectionExt as RandrConnectionExt,
        xproto::{
            AtomEnum, ConfigureWindowAux, ConnectionExt as XprotoConnectionExt, PropMode, Window,
        },
        xproto::{ClientMessageEvent, EventMask},
    },
    rust_connection::RustConnection,
    wrapper::ConnectionExt as _,
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

    pub const fn is_valid(self) -> bool {
        self.width != 0 && self.height != 0
    }

    pub const fn extent(self) -> Extent {
        Extent::new(self.width, self.height)
    }

    fn contains_center(self, window: Self) -> bool {
        if !self.is_valid() {
            return false;
        }
        let center_x = i64::from(window.x) * 2 + i64::from(window.width);
        let center_y = i64::from(window.y) * 2 + i64::from(window.height);
        let left = i64::from(self.x) * 2;
        let top = i64::from(self.y) * 2;
        let right = (i64::from(self.x) + i64::from(self.width)) * 2;
        let bottom = (i64::from(self.y) + i64::from(self.height)) * 2;

        left <= center_x && center_x < right && top <= center_y && center_y < bottom
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    pub width: u32,
    pub height: u32,
}

impl Extent {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub const fn is_valid(self) -> bool {
        self.width != 0 && self.height != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceExtent {
    Fixed(Extent),
    Range { minimum: Extent, maximum: Extent },
}

impl SurfaceExtent {
    pub const fn fixed(extent: Extent) -> Self {
        Self::Fixed(extent)
    }

    fn accepts(self, target: Extent) -> bool {
        match self {
            Self::Fixed(extent) => extent == target,
            Self::Range { minimum, maximum } => {
                target.width >= minimum.width
                    && target.width <= maximum.width
                    && target.height >= minimum.height
                    && target.height <= maximum.height
            }
        }
    }
}

pub const NEGOTIATION_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentationState {
    Direct,
    Negotiating,
    Virtualized,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegotiationFailure {
    BorderlessRequest,
    DeadlineExpired,
    OutputRecreation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NegotiationStage {
    Direct,
    RequestingBorderless { target: Rect, deadline: Instant },
    WaitingForNativeExtent { target: Rect, deadline: Instant },
    RecreatingOutput { target: Rect, deadline: Instant },
    Active { target: Rect },
    Failed { reason: NegotiationFailure },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentationNegotiation {
    stage: NegotiationStage,
}

impl PresentationNegotiation {
    pub const fn direct() -> Self {
        Self {
            stage: NegotiationStage::Direct,
        }
    }

    pub fn request_borderless(&mut self, target: Rect, now: Instant) -> bool {
        if !target.is_valid() || !matches!(self.stage, NegotiationStage::Direct) {
            return false;
        }
        self.stage = NegotiationStage::RequestingBorderless {
            target,
            deadline: now + NEGOTIATION_TIMEOUT,
        };
        true
    }

    pub fn borderless_requested(&mut self, now: Instant) -> bool {
        let NegotiationStage::RequestingBorderless { target, deadline } = self.stage else {
            return false;
        };
        if now >= deadline {
            self.fail(NegotiationFailure::DeadlineExpired);
            return false;
        }
        self.stage = NegotiationStage::WaitingForNativeExtent { target, deadline };
        true
    }

    pub fn observe(
        &mut self,
        window: Rect,
        fullscreen: bool,
        surface: SurfaceExtent,
        now: Instant,
    ) -> bool {
        match self.stage {
            NegotiationStage::RequestingBorderless { deadline, .. } => {
                if now >= deadline {
                    self.fail(NegotiationFailure::DeadlineExpired);
                }
            }
            NegotiationStage::WaitingForNativeExtent { target, deadline } => {
                if now >= deadline {
                    self.fail(NegotiationFailure::DeadlineExpired);
                    return false;
                }
                if is_native(target, window, fullscreen, surface) {
                    self.stage = NegotiationStage::RecreatingOutput { target, deadline };
                }
            }
            NegotiationStage::RecreatingOutput { target, deadline } => {
                if now >= deadline {
                    self.fail(NegotiationFailure::DeadlineExpired);
                } else if !is_native(target, window, fullscreen, surface) {
                    self.stage = NegotiationStage::WaitingForNativeExtent { target, deadline };
                } else {
                    return true;
                }
            }
            NegotiationStage::Active { target }
                if !is_native(target, window, fullscreen, surface) =>
            {
                self.stage = NegotiationStage::WaitingForNativeExtent {
                    target,
                    deadline: now + NEGOTIATION_TIMEOUT,
                };
            }
            _ => {}
        }
        false
    }

    pub fn output_recreated(
        &mut self,
        window: Rect,
        fullscreen: bool,
        surface: SurfaceExtent,
        now: Instant,
    ) -> bool {
        let NegotiationStage::RecreatingOutput { target, deadline } = self.stage else {
            return false;
        };
        if now >= deadline {
            self.fail(NegotiationFailure::DeadlineExpired);
            return false;
        }
        if !is_native(target, window, fullscreen, surface) {
            self.stage = NegotiationStage::WaitingForNativeExtent { target, deadline };
            return false;
        }
        self.stage = NegotiationStage::Active { target };
        true
    }

    pub fn fail(&mut self, reason: NegotiationFailure) {
        if !matches!(self.stage, NegotiationStage::Failed { .. }) {
            self.stage = NegotiationStage::Failed { reason };
        }
    }

    pub const fn public_state(self) -> PresentationState {
        match self.stage {
            NegotiationStage::Direct => PresentationState::Direct,
            NegotiationStage::RequestingBorderless { .. }
            | NegotiationStage::WaitingForNativeExtent { .. }
            | NegotiationStage::RecreatingOutput { .. } => PresentationState::Negotiating,
            NegotiationStage::Active { .. } => PresentationState::Virtualized,
            NegotiationStage::Failed { .. } => PresentationState::Failed,
        }
    }

    pub const fn failure(self) -> Option<NegotiationFailure> {
        match self.stage {
            NegotiationStage::Failed { reason } => Some(reason),
            _ => None,
        }
    }

    pub const fn output_recreation_ready(self) -> bool {
        matches!(self.stage, NegotiationStage::RecreatingOutput { .. })
    }

    pub fn native_observation_is_current(
        self,
        window: Rect,
        fullscreen: bool,
        surface: SurfaceExtent,
        now: Instant,
    ) -> bool {
        let NegotiationStage::RecreatingOutput { target, deadline } = self.stage else {
            return false;
        };
        now < deadline && is_native(target, window, fullscreen, surface)
    }
}

fn is_native(target: Rect, window: Rect, _fullscreen: bool, surface: SurfaceExtent) -> bool {
    // Exact geometry match is borderless fullscreen by observation: the
    // EWMH flag is not required. Wine-owned windows never retain an
    // externally requested _NET_WM_STATE_FULLSCREEN, and borderless-windowed
    // native windows may not carry it either. The flag parameter is kept so
    // existing call sites stay unchanged.
    window == target && surface.accepts(target.extent())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowSnapshot {
    pub rect: Rect,
    pub fullscreen: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BorderlessLease {
    pub window: u64,
    pub original: WindowSnapshot,
    pub monitor: Monitor,
}

impl BorderlessLease {
    pub const fn added_fullscreen(self) -> bool {
        !self.original.fullscreen
    }

    pub const fn should_remove_fullscreen(self, current_fullscreen: bool) -> bool {
        self.added_fullscreen() && current_fullscreen
    }
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
    pub fn is_fullscreen(self) -> bool {
        self.window.fullscreen || self.window.rect == self.monitor.rect
    }

    pub const fn new(window: X11Window, monitor: Monitor) -> Self {
        Self { window, monitor }
    }

    pub const fn output_extent(self) -> [u32; 2] {
        [self.monitor.rect.width, self.monitor.rect.height]
    }
}

pub fn select_monitor(window: Rect, monitors: &[Monitor]) -> Option<Monitor> {
    if !window.is_valid() {
        return None;
    }
    monitors
        .iter()
        .copied()
        .find(|monitor| monitor.rect.contains_center(window))
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

/// Translates a desired root-relative position into the parent-relative
/// coordinates that `configure_window` expects. Wine reparents game windows
/// into frame windows, so root coordinates must be shifted by the parent
/// origin; for directly parented windows the origin is (0, 0).
fn parent_relative_position(desired: (i32, i32), parent_origin: (i32, i32)) -> (i32, i32) {
    (
        desired.0.saturating_sub(parent_origin.0),
        desired.1.saturating_sub(parent_origin.1),
    )
}

fn select_window_for_pid(windows: &[(u64, Option<u32>, Rect)], pid: u32) -> Option<u64> {
    windows
        .iter()
        .filter(|(_, window_pid, rect)| *window_pid == Some(pid) && rect.is_valid())
        .max_by_key(|(window, _, rect)| {
            (
                u64::from(rect.width) * u64::from(rect.height),
                // Tie-break toward the smallest window id for determinism.
                std::cmp::Reverse(*window),
            )
        })
        .map(|(window, _, _)| *window)
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

    pub fn window_for_process(&self, pid: u32) -> Result<u64, DisplayError> {
        let pid_atom = self
            .connection
            .intern_atom(false, b"_NET_WM_PID")
            .map_err(operation)?
            .reply()
            .map_err(operation)?
            .atom;
        let mut pending = vec![self.root];
        let mut windows = Vec::new();
        while let Some(parent) = pending.pop() {
            let children = self
                .connection
                .query_tree(parent)
                .map_err(operation)?
                .reply()
                .map_err(operation)?
                .children;
            for window in children {
                let window_pid = self
                    .connection
                    .get_property(false, window, pid_atom, AtomEnum::CARDINAL, 0, 1)
                    .ok()
                    .and_then(|cookie| cookie.reply().ok())
                    .and_then(|property| property.value32()?.next());
                let rect = self
                    .window_rect(window as u64)
                    .unwrap_or(Rect::new(0, 0, 0, 0));
                windows.push((window as u64, window_pid, rect));
                pending.push(window);
            }
        }
        select_window_for_pid(&windows, pid).ok_or(DisplayError::Geometry)
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
            .randr_get_monitors(self.root, true)
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

    fn set_fullscreen(&self, window: u64, enabled: bool) -> Result<(), DisplayError> {
        let state = self
            .connection
            .intern_atom(false, b"_NET_WM_STATE")
            .map_err(operation)?
            .reply()
            .map_err(operation)?
            .atom;
        let fullscreen = self
            .connection
            .intern_atom(false, b"_NET_WM_STATE_FULLSCREEN")
            .map_err(operation)?
            .reply()
            .map_err(operation)?
            .atom;
        self.connection
            .send_event(
                false,
                self.root,
                EventMask::SUBSTRUCTURE_REDIRECT | EventMask::SUBSTRUCTURE_NOTIFY,
                ClientMessageEvent::new(
                    32,
                    window as Window,
                    state,
                    [u32::from(enabled), fullscreen, 0, 0, 0],
                ),
            )
            .map_err(operation)?
            .check()
            .map_err(operation)?;
        self.connection.flush().map_err(operation)
    }

    fn reconcile_fullscreen_property(
        &self,
        window: u64,
        enabled: bool,
    ) -> Result<(), DisplayError> {
        let state = self
            .connection
            .intern_atom(false, b"_NET_WM_STATE")
            .map_err(operation)?
            .reply()
            .map_err(operation)?
            .atom;
        let fullscreen = self
            .connection
            .intern_atom(false, b"_NET_WM_STATE_FULLSCREEN")
            .map_err(operation)?
            .reply()
            .map_err(operation)?
            .atom;
        let atoms = self
            .connection
            .get_property(false, window as Window, state, AtomEnum::ATOM, 0, 1024)
            .map_err(operation)?
            .reply()
            .map_err(operation)?
            .value32()
            .map(|values| {
                values
                    .filter(|atom| *atom != fullscreen)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut atoms = atoms;
        if enabled {
            atoms.push(fullscreen);
        }
        self.connection
            .change_property32(
                PropMode::REPLACE,
                window as Window,
                state,
                AtomEnum::ATOM,
                &atoms,
            )
            .map_err(operation)?
            .check()
            .map_err(operation)?;
        self.connection.flush().map_err(operation)
    }

    fn parent_origin(&self, window: u64) -> Result<(i32, i32), DisplayError> {
        let parent = self
            .connection
            .query_tree(window as Window)
            .map_err(operation)?
            .reply()
            .map_err(operation)?
            .parent;
        if parent == self.root {
            return Ok((0, 0));
        }
        let translated = self
            .connection
            .translate_coordinates(parent, self.root, 0, 0)
            .map_err(operation)?
            .reply()
            .map_err(operation)?;
        Ok((translated.dst_x.into(), translated.dst_y.into()))
    }

    fn configure_rect(&self, window: u64, rect: Rect) -> Result<(), DisplayError> {
        if !rect.is_valid() {
            return Err(DisplayError::Geometry);
        }
        let parent_origin = self.parent_origin(window)?;
        let (x, y) = parent_relative_position((rect.x, rect.y), parent_origin);
        self.connection
            .configure_window(
                window as Window,
                &ConfigureWindowAux::new()
                    .x(x)
                    .y(y)
                    .width(rect.width)
                    .height(rect.height),
            )
            .map_err(operation)?
            .check()
            .map_err(operation)?;
        self.connection.flush().map_err(operation)
    }

    pub fn promote_borderless(&self, window: u64) -> Result<BorderlessLease, DisplayError> {
        let current = self.describe_window(window)?;
        if !current.rect.is_valid() {
            return Err(DisplayError::Geometry);
        }
        let monitor = self.monitor_for_window(window)?;
        if !monitor.rect.is_valid() {
            return Err(DisplayError::Monitor);
        }
        let lease = BorderlessLease {
            window,
            original: WindowSnapshot {
                rect: current.rect,
                fullscreen: current.fullscreen,
            },
            monitor,
        };
        if lease.added_fullscreen() {
            self.set_fullscreen(window, true)?;
        }
        if let Err(error) = self.configure_rect(window, monitor.rect) {
            if lease.added_fullscreen() {
                let _ = self.set_fullscreen(window, false);
            }
            return Err(error);
        }
        Ok(lease)
    }

    pub fn restore(&self, lease: BorderlessLease) -> Result<(), DisplayError> {
        let current_fullscreen = self.is_fullscreen(lease.window);
        if lease.should_remove_fullscreen(current_fullscreen) {
            self.set_fullscreen(lease.window, false)?;
        }
        self.configure_rect(lease.window, lease.original.rect)?;
        let restored_fullscreen = self.is_fullscreen(lease.window);
        if lease.original.fullscreen && !restored_fullscreen {
            self.set_fullscreen(lease.window, true)?;
            if !self.is_fullscreen(lease.window) {
                self.reconcile_fullscreen_property(lease.window, true)?;
            }
        } else if lease.added_fullscreen() && restored_fullscreen {
            self.set_fullscreen(lease.window, false)?;
            if self.is_fullscreen(lease.window) {
                self.reconcile_fullscreen_property(lease.window, false)?;
            }
        }
        Ok(())
    }

    pub fn resize_window(&self, window: u64, monitor: Monitor) -> Result<(), DisplayError> {
        if !monitor.rect.is_valid() {
            return Err(DisplayError::Monitor);
        }
        self.configure_rect(window, monitor.rect)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BorderlessLease, DisplayTarget, Extent, Monitor, NegotiationFailure,
        PresentationNegotiation, PresentationState, Rect, SurfaceExtent, WindowSnapshot, X11Window,
        parent_relative_position, select_monitor, select_window_for_pid,
    };
    use std::time::{Duration, Instant};

    #[test]
    fn selects_the_monitor_containing_the_window_center() {
        let monitors = [
            Monitor::new(Rect::new(0, 0, 1920, 1080)),
            Monitor::new(Rect::new(1920, 0, 2560, 1440)),
        ];
        let window = Rect::new(1800, 100, 1200, 800);

        assert_eq!(select_monitor(window, &monitors), Some(monitors[1]));
    }

    #[test]
    fn selects_the_largest_valid_window_for_process() {
        let windows = [
            (41, Some(7), Rect::new(0, 0, 1280, 720)),
            (42, Some(9), Rect::new(0, 0, 640, 480)),
            (43, Some(9), Rect::new(0, 0, 1280, 720)),
            (44, Some(9), Rect::new(0, 0, 0, 0)),
        ];

        assert_eq!(select_window_for_pid(&windows, 9), Some(43));
        assert_eq!(select_window_for_pid(&windows, 7), Some(41));
        assert_eq!(select_window_for_pid(&windows, 10), None);
    }

    #[test]
    fn reparented_windows_configure_relative_to_their_parent_origin() {
        assert_eq!(parent_relative_position((652, 1440), (0, 0)), (652, 1440));
        assert_eq!(parent_relative_position((652, 1440), (652, 1479)), (0, -39));
        assert_eq!(parent_relative_position((-1920, 0), (-1920, 0)), (0, 0));
    }

    #[test]
    fn selects_center_monitor_even_when_another_monitor_has_more_intersection() {
        let monitors = [
            Monitor::new(Rect::new(0, 0, 1920, 1080)),
            Monitor::new(Rect::new(1920, 0, 160, 200)),
        ];
        let window = Rect::new(0, 0, 4000, 100);

        assert_eq!(select_monitor(window, &monitors), Some(monitors[1]));
    }

    #[test]
    fn selects_center_monitor_with_negative_origins() {
        let monitors = [
            Monitor::new(Rect::new(-1920, 0, 1920, 1080)),
            Monitor::new(Rect::new(0, 0, 1920, 1080)),
        ];
        let window = Rect::new(-1000, 100, 800, 600);

        assert_eq!(select_monitor(window, &monitors), Some(monitors[0]));
    }

    #[test]
    fn negotiation_requires_exact_geometry_and_surface_extent_before_activation() {
        let target = Rect::new(0, 0, 1920, 1080);
        let mut negotiation = PresentationNegotiation::direct();
        let started = Instant::now();

        negotiation.request_borderless(target, started);
        negotiation.borderless_requested(started);
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);
        assert!(!negotiation.observe(
            Rect::new(0, 0, 1920, 1040),
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1040)),
            started + Duration::from_secs(1),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);

        assert!(!negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(2),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);
        assert!(negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(3),
        ));

        assert!(negotiation.output_recreated(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(3),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Virtualized);
    }

    #[test]
    fn native_output_waits_for_stable_geometry_before_recreation() {
        let target = Rect::new(0, 0, 1920, 1080);
        let mut negotiation = PresentationNegotiation::direct();
        let started = Instant::now();

        negotiation.request_borderless(target, started);
        negotiation.borderless_requested(started);

        assert!(!negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(1),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);
        assert!(negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(2),
        ));
        assert!(negotiation.output_recreated(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(2),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Virtualized);
    }

    #[test]
    fn negotiation_times_out_after_five_monotonic_seconds() {
        let target = Rect::new(0, 0, 1920, 1080);
        let mut negotiation = PresentationNegotiation::direct();
        let started = Instant::now();

        negotiation.request_borderless(target, started);
        negotiation.borderless_requested(started);
        assert!(!negotiation.observe(
            Rect::new(0, 0, 1920, 1040),
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1040)),
            started + Duration::from_secs(5),
        ));

        assert_eq!(negotiation.public_state(), PresentationState::Failed);
        assert_eq!(
            negotiation.failure(),
            Some(NegotiationFailure::DeadlineExpired)
        );
    }

    #[test]
    fn negotiation_times_out_while_borderless_request_is_unacknowledged() {
        let target = Rect::new(0, 0, 1920, 1080);
        let mut negotiation = PresentationNegotiation::direct();
        let started = Instant::now();

        negotiation.request_borderless(target, started);
        assert!(!negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(5),
        ));

        assert_eq!(negotiation.public_state(), PresentationState::Failed);
        assert_eq!(
            negotiation.failure(),
            Some(NegotiationFailure::DeadlineExpired)
        );
    }

    #[test]
    fn negotiation_rejects_exact_geometry_with_rejected_surface_extent() {
        let target = Rect::new(0, 0, 1920, 1080);
        let mut negotiation = PresentationNegotiation::direct();
        let started = Instant::now();

        negotiation.request_borderless(target, started);
        negotiation.borderless_requested(started);

        assert!(!negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1040)),
            started + Duration::from_secs(1),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);
    }

    #[test]
    fn negotiation_accepts_a_surface_extent_range_containing_the_target() {
        let target = Rect::new(0, 0, 1920, 1080);
        let mut negotiation = PresentationNegotiation::direct();
        let started = Instant::now();

        negotiation.request_borderless(target, started);
        negotiation.borderless_requested(started);

        assert!(!negotiation.observe(
            target,
            true,
            SurfaceExtent::Range {
                minimum: Extent::new(1280, 720),
                maximum: Extent::new(3840, 2160),
            },
            started + Duration::from_secs(1),
        ));
        assert!(negotiation.observe(
            target,
            true,
            SurfaceExtent::Range {
                minimum: Extent::new(1280, 720),
                maximum: Extent::new(3840, 2160),
            },
            started + Duration::from_secs(2),
        ));
        assert!(negotiation.output_recreated(
            target,
            true,
            SurfaceExtent::Range {
                minimum: Extent::new(1280, 720),
                maximum: Extent::new(3840, 2160),
            },
            started + Duration::from_secs(2),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Virtualized);
    }

    #[test]
    fn exact_geometry_counts_as_native_without_the_ewmh_flag() {
        let target = Rect::new(0, 0, 1920, 1080);
        let mut negotiation = PresentationNegotiation::direct();
        let started = Instant::now();

        negotiation.request_borderless(target, started);
        negotiation.borderless_requested(started);
        // Wine-owned windows never retain an externally requested
        // _NET_WM_STATE_FULLSCREEN; geometry equality alone must suffice.
        assert!(!negotiation.observe(
            target,
            false,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(1),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);

        assert!(negotiation.observe(
            target,
            false,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(2),
        ));
        assert!(negotiation.output_recreated(
            target,
            false,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(2),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Virtualized);
    }

    #[test]
    fn negotiation_revalidates_geometry_and_extent_during_recreation() {
        let target = Rect::new(-1920, 0, 1920, 1080);
        let drifted = Rect::new(-1920, 0, 1920, 1040);
        let mut negotiation = PresentationNegotiation::direct();
        let started = Instant::now();

        negotiation.request_borderless(target, started);
        negotiation.borderless_requested(started);
        assert!(!negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(1),
        ));

        assert!(!negotiation.observe(
            drifted,
            true,
            SurfaceExtent::fixed(drifted.extent()),
            started + Duration::from_secs(2),
        ));
        assert!(!negotiation.output_recreated(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(2),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);

        assert!(!negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(3),
        ));
        assert!(negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(4),
        ));
        assert!(negotiation.output_recreated(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(4),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Virtualized);
    }

    #[test]
    fn output_recreation_does_not_activate_with_a_stale_surface_extent() {
        let target = Rect::new(0, 0, 1920, 1080);
        let mut negotiation = PresentationNegotiation::direct();
        let started = Instant::now();

        negotiation.request_borderless(target, started);
        negotiation.borderless_requested(started);
        assert!(!negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(1),
        ));

        assert!(!negotiation.output_recreated(
            target,
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1040)),
            started + Duration::from_secs(2),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);
    }

    #[test]
    fn negotiation_times_out_during_output_recreation() {
        let target = Rect::new(0, 0, 1920, 1080);
        let mut negotiation = PresentationNegotiation::direct();
        let started = Instant::now();

        negotiation.request_borderless(target, started);
        negotiation.borderless_requested(started);
        assert!(!negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(1),
        ));
        assert!(!negotiation.output_recreated(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(5),
        ));

        assert_eq!(negotiation.public_state(), PresentationState::Failed);
        assert_eq!(
            negotiation.failure(),
            Some(NegotiationFailure::DeadlineExpired)
        );
    }

    #[test]
    fn active_negotiation_returns_to_waiting_when_native_extent_is_lost() {
        let target = Rect::new(-1920, 0, 1920, 1080);
        let mut negotiation = PresentationNegotiation::direct();
        let started = Instant::now();

        negotiation.request_borderless(target, started);
        negotiation.borderless_requested(started);
        assert!(!negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(1),
        ));
        assert!(negotiation.observe(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(2),
        ));
        assert!(negotiation.output_recreated(
            target,
            true,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(2),
        ));

        assert!(!negotiation.observe(
            Rect::new(-1920, 0, 1920, 1040),
            true,
            SurfaceExtent::fixed(Extent::new(1920, 1040)),
            started + Duration::from_secs(2),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);
    }

    #[test]
    fn negotiation_keeps_the_first_failure_reason_stable() {
        let mut negotiation = PresentationNegotiation::direct();

        negotiation.fail(NegotiationFailure::BorderlessRequest);
        negotiation.fail(NegotiationFailure::OutputRecreation);

        assert_eq!(negotiation.public_state(), PresentationState::Failed);
        assert_eq!(
            negotiation.failure(),
            Some(NegotiationFailure::BorderlessRequest)
        );
    }

    #[test]
    fn negotiation_maps_internal_stages_to_public_states() {
        let target = Rect::new(0, 0, 1920, 1080);
        let mut negotiation = PresentationNegotiation::direct();

        assert_eq!(negotiation.public_state(), PresentationState::Direct);
        negotiation.request_borderless(target, Instant::now());
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);
        negotiation.fail(NegotiationFailure::BorderlessRequest);
        assert_eq!(negotiation.public_state(), PresentationState::Failed);
    }

    #[test]
    fn domain_types_keep_window_and_output_target_explicit() {
        let window = X11Window::new(42);
        let monitor = Monitor::new(Rect::new(0, 0, 1920, 1080));
        let target = DisplayTarget::new(window, monitor);

        assert_eq!(target.window.id, 42);
        assert_eq!(target.output_extent(), [1920, 1080]);
    }

    #[test]
    fn recognizes_explicit_and_borderless_fullscreen() {
        let monitor = Monitor::new(Rect::new(1920, 0, 1920, 1080));
        let mut window = X11Window::new(42);
        window.rect = Rect::new(2000, 100, 1280, 720);
        assert!(!DisplayTarget::new(window, monitor).is_fullscreen());
        window.fullscreen = true;
        assert!(DisplayTarget::new(window, monitor).is_fullscreen());
        window.fullscreen = false;
        window.rect = monitor.rect;
        assert!(DisplayTarget::new(window, monitor).is_fullscreen());
    }

    #[test]
    fn borderless_lease_snapshots_geometry_and_fullscreen_ownership() {
        let original = WindowSnapshot {
            rect: Rect::new(-1200, 80, 1280, 720),
            fullscreen: false,
        };
        let monitor = Monitor::new(Rect::new(-1920, 0, 1920, 1080));
        let lease = BorderlessLease {
            window: 42,
            original,
            monitor,
        };

        assert_eq!(lease.window, 42);
        assert!(lease.added_fullscreen());
        assert!(lease.should_remove_fullscreen(true));
        assert!(!lease.should_remove_fullscreen(false));
    }

    #[test]
    fn zero_sized_windows_and_monitors_are_rejected() {
        assert!(!Rect::new(0, 0, 0, 100).is_valid());
        assert!(!Rect::new(0, 0, 100, 0).is_valid());
        assert_eq!(
            select_monitor(
                Rect::new(0, 0, 1280, 720),
                &[Monitor::new(Rect::new(0, 0, 0, 1080))]
            ),
            None
        );
    }
}
