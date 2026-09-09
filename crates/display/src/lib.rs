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
    RequestingBorderless { target: Rect },
    WaitingForNativeExtent { target: Rect, deadline: Instant },
    RecreatingOutput { target: Rect },
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

    pub fn request_borderless(&mut self, target: Rect) -> bool {
        if !target.is_valid() || !matches!(self.stage, NegotiationStage::Direct) {
            return false;
        }
        self.stage = NegotiationStage::RequestingBorderless { target };
        true
    }

    pub fn borderless_requested(&mut self, now: Instant) -> bool {
        let NegotiationStage::RequestingBorderless { target } = self.stage else {
            return false;
        };
        self.stage = NegotiationStage::WaitingForNativeExtent {
            target,
            deadline: now + NEGOTIATION_TIMEOUT,
        };
        true
    }

    pub fn observe(&mut self, window: Rect, surface: SurfaceExtent, now: Instant) -> bool {
        match self.stage {
            NegotiationStage::WaitingForNativeExtent { target, deadline } => {
                if now >= deadline {
                    self.fail(NegotiationFailure::DeadlineExpired);
                    return false;
                }
                if window == target && surface.accepts(target.extent()) {
                    self.stage = NegotiationStage::RecreatingOutput { target };
                    return true;
                }
            }
            NegotiationStage::Active { target }
                if window != target || !surface.accepts(target.extent()) =>
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

    pub fn output_recreated(&mut self) -> bool {
        let NegotiationStage::RecreatingOutput { target } = self.stage else {
            return false;
        };
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

    fn configure_rect(&self, window: u64, rect: Rect) -> Result<(), DisplayError> {
        if !rect.is_valid() {
            return Err(DisplayError::Geometry);
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
        select_monitor,
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

        negotiation.request_borderless(target);
        negotiation.borderless_requested(started);
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);
        assert!(!negotiation.observe(
            Rect::new(0, 0, 1920, 1040),
            SurfaceExtent::fixed(Extent::new(1920, 1040)),
            started + Duration::from_secs(1),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);

        assert!(negotiation.observe(
            target,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(2),
        ));
        assert_eq!(negotiation.public_state(), PresentationState::Negotiating);

        negotiation.output_recreated();
        assert_eq!(negotiation.public_state(), PresentationState::Virtualized);
    }

    #[test]
    fn negotiation_times_out_after_five_monotonic_seconds() {
        let target = Rect::new(0, 0, 1920, 1080);
        let mut negotiation = PresentationNegotiation::direct();
        let started = Instant::now();

        negotiation.request_borderless(target);
        negotiation.borderless_requested(started);
        assert!(!negotiation.observe(
            Rect::new(0, 0, 1920, 1040),
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
    fn active_negotiation_returns_to_waiting_when_native_extent_is_lost() {
        let target = Rect::new(-1920, 0, 1920, 1080);
        let mut negotiation = PresentationNegotiation::direct();
        let started = Instant::now();

        negotiation.request_borderless(target);
        negotiation.borderless_requested(started);
        assert!(negotiation.observe(
            target,
            SurfaceExtent::fixed(target.extent()),
            started + Duration::from_secs(1),
        ));
        negotiation.output_recreated();

        assert!(!negotiation.observe(
            Rect::new(-1920, 0, 1920, 1040),
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
        negotiation.request_borderless(target);
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
