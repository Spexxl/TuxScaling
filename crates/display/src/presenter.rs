use std::ffi::c_void;

use x11rb::{
    COPY_DEPTH_FROM_PARENT,
    connection::Connection,
    protocol::xfixes::ConnectionExt as _,
    protocol::xproto::{
        AtomEnum, ConnectionExt as _, CreateWindowAux, EventMask, PropMode, WindowClass,
    },
    wrapper::ConnectionExt as _,
    xcb_ffi::XCBConnection,
};

use super::{DisplayError, Extent, Monitor, Rect, operation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresenterSurfaceInfo {
    pub connection: *mut c_void,
    pub window: u32,
    pub extent: Extent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresenterLifecycle {
    Created,
    CursorHidden,
    Destroyed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresenterEvent {
    HideCursor,
    ShowCursor,
    Destroy,
}

pub fn presenter_lifecycle_transition(
    state: PresenterLifecycle,
    event: PresenterEvent,
) -> PresenterLifecycle {
    match (state, event) {
        (PresenterLifecycle::Destroyed, _) | (_, PresenterEvent::Destroy) => {
            PresenterLifecycle::Destroyed
        }
        (PresenterLifecycle::Created, PresenterEvent::HideCursor) => {
            PresenterLifecycle::CursorHidden
        }
        (PresenterLifecycle::CursorHidden, PresenterEvent::ShowCursor) => {
            PresenterLifecycle::Created
        }
        (state, _) => state,
    }
}

pub fn presenter_event_mask() -> EventMask {
    EventMask::POINTER_MOTION
        | EventMask::BUTTON_PRESS
        | EventMask::BUTTON_RELEASE
        | EventMask::ENTER_WINDOW
        | EventMask::LEAVE_WINDOW
        | EventMask::FOCUS_CHANGE
        | EventMask::KEY_PRESS
        | EventMask::KEY_RELEASE
}

pub fn presenter_geometry(monitor: Monitor) -> Option<Rect> {
    let rect = monitor.rect;
    (rect.is_valid()
        && i16::try_from(rect.x).is_ok()
        && i16::try_from(rect.y).is_ok()
        && u16::try_from(rect.width).is_ok()
        && u16::try_from(rect.height).is_ok())
    .then_some(rect)
}

pub struct PresenterWindow {
    connection: XCBConnection,
    window: u32,
    game_window: u64,
    monitor: Monitor,
    cursor_hidden: bool,
    destroyed: bool,
}

impl PresenterWindow {
    pub fn for_game_window(game_window: u64) -> Result<Self, DisplayError> {
        let display = super::X11Display::connect()?;
        let monitor = display.monitor_for_window(game_window)?;
        Self::new(game_window, monitor)
    }

    pub fn new(game_window: u64, monitor: Monitor) -> Result<Self, DisplayError> {
        let rect = presenter_geometry(monitor).ok_or(DisplayError::Geometry)?;
        let (connection, screen_number) = XCBConnection::connect(None).map_err(operation)?;
        let screen = connection
            .setup()
            .roots
            .get(screen_number)
            .ok_or(DisplayError::Monitor)?;
        let window = connection.generate_id().map_err(operation)?;
        let x = i16::try_from(rect.x).map_err(|_| DisplayError::Geometry)?;
        let y = i16::try_from(rect.y).map_err(|_| DisplayError::Geometry)?;
        let width = u16::try_from(rect.width).map_err(|_| DisplayError::Geometry)?;
        let height = u16::try_from(rect.height).map_err(|_| DisplayError::Geometry)?;
        let attributes = CreateWindowAux::new()
            .background_pixel(screen.black_pixel)
            .border_pixel(screen.black_pixel)
            .override_redirect(1)
            .event_mask(presenter_event_mask());
        connection
            .create_window(
                COPY_DEPTH_FROM_PARENT,
                window,
                screen.root,
                x,
                y,
                width,
                height,
                0,
                WindowClass::INPUT_OUTPUT,
                0,
                &attributes,
            )
            .map_err(operation)?
            .check()
            .map_err(operation)?;
        connection
            .change_property8(
                PropMode::REPLACE,
                window,
                AtomEnum::WM_NAME,
                AtomEnum::STRING,
                b"TuxScaling Presenter",
            )
            .map_err(operation)?
            .check()
            .map_err(operation)?;
        connection
            .map_window(window)
            .map_err(operation)?
            .check()
            .map_err(operation)?;
        connection.flush().map_err(operation)?;

        Ok(Self {
            connection,
            window,
            game_window,
            monitor,
            cursor_hidden: false,
            destroyed: false,
        })
    }

    pub fn surface_info(&self) -> PresenterSurfaceInfo {
        PresenterSurfaceInfo {
            connection: self.connection.get_raw_xcb_connection(),
            window: self.window,
            extent: self.monitor.rect.extent(),
        }
    }

    pub const fn window(&self) -> u32 {
        self.window
    }

    pub const fn game_window(&self) -> u64 {
        self.game_window
    }

    pub const fn monitor(&self) -> Monitor {
        self.monitor
    }

    pub const fn lifecycle(&self) -> PresenterLifecycle {
        if self.destroyed {
            PresenterLifecycle::Destroyed
        } else if self.cursor_hidden {
            PresenterLifecycle::CursorHidden
        } else {
            PresenterLifecycle::Created
        }
    }

    pub fn hide_cursor(&mut self) -> Result<(), DisplayError> {
        if self.destroyed || self.cursor_hidden {
            return Ok(());
        }
        self.connection
            .xfixes_hide_cursor(self.window)
            .map_err(operation)?
            .check()
            .map_err(operation)?;
        self.connection.flush().map_err(operation)?;
        self.cursor_hidden = true;
        Ok(())
    }

    pub fn show_cursor(&mut self) -> Result<(), DisplayError> {
        if self.destroyed || !self.cursor_hidden {
            return Ok(());
        }
        self.connection
            .xfixes_show_cursor(self.window)
            .map_err(operation)?
            .check()
            .map_err(operation)?;
        self.connection.flush().map_err(operation)?;
        self.cursor_hidden = false;
        Ok(())
    }

    pub fn destroy(&mut self) {
        if self.destroyed {
            return;
        }
        let _ = self.show_cursor();
        if let Ok(cookie) = self.connection.destroy_window(self.window) {
            let _ = cookie.check();
        }
        let _ = self.connection.flush();
        self.destroyed = true;
    }
}

impl Drop for PresenterWindow {
    fn drop(&mut self) {
        self.destroy();
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Monitor, Rect, select_monitor};
    use super::{
        PresenterEvent, PresenterLifecycle, presenter_event_mask, presenter_geometry,
        presenter_lifecycle_transition,
    };
    use x11rb::protocol::xproto::EventMask;

    #[test]
    fn selects_the_monitor_containing_the_presenter_source() {
        let monitors = [
            Monitor::new(Rect::new(0, 0, 1920, 1080)),
            Monitor::new(Rect::new(1920, 0, 2560, 1440)),
        ];

        assert_eq!(
            select_monitor(Rect::new(2100, 80, 640, 360), &monitors),
            Some(monitors[1])
        );
    }

    #[test]
    fn presenter_geometry_matches_the_selected_monitor_rectangle() {
        let monitor = Monitor::new(Rect::new(-1920, 32, 1920, 1080));

        assert_eq!(presenter_geometry(monitor), Some(monitor.rect));
        assert_eq!(
            presenter_geometry(Monitor::new(Rect::new(0, 0, 0, 1080))),
            None
        );
    }

    #[test]
    fn presenter_event_mask_contains_pointer_focus_and_keyboard_events() {
        let mask = presenter_event_mask();

        assert!(mask.contains(EventMask::POINTER_MOTION));
        assert!(mask.contains(EventMask::BUTTON_PRESS | EventMask::BUTTON_RELEASE));
        assert!(mask.contains(EventMask::ENTER_WINDOW | EventMask::LEAVE_WINDOW));
        assert!(mask.contains(EventMask::FOCUS_CHANGE));
        assert!(mask.contains(EventMask::KEY_PRESS | EventMask::KEY_RELEASE));
    }

    #[test]
    fn presenter_lifecycle_restores_cursor_before_destroying_window() {
        assert_eq!(
            presenter_lifecycle_transition(PresenterLifecycle::Created, PresenterEvent::HideCursor,),
            PresenterLifecycle::CursorHidden
        );
        assert_eq!(
            presenter_lifecycle_transition(
                PresenterLifecycle::CursorHidden,
                PresenterEvent::Destroy,
            ),
            PresenterLifecycle::Destroyed
        );
        assert_eq!(
            presenter_lifecycle_transition(
                PresenterLifecycle::Destroyed,
                PresenterEvent::ShowCursor,
            ),
            PresenterLifecycle::Destroyed
        );
    }

    #[test]
    #[ignore = "requires a nested XWayland display"]
    fn presenter_preserves_original_window_and_owns_its_xcb_connection() {
        use x11rb::protocol::xproto::{ConnectionExt, CreateWindowAux, WindowClass};
        use x11rb::{COPY_DEPTH_FROM_PARENT, COPY_FROM_PARENT, connection::Connection};

        let (connection, screen_number) = x11rb::xcb_ffi::XCBConnection::connect(None).unwrap();
        let screen = &connection.setup().roots[screen_number];
        let game_window = connection.generate_id().unwrap();
        connection
            .create_window(
                COPY_DEPTH_FROM_PARENT,
                game_window,
                screen.root,
                16,
                24,
                320,
                180,
                0,
                WindowClass::INPUT_OUTPUT,
                COPY_FROM_PARENT,
                &CreateWindowAux::new(),
            )
            .unwrap()
            .check()
            .unwrap();
        connection.map_window(game_window).unwrap().check().unwrap();
        connection.flush().unwrap();
        let before = connection
            .get_geometry(game_window)
            .unwrap()
            .reply()
            .unwrap();
        let root_geometry = connection
            .get_geometry(screen.root)
            .unwrap()
            .reply()
            .unwrap();
        let monitor = Monitor::new(Rect::new(
            0,
            0,
            u32::from(root_geometry.width),
            u32::from(root_geometry.height),
        ));
        let presenter = super::super::PresenterWindow::new(game_window.into(), monitor).unwrap();
        let info = presenter.surface_info();
        assert!(!info.connection.is_null());
        assert_ne!(info.window, game_window);
        assert_eq!(info.extent, monitor.rect.extent());
        let attributes = connection
            .get_window_attributes(info.window)
            .unwrap()
            .reply()
            .unwrap();
        assert!(attributes.override_redirect);
        let after = connection
            .get_geometry(game_window)
            .unwrap()
            .reply()
            .unwrap();
        assert_eq!(
            (before.x, before.y, before.width, before.height),
            (after.x, after.y, after.width, after.height)
        );
        drop(presenter);
        assert!(
            connection
                .get_geometry(game_window)
                .unwrap()
                .reply()
                .is_ok()
        );
        connection
            .destroy_window(game_window)
            .unwrap()
            .check()
            .unwrap();
        connection.flush().unwrap();
    }
}
