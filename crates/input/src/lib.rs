use egui::{Event, Key, Modifiers, PointerButton, Pos2, Vec2};
use libloading::Library;
use std::os::raw::{c_char, c_int, c_long, c_uint, c_ulong};
use thiserror::Error;

pub const CRATE_NAME: &str = "tuxscaling-input";

const KEY_PRESS: c_int = 2;
const KEY_RELEASE: c_int = 3;
const BUTTON_PRESS: c_int = 4;
const BUTTON_RELEASE: c_int = 5;
const MOTION_NOTIFY: c_int = 6;
const ENTER_NOTIFY: c_int = 7;
const LEAVE_NOTIFY: c_int = 8;
const FOCUS_IN: c_int = 9;
const FOCUS_OUT: c_int = 10;
const INSERT_KEYSYM: c_ulong = 0xff63;
const POINTER_GRAB_MASK: c_ulong = (1 << 2) | (1 << 3) | (1 << 6);
const PASSIVE_EVENT_MASK: c_long = 1 | 2 | 4 | 8 | 16 | 32 | (1 << 6) | (1 << 21);

#[repr(C)]
struct Display;

#[repr(C)]
#[derive(Clone, Copy)]
struct KeyEvent {
    type_: c_int,
    serial: c_ulong,
    send_event: c_int,
    display: *mut Display,
    window: c_ulong,
    root: c_ulong,
    subwindow: c_ulong,
    time: c_ulong,
    x: c_int,
    y: c_int,
    x_root: c_int,
    y_root: c_int,
    state: c_uint,
    keycode: c_uint,
    same_screen: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ButtonEvent {
    type_: c_int,
    serial: c_ulong,
    send_event: c_int,
    display: *mut Display,
    window: c_ulong,
    root: c_ulong,
    subwindow: c_ulong,
    time: c_ulong,
    x: c_int,
    y: c_int,
    x_root: c_int,
    y_root: c_int,
    state: c_uint,
    button: c_uint,
    same_screen: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MotionEvent {
    type_: c_int,
    serial: c_ulong,
    send_event: c_int,
    display: *mut Display,
    window: c_ulong,
    root: c_ulong,
    subwindow: c_ulong,
    time: c_ulong,
    x: c_int,
    y: c_int,
    x_root: c_int,
    y_root: c_int,
    state: c_uint,
    is_hint: c_char,
    same_screen: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CrossingEvent {
    type_: c_int,
    serial: c_ulong,
    send_event: c_int,
    display: *mut Display,
    window: c_ulong,
    root: c_ulong,
    subwindow: c_ulong,
    time: c_ulong,
    x: c_int,
    y: c_int,
    x_root: c_int,
    y_root: c_int,
    mode: c_int,
    detail: c_int,
    same_screen: c_int,
    focus: c_int,
    state: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FocusEvent {
    type_: c_int,
    serial: c_ulong,
    send_event: c_int,
    display: *mut Display,
    window: c_ulong,
    mode: c_int,
    detail: c_int,
}

#[repr(C)]
union XEvent {
    type_: c_int,
    key: KeyEvent,
    button: ButtonEvent,
    motion: MotionEvent,
    crossing: CrossingEvent,
    focus: FocusEvent,
    padding: [c_long; 24],
}

type OpenDisplay = unsafe extern "C" fn(*const c_char) -> *mut Display;
type DefaultScreen = unsafe extern "C" fn(*mut Display) -> c_int;
type RootWindow = unsafe extern "C" fn(*mut Display, c_int) -> c_ulong;
type KeysymToKeycode = unsafe extern "C" fn(*mut Display, c_ulong) -> u8;
type Pending = unsafe extern "C" fn(*mut Display) -> c_int;
type NextEvent = unsafe extern "C" fn(*mut Display, *mut XEvent) -> c_int;
type GrabKeyboard =
    unsafe extern "C" fn(*mut Display, c_ulong, c_int, c_int, c_int, c_ulong) -> c_int;
type UngrabKeyboard = unsafe extern "C" fn(*mut Display, c_ulong) -> c_int;
type GrabPointer = unsafe extern "C" fn(
    *mut Display,
    c_ulong,
    c_int,
    c_ulong,
    c_int,
    c_int,
    c_ulong,
    c_ulong,
    c_ulong,
) -> c_int;
type UngrabPointer = unsafe extern "C" fn(*mut Display, c_ulong) -> c_int;
type Flush = unsafe extern "C" fn(*mut Display) -> c_int;
type CloseDisplay = unsafe extern "C" fn(*mut Display) -> c_int;
type Sync = unsafe extern "C" fn(*mut Display, c_int) -> c_int;
type SelectInput = unsafe extern "C" fn(*mut Display, c_ulong, c_long) -> c_int;
type WarpPointer = unsafe extern "C" fn(
    *mut Display,
    c_ulong,
    c_ulong,
    c_int,
    c_int,
    c_uint,
    c_uint,
    c_int,
    c_int,
) -> c_int;
type FakeButtonEvent = unsafe extern "C" fn(*mut Display, c_uint, c_int, c_ulong) -> c_int;
type FakeRelativeMotionEvent = unsafe extern "C" fn(*mut Display, c_int, c_int, c_ulong) -> c_int;
type HideCursor = unsafe extern "C" fn(*mut Display, c_ulong) -> c_int;
type ShowCursor = unsafe extern "C" fn(*mut Display, c_ulong) -> c_int;

#[derive(Debug, Error)]
pub enum InputError {
    #[error("libX11.so.6 is unavailable")]
    Library(#[from] libloading::Error),
    #[error("XOpenDisplay failed")]
    DisplayUnavailable,
}

#[derive(Debug, Default)]
pub struct InputFrame {
    pub events: Vec<Event>,
    pub toggle_overlay: bool,
    pub pointer_position: Option<[f32; 2]>,
    pub pointer_present: bool,
    pub cursor_owner: CursorOwner,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CursorOwner {
    #[default]
    Native,
    Overlay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorCleanupReason {
    OverlayClosed,
    FocusLost,
    PresenterDestroyed,
    RendererError,
    Panic,
    ProcessExit,
}

pub const fn cursor_owner_after(_owner: CursorOwner, reason: CursorCleanupReason) -> CursorOwner {
    match reason {
        CursorCleanupReason::OverlayClosed
        | CursorCleanupReason::FocusLost
        | CursorCleanupReason::PresenterDestroyed
        | CursorCleanupReason::RendererError
        | CursorCleanupReason::Panic
        | CursorCleanupReason::ProcessExit => CursorOwner::Native,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointerViewport {
    pub offset: [f32; 2],
    pub extent: [f32; 2],
}

impl PointerViewport {
    fn is_valid(self) -> bool {
        self.offset
            .iter()
            .chain(self.extent.iter())
            .all(|value| value.is_finite() && *value >= 0.0)
            && self.extent.iter().all(|value| *value > 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InputRoute {
    pub event_window: u64,
    pub game_window: u64,
    pub output_extent: [u32; 2],
    pub game_extent: [u32; 2],
    pub viewport: PointerViewport,
}

impl InputRoute {
    pub const fn new(
        event_window: u64,
        game_window: u64,
        output_extent: [u32; 2],
        game_extent: [u32; 2],
        viewport: PointerViewport,
    ) -> Self {
        Self {
            event_window,
            game_window,
            output_extent,
            game_extent,
            viewport,
        }
    }

    pub fn map_absolute(self, position: [f32; 2], button: bool) -> Option<[f32; 2]> {
        if self.output_extent.contains(&0)
            || self.game_extent.contains(&0)
            || !self.viewport.is_valid()
            || position.iter().any(|value| !value.is_finite())
        {
            return None;
        }
        let min = self.viewport.offset;
        let max = [
            min[0] + self.viewport.extent[0],
            min[1] + self.viewport.extent[1],
        ];
        let inside = position[0] >= min[0]
            && position[0] <= max[0]
            && position[1] >= min[1]
            && position[1] <= max[1];
        if !inside && !button {
            return None;
        }
        let point = [
            position[0].clamp(min[0], max[0]),
            position[1].clamp(min[1], max[1]),
        ];
        Some([
            ((point[0] - min[0]) / self.viewport.extent[0]).clamp(0.0, 1.0)
                * self.game_extent[0] as f32,
            ((point[1] - min[1]) / self.viewport.extent[1]).clamp(0.0, 1.0)
                * self.game_extent[1] as f32,
        ])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerMode {
    Absolute,
    Relative,
}

pub fn route_pointer(
    route: &InputRoute,
    mode: PointerMode,
    position_or_delta: [f32; 2],
    button: bool,
) -> Option<[f32; 2]> {
    match mode {
        PointerMode::Absolute => route.map_absolute(position_or_delta, button),
        PointerMode::Relative => Some(position_or_delta),
    }
}

pub fn should_accept_event(route: &InputRoute, event_window: u64) -> bool {
    route.event_window != 0 && event_window == route.event_window
}

pub fn should_forward_to_game(route: &InputRoute, overlay_open: bool) -> bool {
    !overlay_open
        && route.event_window != 0
        && route.game_window != 0
        && route.event_window != route.game_window
}

pub struct X11Input {
    _library: Library,
    _xtest_library: Option<Library>,
    _xfixes_library: Option<Library>,
    display: *mut Display,
    route: InputRoute,
    event_window: c_ulong,
    insert_keycode: u8,
    overlay_open: bool,
    pointer_mode: PointerMode,
    last_root: Option<[i32; 2]>,
    last_position: Option<[f32; 2]>,
    pointer_present: bool,
    pending: Pending,
    next_event: NextEvent,
    grab_keyboard: GrabKeyboard,
    ungrab_keyboard: UngrabKeyboard,
    grab_pointer: GrabPointer,
    ungrab_pointer: UngrabPointer,
    flush: Flush,
    close_display: CloseDisplay,
    warp_pointer: WarpPointer,
    fake_button: Option<FakeButtonEvent>,
    fake_relative_motion: Option<FakeRelativeMotionEvent>,
    hide_cursor: Option<HideCursor>,
    show_cursor: Option<ShowCursor>,
}

unsafe impl Send for X11Input {}

impl X11Input {
    pub fn connect(route: InputRoute) -> Result<Self, InputError> {
        let library = unsafe { Library::new("libX11.so.6") }?;
        let open_display = load::<OpenDisplay>(&library, b"XOpenDisplay\0")?;
        let default_screen = load::<DefaultScreen>(&library, b"XDefaultScreen\0")?;
        let root_window = load::<RootWindow>(&library, b"XRootWindow\0")?;
        let keysym_to_keycode = load::<KeysymToKeycode>(&library, b"XKeysymToKeycode\0")?;
        let pending = load::<Pending>(&library, b"XPending\0")?;
        let next_event = load::<NextEvent>(&library, b"XNextEvent\0")?;
        let grab_keyboard = load::<GrabKeyboard>(&library, b"XGrabKeyboard\0")?;
        let ungrab_keyboard = load::<UngrabKeyboard>(&library, b"XUngrabKeyboard\0")?;
        let grab_pointer = load::<GrabPointer>(&library, b"XGrabPointer\0")?;
        let ungrab_pointer = load::<UngrabPointer>(&library, b"XUngrabPointer\0")?;
        let flush = load::<Flush>(&library, b"XFlush\0")?;
        let close_display = load::<CloseDisplay>(&library, b"XCloseDisplay\0")?;
        let warp_pointer = load::<WarpPointer>(&library, b"XWarpPointer\0")?;
        let display = unsafe { open_display(std::ptr::null()) };
        if display.is_null() {
            return Err(InputError::DisplayUnavailable);
        }
        let screen = unsafe { default_screen(display) };
        let root = unsafe { root_window(display, screen) };
        let insert_keycode = unsafe { keysym_to_keycode(display, INSERT_KEYSYM) };
        let select_input = load::<SelectInput>(&library, b"XSelectInput\0")?;
        unsafe {
            select_input(
                display,
                if route.event_window == 0 {
                    root
                } else {
                    route.event_window as c_ulong
                },
                PASSIVE_EVENT_MASK,
            );
        }
        let xtest_library = unsafe { Library::new("libXtst.so.6") }.ok();
        let fake_button = xtest_library
            .as_ref()
            .and_then(|library| load::<FakeButtonEvent>(library, b"XTestFakeButtonEvent\0").ok());
        let fake_relative_motion = xtest_library.as_ref().and_then(|library| {
            load::<FakeRelativeMotionEvent>(library, b"XTestFakeRelativeMotionEvent\0").ok()
        });
        let xfixes_library = unsafe { Library::new("libXfixes.so.3") }.ok();
        let hide_cursor = xfixes_library
            .as_ref()
            .and_then(|library| load::<HideCursor>(library, b"XFixesHideCursor\0").ok());
        let show_cursor = xfixes_library
            .as_ref()
            .and_then(|library| load::<ShowCursor>(library, b"XFixesShowCursor\0").ok());
        let input = Self {
            _library: library,
            _xtest_library: xtest_library,
            _xfixes_library: xfixes_library,
            display,
            route,
            event_window: if route.event_window == 0 {
                root
            } else {
                route.event_window as c_ulong
            },
            insert_keycode,
            overlay_open: false,
            pointer_mode: PointerMode::Absolute,
            last_root: None,
            last_position: None,
            pointer_present: false,
            pending,
            next_event,
            grab_keyboard,
            ungrab_keyboard,
            grab_pointer,
            ungrab_pointer,
            flush,
            close_display,
            warp_pointer,
            fake_button,
            fake_relative_motion,
            hide_cursor,
            show_cursor,
        };
        input.update_grab();
        let sync = load::<Sync>(&input._library, b"XSync\0")?;
        unsafe { sync(display, 0) };
        Ok(input)
    }

    pub fn set_pointer_mode(&mut self, mode: PointerMode) {
        self.pointer_mode = mode;
        self.last_root = None;
    }

    pub fn cursor_owner(&self) -> CursorOwner {
        if self.overlay_open {
            CursorOwner::Overlay
        } else {
            CursorOwner::Native
        }
    }

    pub fn poll(&mut self) -> InputFrame {
        let mut frame = InputFrame::default();
        while unsafe { (self.pending)(self.display) } > 0 {
            let mut event = XEvent { padding: [0; 24] };
            unsafe { (self.next_event)(self.display, &mut event) };
            let kind = unsafe { event.type_ };
            match kind {
                KEY_PRESS => {
                    let event = unsafe { event.key };
                    if !should_accept_event(&self.route, event.window) || event.send_event != 0 {
                        continue;
                    }
                    if event.keycode as u8 == self.insert_keycode {
                        frame.toggle_overlay = true;
                        self.overlay_open = !self.overlay_open;
                        self.update_grab();
                        self.update_cursor_visibility();
                        frame.events.push(key_event(true));
                    }
                }
                KEY_RELEASE => {
                    let event = unsafe { event.key };
                    if !should_accept_event(&self.route, event.window) || event.send_event != 0 {
                        continue;
                    }
                    if event.keycode as u8 == self.insert_keycode {
                        frame.events.push(key_event(false));
                    }
                }
                MOTION_NOTIFY => {
                    let event = unsafe { event.motion };
                    if !should_accept_event(&self.route, event.window) || event.send_event != 0 {
                        continue;
                    }
                    let root = [event.x_root, event.y_root];
                    let position = [event.x as f32, event.y as f32];
                    let delta = self.last_root.map_or([0.0, 0.0], |last| {
                        [(root[0] - last[0]) as f32, (root[1] - last[1]) as f32]
                    });
                    self.last_root = Some(root);
                    self.last_position = Some(position);
                    self.pointer_present = true;
                    match self.pointer_mode {
                        PointerMode::Absolute => {
                            if self.overlay_open {
                                frame
                                    .events
                                    .push(Event::PointerMoved(Pos2::new(position[0], position[1])));
                            } else if let Some(position) = self.route.map_absolute(position, false)
                            {
                                self.forward_absolute(position);
                            }
                        }
                        PointerMode::Relative => {
                            if self.overlay_open {
                                frame
                                    .events
                                    .push(Event::MouseMoved(Vec2::new(delta[0], delta[1])));
                            } else {
                                self.forward_relative(delta);
                            }
                        }
                    }
                }
                BUTTON_PRESS => {
                    let event = unsafe { event.button };
                    if !should_accept_event(&self.route, event.window) || event.send_event != 0 {
                        continue;
                    }
                    let position = [event.x as f32, event.y as f32];
                    if let Some(button) = pointer_button(event.button as u8) {
                        let position = if self.overlay_open {
                            Some(position)
                        } else {
                            self.route.map_absolute(position, true)
                        };
                        if let Some(position) = position {
                            if !self.overlay_open {
                                self.forward_absolute(position);
                            } else {
                                frame.events.push(Event::PointerButton {
                                    pos: Pos2::new(position[0], position[1]),
                                    button,
                                    pressed: true,
                                    modifiers: modifiers(event.state),
                                });
                            }
                        }
                        if !self.overlay_open {
                            self.forward_button(event.button, true);
                        }
                    } else if event.button == 4 || event.button == 5 {
                        if !self.overlay_open
                            && let Some(position) = self.route.map_absolute(position, true)
                        {
                            self.forward_absolute(position);
                        }
                        if self.overlay_open {
                            frame.events.push(Event::MouseWheel {
                                unit: egui::MouseWheelUnit::Line,
                                delta: Vec2::new(0.0, if event.button == 4 { 1.0 } else { -1.0 }),
                                modifiers: modifiers(event.state),
                            });
                        }
                        if !self.overlay_open {
                            self.forward_button(event.button, true);
                        }
                    }
                }
                BUTTON_RELEASE => {
                    let event = unsafe { event.button };
                    if !should_accept_event(&self.route, event.window) || event.send_event != 0 {
                        continue;
                    }
                    let position = [event.x as f32, event.y as f32];
                    if let Some(button) = pointer_button(event.button as u8) {
                        let position = if self.overlay_open {
                            Some(position)
                        } else {
                            self.route.map_absolute(position, true)
                        };
                        if let Some(position) = position {
                            if !self.overlay_open {
                                self.forward_absolute(position);
                            } else {
                                frame.events.push(Event::PointerButton {
                                    pos: Pos2::new(position[0], position[1]),
                                    button,
                                    pressed: false,
                                    modifiers: modifiers(event.state),
                                });
                            }
                        }
                        if !self.overlay_open {
                            self.forward_button(event.button, false);
                        }
                    }
                }
                ENTER_NOTIFY | LEAVE_NOTIFY => {
                    let event = unsafe { event.crossing };
                    if !should_accept_event(&self.route, event.window) || event.send_event != 0 {
                        continue;
                    }
                    if kind == ENTER_NOTIFY {
                        self.last_position = Some([event.x as f32, event.y as f32]);
                        self.pointer_present = true;
                    } else {
                        self.pointer_present = false;
                        if self.overlay_open {
                            frame.events.push(Event::PointerGone);
                        }
                    }
                }
                FOCUS_IN | FOCUS_OUT => {
                    let event = unsafe { event.focus };
                    if !should_accept_event(&self.route, event.window) || event.send_event != 0 {
                        continue;
                    }
                    if self.overlay_open {
                        frame.events.push(Event::WindowFocused(kind == FOCUS_IN));
                    }
                    if kind == FOCUS_OUT {
                        self.pointer_present = false;
                        if self.overlay_open {
                            self.overlay_open = false;
                            self.update_grab();
                            self.update_cursor_visibility();
                            frame.toggle_overlay = true;
                        }
                    }
                }
                _ => {}
            }
        }
        frame.pointer_position = self.last_position;
        frame.pointer_present = self.pointer_present;
        frame.cursor_owner = self.cursor_owner();
        frame
    }

    fn update_grab(&self) {
        if self.overlay_open {
            unsafe {
                (self.grab_keyboard)(self.display, self.event_window, 0, 1, 1, 0);
                (self.grab_pointer)(
                    self.display,
                    self.event_window,
                    0,
                    POINTER_GRAB_MASK,
                    1,
                    1,
                    0,
                    0,
                    0,
                );
            }
        } else {
            unsafe {
                (self.ungrab_keyboard)(self.display, 0);
                (self.ungrab_pointer)(self.display, 0);
            }
        }
        unsafe {
            (self.flush)(self.display);
        }
    }

    fn update_cursor_visibility(&self) {
        if self.overlay_open {
            self.hide_native_cursor();
        } else {
            self.show_native_cursor();
        }
    }

    fn hide_native_cursor(&self) {
        if let Some(hide_cursor) = self.hide_cursor
            && self.event_window != 0
        {
            unsafe {
                hide_cursor(self.display, self.event_window);
                (self.flush)(self.display);
            }
        }
    }

    fn show_native_cursor(&self) {
        if let Some(show_cursor) = self.show_cursor
            && self.event_window != 0
        {
            unsafe {
                show_cursor(self.display, self.event_window);
                (self.flush)(self.display);
            }
        }
    }

    fn forward_absolute(&self, position: [f32; 2]) {
        if !should_forward_to_game(&self.route, self.overlay_open) {
            return;
        }
        unsafe {
            (self.warp_pointer)(
                self.display,
                0,
                self.route.game_window as c_ulong,
                0,
                0,
                0,
                0,
                position[0].round() as c_int,
                position[1].round() as c_int,
            );
            (self.flush)(self.display);
        }
    }

    fn forward_relative(&self, delta: [f32; 2]) {
        if !should_forward_to_game(&self.route, self.overlay_open) {
            return;
        }
        if let Some(fake_relative_motion) = self.fake_relative_motion {
            unsafe {
                fake_relative_motion(
                    self.display,
                    delta[0].round() as c_int,
                    delta[1].round() as c_int,
                    0,
                );
                (self.flush)(self.display);
            }
        }
    }

    fn forward_button(&self, button: c_uint, pressed: bool) {
        if !should_forward_to_game(&self.route, self.overlay_open) {
            return;
        }
        if let Some(fake_button) = self.fake_button {
            unsafe {
                fake_button(self.display, button, c_int::from(pressed), 0);
                (self.flush)(self.display);
            }
        }
    }
}

impl Drop for X11Input {
    fn drop(&mut self) {
        if !self.display.is_null() {
            unsafe {
                self.show_native_cursor();
                (self.ungrab_keyboard)(self.display, 0);
                (self.ungrab_pointer)(self.display, 0);
                (self.close_display)(self.display);
            }
        }
    }
}

fn load<T: Copy>(library: &Library, name: &[u8]) -> Result<T, libloading::Error> {
    Ok(*unsafe { library.get::<T>(name) }?)
}

fn key_event(pressed: bool) -> Event {
    Event::Key {
        key: Key::Insert,
        physical_key: None,
        pressed,
        repeat: false,
        modifiers: Modifiers::default(),
    }
}

fn pointer_button(detail: u8) -> Option<PointerButton> {
    match detail {
        1 => Some(PointerButton::Primary),
        2 => Some(PointerButton::Middle),
        3 => Some(PointerButton::Secondary),
        _ => None,
    }
}

fn modifiers(state: c_uint) -> Modifiers {
    Modifiers {
        alt: state & (1 << 3) != 0,
        ctrl: state & (1 << 2) != 0,
        shift: state & 1 != 0,
        mac_cmd: false,
        command: false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CursorOwner, InputFrame, InputRoute, PointerMode, PointerViewport, pointer_button,
        route_pointer, should_accept_event, should_forward_to_game,
    };
    use egui::PointerButton;

    #[test]
    fn maps_x11_buttons_to_egui() {
        assert_eq!(pointer_button(1), Some(PointerButton::Primary));
        assert_eq!(pointer_button(3), Some(PointerButton::Secondary));
        assert_eq!(pointer_button(8), None);
    }

    fn route() -> InputRoute {
        InputRoute::new(
            0x900,
            0x400,
            [2160, 1440],
            [1280, 720],
            PointerViewport {
                offset: [0.0, 0.0],
                extent: [2160.0, 1440.0],
            },
        )
    }

    #[test]
    fn maps_presenter_corners_and_center_into_logical_game_space() {
        let route = route();

        assert_eq!(route.map_absolute([0.0, 0.0], false), Some([0.0, 0.0]));
        assert_eq!(
            route.map_absolute([2160.0, 1440.0], false),
            Some([1280.0, 720.0])
        );
        assert_eq!(
            route.map_absolute([1080.0, 720.0], false),
            Some([640.0, 360.0])
        );
    }

    #[test]
    fn maps_fractional_viewports_and_ignores_monitor_origin() {
        let route = InputRoute::new(
            0x1900,
            0x1400,
            [1234, 987],
            [853, 479],
            PointerViewport {
                offset: [23.5, 17.25],
                extent: [1011.5, 569.75],
            },
        );

        assert_eq!(route.event_window, 0x1900);
        assert_eq!(route.game_window, 0x1400);
        assert_eq!(route.map_absolute([23.5, 17.25], false), Some([0.0, 0.0]));
        let center = route.map_absolute([529.25, 302.125], false).unwrap();
        assert!((center[0] - 426.5).abs() < 0.001);
        assert!((center[1] - 239.5).abs() < 0.001);
    }

    #[test]
    fn rejects_motion_in_aspect_fit_bars_but_clamps_buttons_to_content_edge() {
        let route = InputRoute::new(
            0x900,
            0x400,
            [1920, 1200],
            [1280, 720],
            PointerViewport {
                offset: [0.0, 60.0],
                extent: [1920.0, 1080.0],
            },
        );

        assert_eq!(route.map_absolute([960.0, 0.0], false), None);
        assert_eq!(route.map_absolute([960.0, 0.0], true), Some([640.0, 0.0]));
        assert_eq!(
            route.map_absolute([1920.0, 1140.0], false),
            Some([1280.0, 720.0])
        );
    }

    #[test]
    fn relative_motion_is_forwarded_without_output_scaling() {
        let route = route();
        assert_eq!(
            route_pointer(&route, PointerMode::Relative, [17.25, -9.5], false),
            Some([17.25, -9.5])
        );
    }

    #[test]
    fn event_and_forwarding_policy_prevents_feedback_when_overlay_is_open() {
        let route = route();

        assert!(should_accept_event(&route, route.event_window));
        assert!(!should_accept_event(&route, route.game_window));
        assert!(should_forward_to_game(&route, false));
        assert!(!should_forward_to_game(&route, true));
        assert!(!should_forward_to_game(
            &InputRoute::new(
                route.game_window,
                route.game_window,
                route.output_extent,
                route.game_extent,
                route.viewport,
            ),
            false,
        ));
    }

    #[test]
    fn input_frame_defaults_to_native_cursor_ownership_without_pointer_presence() {
        let frame = InputFrame::default();

        assert_eq!(frame.cursor_owner, CursorOwner::Native);
        assert!(!frame.pointer_present);
        assert_eq!(frame.pointer_position, None);
    }

    #[test]
    fn every_overlay_cleanup_path_restores_native_cursor_ownership() {
        for reason in [
            super::CursorCleanupReason::OverlayClosed,
            super::CursorCleanupReason::FocusLost,
            super::CursorCleanupReason::PresenterDestroyed,
            super::CursorCleanupReason::RendererError,
            super::CursorCleanupReason::Panic,
            super::CursorCleanupReason::ProcessExit,
        ] {
            assert_eq!(
                super::cursor_owner_after(CursorOwner::Overlay, reason),
                CursorOwner::Native
            );
        }
    }
}
