use egui::{Event, Key, Modifiers, PointerButton, Pos2, Vec2};
use libloading::Library;
use ownership::{InputOwner, can_open, on_focus_in, open_transition, release_transition};
use std::os::raw::{c_char, c_int, c_long, c_uint, c_ulong};
use thiserror::Error;

pub const CRATE_NAME: &str = "tuxscaling-input";

pub mod ownership;

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
const ANY_MODIFIER: c_uint = 1 << 15;
const GRAB_MODE_ASYNC: c_int = 1;
const SHAPE_INPUT: c_int = 2;
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
type GrabKey =
    unsafe extern "C" fn(*mut Display, c_int, c_uint, c_ulong, c_int, c_int, c_int) -> c_int;
type UngrabKey = unsafe extern "C" fn(*mut Display, c_int, c_uint, c_ulong) -> c_int;
type Flush = unsafe extern "C" fn(*mut Display) -> c_int;
type CloseDisplay = unsafe extern "C" fn(*mut Display) -> c_int;
type Sync = unsafe extern "C" fn(*mut Display, c_int) -> c_int;
type SelectInput = unsafe extern "C" fn(*mut Display, c_ulong, c_long) -> c_int;
type QueryTree = unsafe extern "C" fn(
    *mut Display,
    c_ulong,
    *mut c_ulong,
    *mut c_ulong,
    *mut *mut c_ulong,
    *mut c_uint,
) -> c_int;
type Free = unsafe extern "C" fn(*mut std::ffi::c_void) -> c_int;
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
type QueryXFixesExtension = unsafe extern "C" fn(*mut Display, *mut c_int, *mut c_int) -> c_int;
type QueryXFixesVersion = unsafe extern "C" fn(*mut Display, *mut c_int, *mut c_int) -> c_int;
type HideCursor = unsafe extern "C" fn(*mut Display, c_ulong);
type ShowCursor = unsafe extern "C" fn(*mut Display, c_ulong);
type CreateRegion = unsafe extern "C" fn(*mut Display, *const std::ffi::c_void, c_int) -> c_ulong;
type DestroyRegion = unsafe extern "C" fn(*mut Display, c_ulong);
type SetWindowShapeRegion =
    unsafe extern "C" fn(*mut Display, c_ulong, c_int, c_int, c_int, c_ulong);

#[derive(Debug, Error)]
pub enum InputError {
    #[error("libX11.so.6 is unavailable")]
    Library(#[from] libloading::Error),
    #[error("XOpenDisplay failed")]
    DisplayUnavailable,
    #[error("presenter input routing is unavailable")]
    RoutingUnavailable,
    #[error("X11 {0} grab failed with status {1}")]
    GrabFailed(&'static str, c_int),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CursorVisibilityAction {
    Hide,
    Show,
    None,
}

fn cursor_visibility_action(cursor_hidden: bool, overlay_open: bool) -> CursorVisibilityAction {
    match (cursor_hidden, overlay_open) {
        (false, true) => CursorVisibilityAction::Hide,
        (true, false) => CursorVisibilityAction::Show,
        _ => CursorVisibilityAction::None,
    }
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

pub fn should_accept_toggle_key(route: &InputRoute, hotkey_window: u64, event_window: u64) -> bool {
    route.event_window != 0
        && event_window != 0
        && (event_window == hotkey_window
            || event_window == route.event_window
            || event_window == route.game_window)
}

pub const fn overlay_hotkey_uses_passive_grab(overlay_open: bool) -> bool {
    !overlay_open
}

fn hotkey_window(route: &InputRoute, root_window: c_ulong) -> c_ulong {
    if route.game_window != 0 {
        route.game_window as c_ulong
    } else if route.event_window != 0 {
        route.event_window as c_ulong
    } else {
        root_window
    }
}

fn resolve_hotkey_window(
    display: *mut Display,
    window: c_ulong,
    root: c_ulong,
    query: QueryTree,
    free: Free,
) -> c_ulong {
    // Wine's Vulkan surface can be a child of the keyboard focus window.
    // A passive grab on that child only activates while it contains the pointer.
    // Keep the grab scoped to this game's top-level ancestor, never the desktop.
    let mut current = window;
    for _ in 0..64 {
        if current == 0 || current == root {
            break;
        }
        let mut tree_root = 0;
        let mut parent = 0;
        let mut children = std::ptr::null_mut();
        let mut count = 0;
        let ok = unsafe {
            query(
                display,
                current,
                &mut tree_root,
                &mut parent,
                &mut children,
                &mut count,
            )
        };
        if !children.is_null() {
            unsafe { free(children.cast()) };
        }
        if ok == 0 || parent == 0 || parent == tree_root || parent == current {
            break;
        }
        current = parent;
    }
    current
}

pub fn should_forward_to_game(route: &InputRoute, overlay_open: bool) -> bool {
    let _ = (route, overlay_open);
    false
}

const fn focus_out_closes_overlay(mode: c_int) -> bool {
    mode == 0 || mode == 3
}

fn focus_out_suspends_input(
    window: c_ulong,
    hotkey: c_ulong,
    overlay_open: bool,
    mode: c_int,
) -> bool {
    focus_out_closes_overlay(mode) && (window == hotkey || overlay_open)
}

fn insert_press_is_toggle(down: bool, last_release_time: Option<c_ulong>, time: c_ulong) -> bool {
    !down && last_release_time != Some(time)
}

fn insert_release_time(was_down: bool, time: c_ulong) -> Option<c_ulong> {
    was_down.then_some(time)
}

pub struct X11Input {
    _library: Library,
    _xtest_library: Option<Library>,
    _xfixes_library: Option<Library>,
    display: *mut Display,
    route: InputRoute,
    root_window: c_ulong,
    hotkey_window: c_ulong,
    event_window: c_ulong,
    insert_keycode: u8,
    overlay_open: bool,
    owner: InputOwner,
    insert_down: bool,
    last_insert_release_time: Option<c_ulong>,
    pointer_mode: PointerMode,
    cursor_hidden: bool,
    last_root: Option<[i32; 2]>,
    last_position: Option<[f32; 2]>,
    pointer_present: bool,
    pending: Pending,
    next_event: NextEvent,
    grab_keyboard: GrabKeyboard,
    ungrab_keyboard: UngrabKeyboard,
    grab_pointer: GrabPointer,
    ungrab_pointer: UngrabPointer,
    grab_key: GrabKey,
    ungrab_key: UngrabKey,
    flush: Flush,
    close_display: CloseDisplay,
    query_tree: QueryTree,
    free: Free,
    select_input: SelectInput,
    warp_pointer: WarpPointer,
    fake_button: Option<FakeButtonEvent>,
    fake_relative_motion: Option<FakeRelativeMotionEvent>,
    hide_cursor: Option<HideCursor>,
    show_cursor: Option<ShowCursor>,
    empty_input_region: c_ulong,
    destroy_region: Option<DestroyRegion>,
    set_window_shape_region: Option<SetWindowShapeRegion>,
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
        let grab_key = load::<GrabKey>(&library, b"XGrabKey\0")?;
        let ungrab_key = load::<UngrabKey>(&library, b"XUngrabKey\0")?;
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
        let query_tree = load::<QueryTree>(&library, b"XQueryTree\0")?;
        let free = load::<Free>(&library, b"XFree\0")?;
        let hotkey_window =
            resolve_hotkey_window(display, hotkey_window(&route, root), root, query_tree, free);
        let event_window = if route.event_window == 0 {
            root
        } else {
            route.event_window as c_ulong
        };
        unsafe {
            select_input(display, event_window, PASSIVE_EVENT_MASK);
            if hotkey_window != 0 && hotkey_window != event_window {
                select_input(display, hotkey_window, 1 | 2 | (1 << 21));
            }
        }
        let xtest_library = unsafe { Library::new("libXtst.so.6") }.ok();
        let fake_button = xtest_library
            .as_ref()
            .and_then(|library| load::<FakeButtonEvent>(library, b"XTestFakeButtonEvent\0").ok());
        let fake_relative_motion = xtest_library.as_ref().and_then(|library| {
            load::<FakeRelativeMotionEvent>(library, b"XTestFakeRelativeMotionEvent\0").ok()
        });
        let xfixes_library = unsafe { Library::new("libXfixes.so.3") }.ok();
        let xfixes = xfixes_library.as_ref().and_then(|library| {
            let query_extension =
                load::<QueryXFixesExtension>(library, b"XFixesQueryExtension\0").ok()?;
            let query_version =
                load::<QueryXFixesVersion>(library, b"XFixesQueryVersion\0").ok()?;
            let hide_cursor = load::<HideCursor>(library, b"XFixesHideCursor\0").ok()?;
            let show_cursor = load::<ShowCursor>(library, b"XFixesShowCursor\0").ok()?;
            let create_region = load::<CreateRegion>(library, b"XFixesCreateRegion\0").ok()?;
            let destroy_region = load::<DestroyRegion>(library, b"XFixesDestroyRegion\0").ok()?;
            let set_window_shape_region =
                load::<SetWindowShapeRegion>(library, b"XFixesSetWindowShapeRegion\0").ok()?;
            let mut event_base = 0;
            let mut error_base = 0;
            let mut major = 0;
            let mut minor = 0;
            let available = unsafe {
                query_extension(display, &mut event_base, &mut error_base) != 0
                    && query_version(display, &mut major, &mut minor) != 0
                    && (major > 4 || major == 4 && minor >= 0)
            };
            available.then(|| {
                let empty_input_region = unsafe { create_region(display, std::ptr::null(), 0) };
                (
                    hide_cursor,
                    show_cursor,
                    destroy_region,
                    set_window_shape_region,
                    empty_input_region,
                )
            })
        });
        if xfixes.is_none() {
            eprintln!("TuxScaling input: XFixes input and cursor support is unavailable");
        }
        let (hide_cursor, show_cursor, destroy_region, set_window_shape_region, empty_input_region) =
            xfixes.map_or(
                (None, None, None, None, 0),
                |(hide, show, destroy, shape, region)| {
                    (Some(hide), Some(show), Some(destroy), Some(shape), region)
                },
            );
        let input = Self {
            _library: library,
            _xtest_library: xtest_library,
            _xfixes_library: xfixes_library,
            display,
            route,
            root_window: root,
            hotkey_window,
            event_window: if route.event_window == 0 {
                root
            } else {
                route.event_window as c_ulong
            },
            insert_keycode,
            overlay_open: false,
            owner: InputOwner::Game,
            insert_down: false,
            last_insert_release_time: None,
            pointer_mode: PointerMode::Absolute,
            cursor_hidden: false,
            last_root: None,
            last_position: None,
            pointer_present: false,
            pending,
            next_event,
            grab_keyboard,
            ungrab_keyboard,
            grab_pointer,
            ungrab_pointer,
            grab_key,
            ungrab_key,
            flush,
            close_display,
            query_tree,
            free,
            select_input,
            warp_pointer,
            fake_button,
            fake_relative_motion,
            hide_cursor,
            show_cursor,
            empty_input_region,
            destroy_region,
            set_window_shape_region,
        };
        input.set_presenter_interactive(false);
        input.grab_insert_hotkey();
        let sync = load::<Sync>(&input._library, b"XSync\0")?;
        unsafe { sync(display, 0) };
        Ok(input)
    }

    pub fn set_pointer_mode(&mut self, mode: PointerMode) {
        self.pointer_mode = mode;
        self.last_root = None;
    }

    pub fn update_route(&mut self, route: InputRoute) -> bool {
        let event_window = if route.event_window == 0 {
            self.event_window
        } else {
            route.event_window as c_ulong
        };
        if event_window != self.event_window {
            return false;
        }
        let next_hotkey = resolve_hotkey_window(
            self.display,
            hotkey_window(&route, self.root_window),
            self.root_window,
            self.query_tree,
            self.free,
        );
        let changed = self.hotkey_window != next_hotkey;
        if changed && self.owner == InputOwner::Game {
            self.release_insert_hotkey();
        }
        self.route = route;
        if changed {
            if self.hotkey_window != self.event_window {
                unsafe { (self.select_input)(self.display, self.hotkey_window, 0) };
            }
            self.hotkey_window = next_hotkey;
            if next_hotkey != self.event_window {
                unsafe { (self.select_input)(self.display, next_hotkey, 1 | 2 | (1 << 21)) };
            }
            if self.owner == InputOwner::Game {
                self.grab_insert_hotkey();
            }
            unsafe { (self.flush)(self.display) };
        }
        true
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
                    let is_insert = event.keycode as u8 == self.insert_keycode;
                    if event.send_event != 0
                        || if is_insert {
                            !should_accept_toggle_key(&self.route, self.hotkey_window, event.window)
                        } else {
                            !should_accept_event(&self.route, event.window)
                        }
                    {
                        continue;
                    }
                    if is_insert {
                        let toggle = insert_press_is_toggle(
                            self.insert_down,
                            self.last_insert_release_time,
                            event.time,
                        );
                        self.insert_down = true;
                        if toggle {
                            if self.overlay_open {
                                self.release_overlay();
                                frame.toggle_overlay = true;
                            } else {
                                match self.try_open_overlay() {
                                    Ok(()) => frame.toggle_overlay = true,
                                    Err(error) => {
                                        eprintln!("TuxScaling input: cannot open overlay: {error}")
                                    }
                                }
                            }
                            if frame.toggle_overlay {
                                frame.events.push(key_event(true));
                            }
                        }
                    }
                }
                KEY_RELEASE => {
                    let event = unsafe { event.key };
                    let is_insert = event.keycode as u8 == self.insert_keycode;
                    if event.send_event != 0
                        || if is_insert {
                            !should_accept_toggle_key(&self.route, self.hotkey_window, event.window)
                        } else {
                            !should_accept_event(&self.route, event.window)
                        }
                    {
                        continue;
                    }
                    if is_insert {
                        let was_down = self.insert_down;
                        self.insert_down = false;
                        self.last_insert_release_time = insert_release_time(was_down, event.time);
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
                    if (!should_accept_event(&self.route, event.window)
                        && event.window != self.hotkey_window)
                        || event.send_event != 0
                        || event.detail == 2
                    {
                        // NotifyInferior means focus moved within the game subtree.
                        continue;
                    }
                    if self.overlay_open {
                        frame.events.push(Event::WindowFocused(kind == FOCUS_IN));
                    }
                    if kind == FOCUS_IN && self.owner == InputOwner::Suspended {
                        self.owner = on_focus_in(self.owner);
                        self.grab_insert_hotkey();
                        unsafe { (self.flush)(self.display) };
                    }
                    // XGrabKeyboard/XUngrabKeyboard emit focus transitions
                    // with NotifyGrab/NotifyUngrab. They acknowledge the
                    // overlay's own input grab, rather than a real focus loss.
                    if kind == FOCUS_OUT
                        && focus_out_suspends_input(
                            event.window,
                            self.hotkey_window,
                            self.overlay_open,
                            event.mode,
                        )
                    {
                        self.insert_down = false;
                        self.pointer_present = false;
                        if self.overlay_open {
                            self.release_overlay();
                            frame.toggle_overlay = true;
                        }
                        self.owner = InputOwner::Suspended;
                        self.release_insert_hotkey();
                        unsafe { (self.flush)(self.display) };
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

    pub fn try_open_overlay(&mut self) -> Result<(), InputError> {
        if self.owner == InputOwner::Overlay {
            return Ok(());
        }
        if !can_open(self.owner)
            || self.route.event_window == 0
            || (self.route.event_window != self.route.game_window
                && (self.empty_input_region == 0 || self.set_window_shape_region.is_none()))
        {
            return Err(InputError::RoutingUnavailable);
        }
        self.release_insert_hotkey();
        self.set_presenter_interactive(true);
        let keyboard = unsafe { (self.grab_keyboard)(self.display, self.event_window, 0, 1, 1, 0) };
        let pointer = if keyboard == 0 {
            unsafe {
                (self.grab_pointer)(
                    self.display,
                    self.event_window,
                    0,
                    POINTER_GRAB_MASK,
                    GRAB_MODE_ASYNC,
                    GRAB_MODE_ASYNC,
                    0,
                    0,
                    0,
                )
            }
        } else {
            -1
        };
        let outcome = open_transition(keyboard == 0, pointer == 0);
        if outcome.owner != InputOwner::Overlay {
            unsafe {
                if outcome.release_keyboard {
                    (self.ungrab_keyboard)(self.display, 0);
                }
                if outcome.release_pointer {
                    (self.ungrab_pointer)(self.display, 0);
                }
            }
            self.set_presenter_interactive(false);
            self.grab_insert_hotkey();
            unsafe { (self.flush)(self.display) };
            return Err(if keyboard != 0 {
                InputError::GrabFailed("keyboard", keyboard)
            } else {
                InputError::GrabFailed("pointer", pointer)
            });
        }
        self.owner = InputOwner::Overlay;
        self.overlay_open = true;
        self.update_cursor_visibility();
        unsafe { (self.flush)(self.display) };
        Ok(())
    }

    pub fn release_overlay(&mut self) {
        let transition = release_transition(self.owner);
        if !transition.restore_game_route {
            return;
        }
        if transition.release_grabs {
            unsafe {
                (self.ungrab_keyboard)(self.display, 0);
                (self.ungrab_pointer)(self.display, 0);
            }
        }
        self.overlay_open = false;
        self.owner = transition.owner;
        self.set_presenter_interactive(false);
        self.grab_insert_hotkey();
        self.update_cursor_visibility();
        unsafe { (self.flush)(self.display) };
    }

    fn grab_insert_hotkey(&self) {
        if !overlay_hotkey_uses_passive_grab(self.overlay_open) || self.insert_keycode == 0 {
            return;
        }
        for window in self.hotkey_windows() {
            unsafe {
                (self.grab_key)(
                    self.display,
                    c_int::from(self.insert_keycode),
                    ANY_MODIFIER,
                    window,
                    0,
                    GRAB_MODE_ASYNC,
                    GRAB_MODE_ASYNC,
                );
            }
        }
    }

    fn release_insert_hotkey(&self) {
        if self.insert_keycode == 0 {
            return;
        }
        for window in self.hotkey_windows() {
            unsafe {
                (self.ungrab_key)(
                    self.display,
                    c_int::from(self.insert_keycode),
                    ANY_MODIFIER,
                    window,
                );
            }
        }
    }

    fn hotkey_windows(&self) -> impl Iterator<Item = c_ulong> {
        std::iter::once(self.hotkey_window).filter(|window| *window != 0)
    }

    fn set_presenter_interactive(&self, interactive: bool) {
        if self.route.event_window == 0
            || self.route.event_window == self.route.game_window
            || self.empty_input_region == 0
        {
            return;
        }
        if let Some(set_window_shape_region) = self.set_window_shape_region {
            unsafe {
                set_window_shape_region(
                    self.display,
                    self.event_window,
                    SHAPE_INPUT,
                    0,
                    0,
                    if interactive {
                        0
                    } else {
                        self.empty_input_region
                    },
                );
            }
        }
    }

    fn update_cursor_visibility(&mut self) {
        match cursor_visibility_action(self.cursor_hidden, self.overlay_open) {
            CursorVisibilityAction::Hide => self.hide_native_cursor(),
            CursorVisibilityAction::Show => self.show_native_cursor(),
            CursorVisibilityAction::None => {}
        }
    }

    fn hide_native_cursor(&mut self) {
        if self.cursor_hidden {
            return;
        }
        if let Some(hide_cursor) = self.hide_cursor
            && self.event_window != 0
        {
            unsafe {
                hide_cursor(self.display, self.event_window);
                (self.flush)(self.display);
            }
            self.cursor_hidden = true;
        }
    }

    fn show_native_cursor(&mut self) {
        if !self.cursor_hidden {
            return;
        }
        if let Some(show_cursor) = self.show_cursor
            && self.event_window != 0
        {
            unsafe {
                show_cursor(self.display, self.event_window);
                (self.flush)(self.display);
            }
            self.cursor_hidden = false;
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
                self.release_overlay();
                self.owner = InputOwner::Destroyed;
                self.release_insert_hotkey();
                if self.empty_input_region != 0
                    && let Some(destroy_region) = self.destroy_region
                {
                    destroy_region(self.display, self.empty_input_region);
                }
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
        CursorOwner, CursorVisibilityAction, InputFrame, InputRoute, PointerMode, PointerViewport,
        cursor_visibility_action, pointer_button, route_pointer, should_accept_event,
        should_accept_toggle_key, should_forward_to_game,
    };
    use egui::PointerButton;

    #[test]
    fn cursor_visibility_does_not_show_an_unhidden_cursor() {
        assert_eq!(
            cursor_visibility_action(false, false),
            CursorVisibilityAction::None
        );
        assert_eq!(
            cursor_visibility_action(false, true),
            CursorVisibilityAction::Hide
        );
        assert_eq!(
            cursor_visibility_action(true, false),
            CursorVisibilityAction::Show
        );
        assert_eq!(
            cursor_visibility_action(true, true),
            CursorVisibilityAction::None
        );
    }

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
    fn separate_presenter_never_synthesizes_pointer_input_back_to_itself() {
        let route = route();

        assert!(should_accept_event(&route, route.event_window));
        assert!(!should_accept_event(&route, route.game_window));
        assert!(should_accept_toggle_key(&route, 0x1, route.event_window));
        assert!(should_accept_toggle_key(&route, 0x1, route.game_window));
        assert!(should_accept_toggle_key(&route, 0x1, 0x1));
        assert!(!should_accept_toggle_key(&route, 0x1, 0x2));
        assert!(!should_forward_to_game(&route, false));
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
    fn closed_overlay_uses_a_passive_insert_grab() {
        assert!(super::overlay_hotkey_uses_passive_grab(false));
        assert!(!super::overlay_hotkey_uses_passive_grab(true));
    }

    #[test]
    fn closed_presenter_focus_loss_keeps_game_hotkey_active() {
        assert!(!super::focus_out_suspends_input(20, 10, false, 0));
        assert!(super::focus_out_suspends_input(10, 10, false, 0));
        assert!(super::focus_out_suspends_input(20, 10, true, 0));
        assert!(!super::focus_out_suspends_input(20, 10, true, 1));
    }

    #[test]
    fn insert_hotkey_is_scoped_to_each_game_window() {
        let first = route();
        let second = InputRoute {
            game_window: 0x500,
            event_window: 0xa00,
            ..first
        };
        assert_eq!(super::hotkey_window(&first, 1), 0x400);
        assert_eq!(super::hotkey_window(&second, 1), 0x500);
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

    #[test]
    fn real_focus_loss_closes_overlay_but_grab_notifications_do_not() {
        assert!(!super::focus_out_closes_overlay(1));
        assert!(!super::focus_out_closes_overlay(2));
        assert!(super::focus_out_closes_overlay(0));
        // Alt-Tab while XGrabKeyboard is active reports NotifyWhileGrabbed.
        assert!(super::focus_out_closes_overlay(3));
    }

    #[test]
    fn insert_auto_repeat_pairs_do_not_toggle_again() {
        assert!(super::insert_press_is_toggle(false, None, 100));
        assert!(!super::insert_press_is_toggle(true, None, 140));
        assert!(!super::insert_press_is_toggle(false, Some(180), 180));
        assert!(super::insert_press_is_toggle(false, Some(180), 220));
    }

    #[test]
    fn stray_release_does_not_suppress_the_first_insert_press() {
        let release_time = super::insert_release_time(false, 100);
        assert!(super::insert_press_is_toggle(false, release_time, 100));
        let repeat_time = super::insert_release_time(true, 140);
        assert!(!super::insert_press_is_toggle(false, repeat_time, 140));
    }
}
