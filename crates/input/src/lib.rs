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
const INSERT_KEYSYM: c_ulong = 0xff63;
const POINTER_EVENT_MASK: c_ulong = (1 << 2) | (1 << 3) | (1 << 6);

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
union XEvent {
    type_: c_int,
    key: KeyEvent,
    button: ButtonEvent,
    motion: MotionEvent,
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
type GrabKey =
    unsafe extern "C" fn(*mut Display, c_int, c_uint, c_ulong, c_int, c_int, c_int) -> c_int;

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
}

pub struct X11Input {
    _library: Library,
    display: *mut Display,
    window: c_ulong,
    insert_keycode: u8,
    overlay_open: bool,
    pending: Pending,
    next_event: NextEvent,
    grab_keyboard: GrabKeyboard,
    ungrab_keyboard: UngrabKeyboard,
    grab_pointer: GrabPointer,
    ungrab_pointer: UngrabPointer,
    flush: Flush,
    close_display: CloseDisplay,
}

unsafe impl Send for X11Input {}

impl X11Input {
    pub fn connect(window: u64) -> Result<Self, InputError> {
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
        let display = unsafe { open_display(std::ptr::null()) };
        if display.is_null() {
            return Err(InputError::DisplayUnavailable);
        }
        let screen = unsafe { default_screen(display) };
        let root = unsafe { root_window(display, screen) };
        let insert_keycode = unsafe { keysym_to_keycode(display, INSERT_KEYSYM) };
        let grab_key = load::<GrabKey>(&library, b"XGrabKey\0")?;
        let select_input = load::<SelectInput>(&library, b"XSelectInput\0")?;
        unsafe {
            select_input(
                display,
                if window == 0 { root } else { window as c_ulong },
                3 | POINTER_EVENT_MASK as c_long,
            );
            grab_key(
                display,
                insert_keycode.into(),
                1 << 15,
                if window == 0 { root } else { window as c_ulong },
                0,
                1,
                1,
            );
        }
        let input = Self {
            _library: library,
            display,
            window: if window == 0 { root } else { window as c_ulong },
            insert_keycode,
            overlay_open: false,
            pending,
            next_event,
            grab_keyboard,
            ungrab_keyboard,
            grab_pointer,
            ungrab_pointer,
            flush,
            close_display,
        };
        input.update_grab();
        let sync = load::<Sync>(&input._library, b"XSync\0")?;
        unsafe { sync(display, 0) };
        Ok(input)
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
                    if event.keycode as u8 == self.insert_keycode {
                        frame.toggle_overlay = true;
                        self.overlay_open = !self.overlay_open;
                        self.update_grab();
                        frame.events.push(key_event(true));
                    }
                }
                KEY_RELEASE => {
                    let event = unsafe { event.key };
                    if event.keycode as u8 == self.insert_keycode {
                        frame.events.push(key_event(false));
                    }
                }
                MOTION_NOTIFY => {
                    let event = unsafe { event.motion };
                    frame.events.push(Event::PointerMoved(Pos2::new(
                        event.x as f32,
                        event.y as f32,
                    )));
                }
                BUTTON_PRESS => {
                    let event = unsafe { event.button };
                    if let Some(button) = pointer_button(event.button as u8) {
                        frame.events.push(Event::PointerButton {
                            pos: Pos2::new(event.x as f32, event.y as f32),
                            button,
                            pressed: true,
                            modifiers: modifiers(event.state),
                        });
                    } else if event.button == 4 || event.button == 5 {
                        frame.events.push(Event::MouseWheel {
                            unit: egui::MouseWheelUnit::Line,
                            delta: Vec2::new(0.0, if event.button == 4 { 1.0 } else { -1.0 }),
                            modifiers: modifiers(event.state),
                        });
                    }
                }
                BUTTON_RELEASE => {
                    let event = unsafe { event.button };
                    if let Some(button) = pointer_button(event.button as u8) {
                        frame.events.push(Event::PointerButton {
                            pos: Pos2::new(event.x as f32, event.y as f32),
                            button,
                            pressed: false,
                            modifiers: modifiers(event.state),
                        });
                    }
                }
                _ => {}
            }
        }
        frame
    }

    fn update_grab(&self) {
        if self.overlay_open {
            unsafe {
                (self.grab_keyboard)(self.display, self.window, 0, 1, 1, 0);
                (self.grab_pointer)(
                    self.display,
                    self.window,
                    0,
                    POINTER_EVENT_MASK,
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
}

impl Drop for X11Input {
    fn drop(&mut self) {
        if !self.display.is_null() {
            unsafe {
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
    use super::pointer_button;
    use egui::PointerButton;

    #[test]
    fn maps_x11_buttons_to_egui() {
        assert_eq!(pointer_button(1), Some(PointerButton::Primary));
        assert_eq!(pointer_button(3), Some(PointerButton::Secondary));
        assert_eq!(pointer_button(8), None);
    }
}
