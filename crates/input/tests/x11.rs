use std::{
    thread,
    time::{Duration, Instant},
};
use tuxscaling_input::{CursorOwner, InputFrame, InputRoute, PointerViewport, X11Input};
use x11rb::{
    connection::Connection,
    protocol::{
        xproto::ConnectionExt as XprotoConnectionExt, xproto::*, xtest::ConnectionExt as _,
    },
};

const FOCUS_DEADLINE: Duration = Duration::from_secs(2);
const EVENT_DEADLINE: Duration = Duration::from_secs(2);
const POLL_INTERVAL: Duration = Duration::from_millis(10);
const MAX_TOGGLE_ATTEMPTS: u32 = 5;

/// Round-trip barrier proving the server processed all prior requests.
fn barrier<C: Connection + XprotoConnectionExt>(connection: &C) {
    connection.get_input_focus().unwrap().reply().unwrap();
}

/// Set input focus to `window` and wait until the server reports it effective.
/// A real window manager may steal focus asynchronously, so this polls with a
/// deadline instead of assuming one asynchronous request is enough.
fn ensure_focused<C: Connection + XprotoConnectionExt>(connection: &C, window: u32) {
    let start = Instant::now();
    loop {
        connection
            .set_input_focus(InputFocus::PARENT, window, 0u32)
            .unwrap()
            .check()
            .unwrap();
        let reply = connection.get_input_focus().unwrap().reply().unwrap();
        if reply.focus == window {
            return;
        }
        assert!(
            start.elapsed() < FOCUS_DEADLINE,
            "X11 focus never settled on the test window",
        );
        thread::sleep(POLL_INTERVAL);
    }
}

/// Poll until `condition` holds or the deadline expires; returns the last frame.
fn wait_frame(
    input: &mut X11Input,
    deadline: Duration,
    condition: impl Fn(&InputFrame) -> bool,
) -> InputFrame {
    let start = Instant::now();
    loop {
        let frame = input.poll();
        if condition(&frame) || start.elapsed() >= deadline {
            return frame;
        }
        thread::sleep(POLL_INTERVAL);
    }
}

#[test]
#[ignore = "requires an X11 desktop and temporarily focuses a test window"]
// Delivery leg: synthetic XTEST/core input must reach this window, which needs
// a real Xorg server. On Wayland-session XWayland (Mutter) the compositor
// routes all device events to its own focused surface, so XTEST keys/buttons
// and warped motion never arrive even with X focus held and matching keycodes
// (probed 2026-09-13: focus verified, keycodes 118/118, timestamped focus,
// keys 'a'+Insert, motion, buttons all undelivered while Focus/Property events
// arrived). Production is unaffected: it steers events with grabs, never with
// SetInputFocus.
fn keyboard_and_pointer_work_after_resize_and_release_on_close() {
    let (connection, screen) = x11rb::connect(None).unwrap();
    let root = connection.setup().roots[screen].root;
    let window = connection.generate_id().unwrap();
    connection
        .create_window(
            0,
            window,
            root,
            0,
            0,
            320,
            240,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &CreateWindowAux::new().override_redirect(1),
        )
        .unwrap()
        .check()
        .unwrap();
    connection.map_window(window).unwrap().check().unwrap();
    connection
        .configure_window(window, &ConfigureWindowAux::new().width(640).height(480))
        .unwrap()
        .check()
        .unwrap();
    ensure_focused(&connection, window);
    let setup = connection.setup();
    let mapping = connection
        .get_keyboard_mapping(setup.min_keycode, setup.max_keycode - setup.min_keycode + 1)
        .unwrap()
        .reply()
        .unwrap();
    let insert = mapping
        .keysyms
        .chunks(mapping.keysyms_per_keycode as usize)
        .position(|keys| keys.first() == Some(&0xff63))
        .unwrap() as u8
        + setup.min_keycode;
    let mut input = X11Input::connect(InputRoute::new(
        window.into(),
        window.into(),
        [640, 480],
        [640, 480],
        PointerViewport {
            offset: [0.0, 0.0],
            extent: [640.0, 480.0],
        },
    ))
    .unwrap();
    let key = |kind| {
        connection
            .xtest_fake_input(kind, insert, 0, root, 100, 100, 0)
            .unwrap()
            .check()
            .unwrap();
        connection.flush().unwrap();
        // Round-trip barrier: the server processed the fake input, so the
        // generated event is queued for delivery before polling starts.
        barrier(&connection);
    };
    key(KEY_RELEASE_EVENT);
    connection
        .warp_pointer(0u32, window, 0, 0, 0, 0, 100, 100)
        .unwrap()
        .check()
        .unwrap();
    ensure_focused(&connection, window);
    connection.flush().unwrap();
    input.poll();
    // Opening the overlay flips `overlay_open` exactly once per Insert press.
    // A real desktop may steal focus between attempts (a FocusOut closes the
    // overlay again), so re-assert focus and retry a bounded number of times.
    // A genuinely broken Insert path fails every attempt.
    let mut opened = None;
    for _ in 0..MAX_TOGGLE_ATTEMPTS {
        ensure_focused(&connection, window);
        key(KEY_PRESS_EVENT);
        let frame = wait_frame(&mut input, EVENT_DEADLINE, |frame| frame.toggle_overlay);
        key(KEY_RELEASE_EVENT);
        input.poll();
        if frame.toggle_overlay && frame.cursor_owner == CursorOwner::Overlay {
            opened = Some(frame);
            break;
        }
    }
    let opened = opened.expect("Insert must open the overlay");
    assert_eq!(opened.cursor_owner, CursorOwner::Overlay);
    let released = wait_frame(&mut input, EVENT_DEADLINE, |_| true);
    assert_eq!(released.cursor_owner, CursorOwner::Overlay);
    connection
        .warp_pointer(0u32, window, 0, 0, 0, 0, 200, 150)
        .unwrap()
        .check()
        .unwrap();
    connection.flush().unwrap();
    barrier(&connection);
    let frame = wait_frame(&mut input, EVENT_DEADLINE, |frame| {
        frame
            .events
            .iter()
            .any(|event| matches!(event, egui::Event::PointerMoved(_)))
    });
    assert!(
        frame
            .events
            .iter()
            .any(|event| matches!(event, egui::Event::PointerMoved(_))),
        "expected a genuine X11 pointer event after opening the overlay, got {frame:?}",
    );
    let mut closed = false;
    for _ in 0..MAX_TOGGLE_ATTEMPTS {
        ensure_focused(&connection, window);
        key(KEY_PRESS_EVENT);
        let frame = wait_frame(&mut input, EVENT_DEADLINE, |frame| frame.toggle_overlay);
        key(KEY_RELEASE_EVENT);
        input.poll();
        if frame.toggle_overlay && frame.cursor_owner == CursorOwner::Native {
            closed = true;
            break;
        }
    }
    assert!(closed, "Insert must close the overlay");
    drop(input);
    let reply = connection
        .grab_keyboard(false, window, 0u32, GrabMode::ASYNC, GrabMode::ASYNC)
        .unwrap()
        .reply()
        .unwrap();
    assert_eq!(reply.status, GrabStatus::SUCCESS);
    connection.ungrab_keyboard(0u32).unwrap().check().unwrap();
    connection.destroy_window(window).unwrap().check().unwrap();
}
