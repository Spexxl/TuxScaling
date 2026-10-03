use std::{
    thread,
    time::{Duration, Instant},
};
use tuxscaling_input::{CursorOwner, InputFrame, InputRoute, PointerViewport, X11Input};
use x11rb::{
    connection::Connection,
    protocol::{
        shape::{ConnectionExt as ShapeConnectionExt, SK},
        xproto::ConnectionExt as XprotoConnectionExt,
        xproto::*,
        xtest::ConnectionExt as _,
    },
};

const FOCUS_DEADLINE: Duration = Duration::from_secs(2);
const EVENT_DEADLINE: Duration = Duration::from_secs(2);
const POLL_INTERVAL: Duration = Duration::from_millis(10);
const MAX_TOGGLE_ATTEMPTS: u32 = 5;

#[test]
#[ignore = "requires an X11 desktop and temporarily focuses test windows"]
fn switching_application_while_overlay_is_open_releases_both_grabs() {
    let (connection, screen) = x11rb::connect(None).unwrap();
    let root = connection.setup().roots[screen].root;
    let original_focus = connection.get_input_focus().unwrap().reply().unwrap().focus;
    let game = connection.generate_id().unwrap();
    let presenter = connection.generate_id().unwrap();
    let other = connection.generate_id().unwrap();
    for window in [game, presenter, other] {
        connection
            .create_window(
                0,
                window,
                root,
                0,
                0,
                64,
                64,
                0,
                WindowClass::INPUT_OUTPUT,
                0,
                &CreateWindowAux::new().override_redirect(1),
            )
            .unwrap()
            .check()
            .unwrap();
        connection.map_window(window).unwrap().check().unwrap();
    }
    ensure_focused(&connection, game);
    let mut input = X11Input::connect(InputRoute::new(
        presenter.into(),
        game.into(),
        [64, 64],
        [64, 64],
        PointerViewport {
            offset: [0.0, 0.0],
            extent: [64.0, 64.0],
        },
    ))
    .unwrap();
    input.poll();
    input.try_open_overlay().unwrap();
    input.poll();
    assert_eq!(input.cursor_owner(), CursorOwner::Overlay);
    ensure_focused(&connection, other);
    let frame = wait_frame(&mut input, EVENT_DEADLINE, |frame| {
        frame.cursor_owner == CursorOwner::Native
    });
    assert_eq!(
        frame.cursor_owner,
        CursorOwner::Native,
        "a real focus change during a keyboard grab must close the overlay"
    );
    assert!(
        connection
            .shape_get_rectangles(presenter, SK::INPUT)
            .unwrap()
            .reply()
            .unwrap()
            .rectangles
            .is_empty()
    );
    assert_eq!(
        connection
            .grab_keyboard(false, other, 0u32, GrabMode::ASYNC, GrabMode::ASYNC)
            .unwrap()
            .reply()
            .unwrap()
            .status,
        GrabStatus::SUCCESS
    );
    connection.ungrab_keyboard(0u32).unwrap().check().unwrap();
    assert_eq!(
        connection
            .grab_pointer(
                false,
                other,
                EventMask::POINTER_MOTION,
                GrabMode::ASYNC,
                GrabMode::ASYNC,
                0u32,
                0u32,
                0u32
            )
            .unwrap()
            .reply()
            .unwrap()
            .status,
        GrabStatus::SUCCESS
    );
    connection.ungrab_pointer(0u32).unwrap().check().unwrap();
    drop(input);
    connection
        .set_input_focus(InputFocus::PARENT, original_focus, 0u32)
        .unwrap()
        .check()
        .unwrap();
    for window in [game, presenter, other] {
        connection.destroy_window(window).unwrap().check().unwrap();
    }
}

#[test]
#[ignore = "requires an X11 server"]
fn insert_grab_uses_game_ancestor_when_vulkan_surface_is_a_child() {
    let (connection, screen) = x11rb::connect(None).unwrap();
    let root = connection.setup().roots[screen].root;
    let parent = connection.generate_id().unwrap();
    let child = connection.generate_id().unwrap();
    let second_parent = connection.generate_id().unwrap();
    let second_child = connection.generate_id().unwrap();
    for (window, ancestor) in [
        (parent, root),
        (child, parent),
        (second_parent, root),
        (second_child, second_parent),
    ] {
        connection
            .create_window(
                0,
                window,
                ancestor,
                0,
                0,
                64,
                64,
                0,
                WindowClass::INPUT_OUTPUT,
                0,
                &CreateWindowAux::new().override_redirect(1),
            )
            .unwrap()
            .check()
            .unwrap();
    }
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
    let route = InputRoute::new(
        child.into(),
        child.into(),
        [64, 64],
        [64, 64],
        PointerViewport {
            offset: [0.0, 0.0],
            extent: [64.0, 64.0],
        },
    );
    let mut input = X11Input::connect(route).unwrap();
    // A second client must find Insert reserved on the keyboard ancestor.
    // Reserving only the render child misses keys when the pointer is on the presenter.
    let collision = connection
        .grab_key(
            false,
            parent,
            ModMask::ANY,
            insert,
            GrabMode::ASYNC,
            GrabMode::ASYNC,
        )
        .unwrap()
        .check();
    assert!(
        matches!(collision,
            Err(x11rb::errors::ReplyError::X11Error(ref error))
            if error.error_kind == x11rb::protocol::ErrorKind::Access
        ),
        "Insert must be grabbed on the game ancestor: {collision:?}"
    );
    // Independent games must reserve their own ancestors without a root grab.
    let second = X11Input::connect(InputRoute {
        event_window: second_child.into(),
        game_window: second_child.into(),
        ..route
    })
    .unwrap();
    let collision = connection
        .grab_key(
            false,
            second_parent,
            ModMask::ANY,
            insert,
            GrabMode::ASYNC,
            GrabMode::ASYNC,
        )
        .unwrap()
        .check();
    assert!(matches!(collision,
        Err(x11rb::errors::ReplyError::X11Error(ref error))
        if error.error_kind == x11rb::protocol::ErrorKind::Access));
    drop(second);
    // A route change releases the old ancestor and reserves the new one.
    assert!(input.update_route(InputRoute {
        game_window: second_child.into(),
        ..route
    }));
    connection
        .grab_key(
            false,
            parent,
            ModMask::ANY,
            insert,
            GrabMode::ASYNC,
            GrabMode::ASYNC,
        )
        .unwrap()
        .check()
        .unwrap();
    connection
        .ungrab_key(insert, parent, ModMask::ANY)
        .unwrap()
        .check()
        .unwrap();
    let collision = connection
        .grab_key(
            false,
            second_parent,
            ModMask::ANY,
            insert,
            GrabMode::ASYNC,
            GrabMode::ASYNC,
        )
        .unwrap()
        .check();
    assert!(matches!(collision,
        Err(x11rb::errors::ReplyError::X11Error(ref error))
        if error.error_kind == x11rb::protocol::ErrorKind::Access));
    drop(input);
    connection
        .grab_key(
            false,
            second_parent,
            ModMask::ANY,
            insert,
            GrabMode::ASYNC,
            GrabMode::ASYNC,
        )
        .unwrap()
        .check()
        .unwrap();
    connection.destroy_window(parent).unwrap().check().unwrap();
    connection
        .destroy_window(second_parent)
        .unwrap()
        .check()
        .unwrap();
}

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

#[test]
#[ignore = "requires an X11 desktop and temporarily focuses a test window"]
fn presenter_accepts_input_only_while_the_overlay_is_open() {
    let (connection, screen) = x11rb::connect(None).unwrap();
    let root = connection.setup().roots[screen].root;
    let root_geometry = connection.get_geometry(root).unwrap().reply().unwrap();
    let game_window = connection.generate_id().unwrap();
    connection
        .create_window(
            0,
            game_window,
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
    connection.map_window(game_window).unwrap().check().unwrap();
    connection.flush().unwrap();
    let presenter = tuxscaling_display::PresenterWindow::new(
        game_window.into(),
        tuxscaling_display::Monitor::new(tuxscaling_display::Rect::new(
            0,
            0,
            root_geometry.width.into(),
            root_geometry.height.into(),
        )),
    )
    .unwrap();
    let presenter_window = presenter.window();
    let mut input = X11Input::connect(InputRoute::new(
        presenter_window.into(),
        game_window.into(),
        [root_geometry.width.into(), root_geometry.height.into()],
        [320, 240],
        PointerViewport {
            offset: [0.0, 0.0],
            extent: [root_geometry.width.into(), root_geometry.height.into()],
        },
    ))
    .unwrap();
    let input_rectangles = || {
        connection
            .shape_get_rectangles(presenter_window, SK::INPUT)
            .unwrap()
            .reply()
            .unwrap()
            .rectangles
    };
    assert!(input_rectangles().is_empty());

    ensure_focused(&connection, game_window);
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
    let key = |kind| {
        connection
            .xtest_fake_input(kind, insert, 0, root, 100, 100, 0)
            .unwrap()
            .check()
            .unwrap();
        connection.flush().unwrap();
        barrier(&connection);
    };

    let mut opened = None;
    for _ in 0..MAX_TOGGLE_ATTEMPTS {
        ensure_focused(&connection, game_window);
        key(KEY_PRESS_EVENT);
        let frame = wait_frame(&mut input, EVENT_DEADLINE, |frame| frame.toggle_overlay);
        key(KEY_RELEASE_EVENT);
        input.poll();
        if frame.toggle_overlay && frame.cursor_owner == CursorOwner::Overlay {
            opened = Some(frame);
            break;
        }
    }
    assert!(opened.is_some(), "Insert must open the overlay");
    assert!(!input_rectangles().is_empty());

    let mut closed = None;
    for _ in 0..MAX_TOGGLE_ATTEMPTS {
        key(KEY_PRESS_EVENT);
        let frame = wait_frame(&mut input, EVENT_DEADLINE, |frame| frame.toggle_overlay);
        key(KEY_RELEASE_EVENT);
        input.poll();
        if frame.toggle_overlay && frame.cursor_owner == CursorOwner::Native {
            closed = Some(frame);
            break;
        }
    }
    assert!(closed.is_some(), "Insert must close the overlay");
    assert!(input_rectangles().is_empty());

    drop(input);
    drop(presenter);
    connection
        .destroy_window(game_window)
        .unwrap()
        .check()
        .unwrap();
}
