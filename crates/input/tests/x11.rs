use std::{thread, time::Duration};
use tuxscaling_input::{CursorOwner, InputRoute, PointerViewport, X11Input};
use x11rb::{
    connection::Connection,
    protocol::{xproto::*, xtest::ConnectionExt as _},
};

#[test]
#[ignore = "requires an X11 desktop and temporarily focuses a test window"]
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
    connection
        .set_input_focus(InputFocus::PARENT, window, 0u32)
        .unwrap()
        .check()
        .unwrap();
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
        thread::sleep(Duration::from_millis(30));
    };
    key(KEY_RELEASE_EVENT);
    connection
        .warp_pointer(0u32, window, 0, 0, 0, 0, 100, 100)
        .unwrap()
        .check()
        .unwrap();
    connection
        .set_input_focus(InputFocus::PARENT, window, 0u32)
        .unwrap()
        .check()
        .unwrap();
    connection.flush().unwrap();
    thread::sleep(Duration::from_millis(30));
    input.poll();
    key(KEY_PRESS_EVENT);
    let opened = input.poll();
    assert!(opened.toggle_overlay, "Insert must open the overlay");
    assert_eq!(opened.cursor_owner, CursorOwner::Overlay);
    key(KEY_RELEASE_EVENT);
    let released = input.poll();
    assert_eq!(released.cursor_owner, CursorOwner::Overlay);
    connection
        .warp_pointer(0u32, window, 0, 0, 0, 0, 200, 150)
        .unwrap()
        .check()
        .unwrap();
    connection.flush().unwrap();
    thread::sleep(Duration::from_millis(30));
    let frame = input.poll();
    assert!(
        frame
            .events
            .iter()
            .any(|event| matches!(event, egui::Event::PointerMoved(_))),
        "expected a genuine X11 pointer event after opening the overlay, got {frame:?}",
    );
    key(KEY_PRESS_EVENT);
    assert!(input.poll().toggle_overlay, "Insert must close the overlay");
    key(KEY_RELEASE_EVENT);
    input.poll();
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
