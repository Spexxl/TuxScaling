use std::{thread, time::Duration};
use tuxscaling_input::X11Input;
use x11rb::{connection::Connection, protocol::xproto::*};

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
    let mut input = X11Input::connect(window.into()).unwrap();
    let key = |kind| {
        connection
            .send_event(
                false,
                window,
                EventMask::KEY_PRESS | EventMask::KEY_RELEASE,
                KeyPressEvent {
                    response_type: kind,
                    detail: insert,
                    sequence: 0,
                    time: 0,
                    root,
                    event: window,
                    child: 0,
                    root_x: 100,
                    root_y: 100,
                    event_x: 100,
                    event_y: 100,
                    state: KeyButMask::default(),
                    same_screen: true,
                },
            )
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
    key(KEY_PRESS_EVENT);
    assert!(input.poll().toggle_overlay, "Insert must open the overlay");
    key(KEY_RELEASE_EVENT);
    input.poll();
    connection
        .send_event(
            false,
            window,
            EventMask::POINTER_MOTION,
            MotionNotifyEvent {
                response_type: MOTION_NOTIFY_EVENT,
                detail: Motion::NORMAL,
                sequence: 0,
                time: 0,
                root,
                event: window,
                child: 0,
                root_x: 200,
                root_y: 150,
                event_x: 200,
                event_y: 150,
                state: KeyButMask::default(),
                same_screen: true,
            },
        )
        .unwrap()
        .check()
        .unwrap();
    connection.flush().unwrap();
    thread::sleep(Duration::from_millis(30));
    assert!(
        input
            .poll()
            .events
            .iter()
            .any(|event| matches!(event, egui::Event::PointerMoved(_)))
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
