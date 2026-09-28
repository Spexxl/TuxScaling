#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputOwner {
    Game,
    Overlay,
    Suspended,
    Destroyed,
}

pub fn can_open(owner: InputOwner) -> bool {
    matches!(owner, InputOwner::Game | InputOwner::Overlay)
}

pub fn on_focus_in(owner: InputOwner) -> InputOwner {
    if owner == InputOwner::Suspended {
        InputOwner::Game
    } else {
        owner
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenTransition {
    pub owner: InputOwner,
    pub release_keyboard: bool,
    pub release_pointer: bool,
    pub presenter_interactive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReleaseTransition {
    pub owner: InputOwner,
    pub release_grabs: bool,
    pub restore_game_route: bool,
}

pub fn release_transition(owner: InputOwner) -> ReleaseTransition {
    let overlay_owned = owner == InputOwner::Overlay;
    ReleaseTransition {
        owner: if overlay_owned {
            InputOwner::Game
        } else {
            owner
        },
        release_grabs: overlay_owned,
        restore_game_route: overlay_owned,
    }
}

pub fn open_transition(keyboard_grabbed: bool, pointer_grabbed: bool) -> OpenTransition {
    let captured = keyboard_grabbed && pointer_grabbed;
    OpenTransition {
        owner: if captured {
            InputOwner::Overlay
        } else {
            InputOwner::Game
        },
        release_keyboard: keyboard_grabbed && !captured,
        release_pointer: pointer_grabbed && !captured,
        presenter_interactive: captured,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        InputOwner, OpenTransition, ReleaseTransition, open_transition, release_transition,
    };

    #[test]
    fn focus_loss_suspends_open_until_focus_returns() {
        assert!(!super::can_open(InputOwner::Suspended));
        assert!(!super::can_open(InputOwner::Destroyed));
        assert!(super::can_open(InputOwner::Game));
        assert_eq!(super::on_focus_in(InputOwner::Suspended), InputOwner::Game);
        assert_eq!(
            super::on_focus_in(InputOwner::Destroyed),
            InputOwner::Destroyed
        );
    }

    #[test]
    fn repeated_release_does_not_repeat_x11_actions() {
        assert_eq!(
            release_transition(InputOwner::Overlay),
            ReleaseTransition {
                owner: InputOwner::Game,
                release_grabs: true,
                restore_game_route: true,
            }
        );
        assert_eq!(
            release_transition(InputOwner::Game),
            ReleaseTransition {
                owner: InputOwner::Game,
                release_grabs: false,
                restore_game_route: false,
            }
        );
        assert_eq!(
            release_transition(InputOwner::Suspended),
            ReleaseTransition {
                owner: InputOwner::Suspended,
                release_grabs: false,
                restore_game_route: false,
            }
        );
    }

    #[test]
    fn partial_grab_rolls_back_owned_keyboard_and_keeps_presenter_transparent() {
        assert_eq!(
            open_transition(true, false),
            OpenTransition {
                owner: InputOwner::Game,
                release_keyboard: true,
                release_pointer: false,
                presenter_interactive: false,
            }
        );
    }

    #[test]
    fn failed_keyboard_grab_never_claims_pointer_or_presenter() {
        assert_eq!(
            open_transition(false, false),
            OpenTransition {
                owner: InputOwner::Game,
                release_keyboard: false,
                release_pointer: false,
                presenter_interactive: false,
            }
        );
    }

    #[test]
    fn full_capture_is_required_for_overlay_ownership() {
        assert_eq!(
            open_transition(true, true),
            OpenTransition {
                owner: InputOwner::Overlay,
                release_keyboard: false,
                release_pointer: false,
                presenter_interactive: true,
            }
        );
    }
}
