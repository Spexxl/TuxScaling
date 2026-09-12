use crate::mapping::LogicalSwapchainHandle;
use crate::state::{SwapchainState, is_reconfiguring_swapchain, is_retired_swapchain, swapchains};
use ash::vk;
use ash::vk::Handle;
use std::sync::{Arc, Mutex};

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SwapchainRouteKind {
    Direct,
    CurrentVirtual,
    Retired,
    UnknownTagged,
}

#[cfg(test)]
pub(crate) fn classify_swapchain_route(
    tracked: bool,
    virtualized: bool,
    retired: bool,
    tagged: bool,
) -> SwapchainRouteKind {
    if tracked && virtualized {
        SwapchainRouteKind::CurrentVirtual
    } else if retired {
        SwapchainRouteKind::Retired
    } else if tagged {
        SwapchainRouteKind::UnknownTagged
    } else {
        SwapchainRouteKind::Direct
    }
}

#[derive(Clone)]
pub(crate) enum SwapchainRoute {
    Direct(vk::SwapchainKHR),
    CurrentVirtual {
        state: Arc<Mutex<SwapchainState>>,
        physical: vk::SwapchainKHR,
        game_surface: vk::SurfaceKHR,
        present_surface: Option<vk::SurfaceKHR>,
    },
}

pub(crate) fn resolve_swapchain_route(
    swapchain: vk::SwapchainKHR,
) -> Result<SwapchainRoute, vk::Result> {
    if is_retired_swapchain(swapchain) || is_reconfiguring_swapchain(swapchain) {
        return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
    }
    let state = swapchains()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&swapchain)
        .cloned();
    let Some(state) = state else {
        if LogicalSwapchainHandle::is_reserved(swapchain.as_raw()) {
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
        }
        return Ok(SwapchainRoute::Direct(swapchain));
    };
    let state_guard = state.lock().unwrap_or_else(|error| error.into_inner());
    let Some(mapping) = state_guard.mapping.as_ref() else {
        return Ok(SwapchainRoute::Direct(swapchain));
    };
    if state_guard.lifecycle.blocks_frame_operations() {
        return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
    }
    if state_guard.contract.is_none() || mapping.generation() != state_guard.generation {
        return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
    }
    Ok(SwapchainRoute::CurrentVirtual {
        state: state.clone(),
        physical: state_guard.physical_handle,
        game_surface: state_guard.game_surface,
        present_surface: state_guard.present_surface,
    })
}

#[cfg(test)]
mod tests {
    use super::{SwapchainRouteKind, classify_swapchain_route};

    #[test]
    fn route_classification_covers_direct_current_retired_and_unknown_tagged() {
        assert_eq!(
            classify_swapchain_route(false, false, false, false),
            SwapchainRouteKind::Direct
        );
        assert_eq!(
            classify_swapchain_route(true, true, false, true),
            SwapchainRouteKind::CurrentVirtual
        );
        assert_eq!(
            classify_swapchain_route(false, false, true, true),
            SwapchainRouteKind::Retired
        );
        assert_eq!(
            classify_swapchain_route(false, false, false, true),
            SwapchainRouteKind::UnknownTagged
        );
    }

    #[test]
    fn current_virtual_route_wins_over_tagged_namespace_checks() {
        assert_eq!(
            classify_swapchain_route(true, true, false, true),
            SwapchainRouteKind::CurrentVirtual
        );
    }
}
