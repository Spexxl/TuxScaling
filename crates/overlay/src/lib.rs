use egui::{ClippedPrimitive, Context, RawInput, Rect, TexturesDelta, vec2};

pub const CRATE_NAME: &str = "tuxscaling-overlay";

#[derive(Debug)]
pub struct OverlayFrame {
    pub pixels_per_point: f32,
    pub primitives: Vec<ClippedPrimitive>,
    pub textures_delta: TexturesDelta,
}

pub fn render_smoke_frame(
    context: &Context,
    size_in_pixels: [u32; 2],
    pixels_per_point: f32,
) -> OverlayFrame {
    assert!(pixels_per_point.is_finite() && pixels_per_point > 0.0);

    let screen_size = vec2(
        size_in_pixels[0] as f32 / pixels_per_point,
        size_in_pixels[1] as f32 / pixels_per_point,
    );
    let output = context.run(
        RawInput {
            screen_rect: Some(Rect::from_min_size(Default::default(), screen_size)),
            ..Default::default()
        },
        |context| {
            egui::Window::new("TuxScaling").show(context, |ui| {
                ui.heading("TuxScaling");
                ui.label("Vulkan overlay smoke test");
            });
        },
    );
    let output_pixels_per_point = output.pixels_per_point;
    let primitives = context.tessellate(output.shapes, output_pixels_per_point);

    OverlayFrame {
        pixels_per_point: output_pixels_per_point,
        primitives,
        textures_delta: output.textures_delta,
    }
}

#[derive(Debug)]
pub struct OverlayState {
    pub visible: bool,
    pub context: Context,
}

impl OverlayState {
    pub fn new() -> Self {
        Self {
            visible: false,
            context: Context::default(),
        }
    }
}

impl Default for OverlayState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{OverlayState, render_smoke_frame};

    #[test]
    fn starts_hidden() {
        assert!(!OverlayState::new().visible);
    }

    #[test]
    fn smoke_frame_contains_meshes_for_a_swapchain_extent() {
        let context = egui::Context::default();
        let _first_frame = render_smoke_frame(&context, [1280, 720], 1.0);
        let second_frame = render_smoke_frame(&context, [1280, 720], 1.0);

        assert!(
            second_frame
                .primitives
                .iter()
                .any(|primitive| matches!(primitive.primitive, egui::epaint::Primitive::Mesh(_)))
        );
    }
}
