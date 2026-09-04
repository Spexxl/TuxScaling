use egui::{ClippedPrimitive, Context, RawInput, Rect, TexturesDelta, vec2};

pub const CRATE_NAME: &str = "tuxscaling-overlay";

#[derive(Debug, Clone, Default)]
pub struct FrameDiagnostics {
    pub frame_id: u64,
    pub state: String,
    pub mode: String,
    pub capture_ms: f32,
    pub motion_ms: f32,
    pub overlay_ms: f32,
}

pub fn render_diagnostics(
    context: &Context,
    size: [u32; 2],
    diagnostics: &FrameDiagnostics,
) -> OverlayFrame {
    let output = context.run(
        RawInput {
            screen_rect: Some(Rect::from_min_size(
                Default::default(),
                vec2(size[0] as f32, size[1] as f32),
            )),
            ..Default::default()
        },
        |context| {
            egui::Window::new("TuxScaling").show(context, |ui| {
                ui.label(&diagnostics.state);
                ui.label(format!(
                    "{} x {} | Frame {}",
                    size[0], size[1], diagnostics.frame_id
                ));
                ui.label(format!("View: {}", diagnostics.mode));
                ui.label("Motion: current -> previous, pixels");
                ui.label("Hue: direction | Brightness: magnitude");
                ui.label(format!(
                    "GPU ms: capture {:.2} | flow {:.2} | overlay {:.2}",
                    diagnostics.capture_ms, diagnostics.motion_ms, diagnostics.overlay_ms
                ));
            });
        },
    );
    OverlayFrame {
        pixels_per_point: output.pixels_per_point,
        primitives: context.tessellate(output.shapes, output.pixels_per_point),
        textures_delta: output.textures_delta,
    }
}

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
