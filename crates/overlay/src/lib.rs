use egui::{ClippedPrimitive, Context, RawInput, Rect, TexturesDelta, vec2};
use tuxscaling_config::MotionQuality;

pub const CRATE_NAME: &str = "tuxscaling-overlay";

#[derive(Debug, Clone, Default)]
pub struct FrameDiagnostics {
    pub frame_id: u64,
    pub state: String,
    pub mode: String,
    pub quality: MotionQuality,
    pub requested_quality: Option<MotionQuality>,
    pub requested_processing_scale: Option<f32>,
    pub processing_scale: f32,
    pub game_extent: [u32; 2],
    pub processing_extent: [u32; 2],
    pub output_extent: [u32; 2],
    pub presentation_mode: String,
    pub frame_delta_ms: f32,
    pub capture_ms: f32,
    pub luma_ms: f32,
    pub pyramid_ms: f32,
    pub forward_flow_ms: f32,
    pub backward_flow_ms: f32,
    pub confidence_ms: f32,
    pub scene_ms: f32,
    pub invalidate_ms: f32,
    pub motion_ms: f32,
    pub reactive_ms: f32,
    pub exposure_ms: f32,
    pub guidance_ms: f32,
    pub reconstruction_ms: f32,
    pub overlay_ms: f32,
    pub p95_ms: f32,
    pub budget_warning: bool,
}

pub fn resolution_mode(game_extent: [u32; 2], output_extent: [u32; 2]) -> &'static str {
    if game_extent == output_extent {
        "Native AA"
    } else {
        "Virtual upscale"
    }
}

pub fn processing_scale_request(value: f32) -> Option<f32> {
    value
        .is_finite()
        .then_some(value)
        .filter(|value| (0.5..=1.0).contains(value))
}

pub fn render_diagnostics(
    context: &Context,
    size: [u32; 2],
    diagnostics: &mut FrameDiagnostics,
    events: &[egui::Event],
    visible: bool,
) -> OverlayFrame {
    diagnostics.requested_quality = None;
    diagnostics.requested_processing_scale = None;
    let output = context.run(
        RawInput {
            screen_rect: Some(Rect::from_min_size(
                Default::default(),
                vec2(size[0] as f32, size[1] as f32),
            )),
            events: events.to_vec(),
            ..Default::default()
        },
        |context| {
            if visible {
                egui::Window::new("TuxScaling").show(context, |ui| {
                    ui.label(&diagnostics.state);
                    ui.label(format!(
                        "{} x {} | Frame {}",
                        size[0], size[1], diagnostics.frame_id
                    ));
                    ui.label(format!("View: {}", diagnostics.mode));
                    egui::ComboBox::from_label("Quality")
                        .selected_text(format_quality(diagnostics.quality))
                        .show_ui(ui, |ui| {
                            for quality in [
                                MotionQuality::Ultra,
                                MotionQuality::High,
                                MotionQuality::Balanced,
                                MotionQuality::Performance,
                            ] {
                                if ui
                                    .selectable_value(
                                        &mut diagnostics.quality,
                                        quality,
                                        format_quality(quality),
                                    )
                                    .changed()
                                {
                                    diagnostics.requested_quality = Some(quality);
                                }
                            }
                        });
                    ui.label(format!(
                        "Processing scale: {:.0}%",
                        diagnostics.processing_scale * 100.0
                    ));
                    if diagnostics.game_extent[0] > 0 && diagnostics.output_extent[0] > 0 {
                        ui.label(format!(
                            "Game: {} x {} | Processing: {} x {} | Output: {} x {}",
                            diagnostics.game_extent[0],
                            diagnostics.game_extent[1],
                            diagnostics.processing_extent[0],
                            diagnostics.processing_extent[1],
                            diagnostics.output_extent[0],
                            diagnostics.output_extent[1],
                        ));
                        ui.label(format!("Mode: {}", diagnostics.presentation_mode));
                    }
                    egui::CollapsingHeader::new("Advanced")
                        .default_open(false)
                        .show(ui, |ui| {
                            let mut scale = diagnostics.processing_scale;
                            if ui
                                .add(
                                    egui::Slider::new(&mut scale, 0.5..=1.0)
                                        .text("Processing scale"),
                                )
                                .changed()
                            {
                                diagnostics.requested_processing_scale =
                                    processing_scale_request(scale);
                            }
                            ui.small(
                                "This does not reduce the game's rendering workload.",
                            );
                        });
                    ui.label(format!("Frame delta: {:.2} ms", diagnostics.frame_delta_ms));
                    if diagnostics.budget_warning {
                        ui.colored_label(
                            egui::Color32::YELLOW,
                            format!(
                                "Warning: temporal work exceeds the {} preset target",
                                format_quality(diagnostics.quality)
                            ),
                        );
                    }
                    ui.label("Motion: current -> previous, pixels");
                    ui.label("Hue: direction | Brightness: magnitude");
                    ui.label(format!(
                        "GPU ms: capture {:.2} | flow {:.2} | guidance {:.2} | reconstruction {:.2} | overlay {:.2}",
                        diagnostics.capture_ms,
                        diagnostics.motion_ms,
                        diagnostics.guidance_ms,
                        diagnostics.reconstruction_ms,
                        diagnostics.overlay_ms
                    ));
                    ui.label(format!(
                        "Passes ms: luma {:.2} | pyramid {:.2} | forward {:.2} | backward {:.2}",
                        diagnostics.luma_ms,
                        diagnostics.pyramid_ms,
                        diagnostics.forward_flow_ms,
                        diagnostics.backward_flow_ms
                    ));
                    ui.label(format!(
                        "Passes ms: confidence {:.2} | scene {:.2} | invalidate {:.2} | reactive {:.2} | exposure {:.2}",
                        diagnostics.confidence_ms,
                        diagnostics.scene_ms,
                        diagnostics.invalidate_ms,
                        diagnostics.reactive_ms,
                        diagnostics.exposure_ms
                    ));
                    if diagnostics.p95_ms > 0.0 {
                        ui.label(format!("Temporal p95: {:.2} ms", diagnostics.p95_ms));
                    }
                });
            }
        },
    );
    OverlayFrame {
        pixels_per_point: output.pixels_per_point,
        primitives: context.tessellate(output.shapes, output.pixels_per_point),
        textures_delta: output.textures_delta,
        requested_quality: diagnostics.requested_quality,
        requested_processing_scale: diagnostics.requested_processing_scale,
    }
}

fn format_quality(quality: MotionQuality) -> &'static str {
    match quality {
        MotionQuality::Ultra => "Ultra",
        MotionQuality::High => "High",
        MotionQuality::Balanced => "Balanced",
        MotionQuality::Performance => "Performance",
    }
}

#[derive(Debug)]
pub struct OverlayFrame {
    pub pixels_per_point: f32,
    pub primitives: Vec<ClippedPrimitive>,
    pub textures_delta: TexturesDelta,
    pub requested_quality: Option<MotionQuality>,
    pub requested_processing_scale: Option<f32>,
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
        requested_quality: None,
        requested_processing_scale: None,
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
    use super::{OverlayState, processing_scale_request, render_smoke_frame, resolution_mode};

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

    #[test]
    fn labels_equal_input_and_output_as_native_aa() {
        assert_eq!(resolution_mode([1920, 1080], [1920, 1080]), "Native AA");
        assert_eq!(
            resolution_mode([1280, 720], [1920, 1080]),
            "Virtual upscale"
        );
    }

    #[test]
    fn accepts_only_processing_scales_in_the_supported_range() {
        assert_eq!(processing_scale_request(0.75), Some(0.75));
        assert_eq!(processing_scale_request(0.49), None);
        assert_eq!(processing_scale_request(1.01), None);
        assert_eq!(processing_scale_request(f32::NAN), None);
    }
}
