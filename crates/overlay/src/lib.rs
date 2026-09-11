use egui::{ClippedPrimitive, Context, RawInput, Rect, TexturesDelta, vec2};
use tuxscaling_config::{DebugView, JitterMode, MotionQuality, Upscaler};

pub const CRATE_NAME: &str = "tuxscaling-overlay";

#[derive(Debug, Clone, Default)]
pub struct FrameDiagnostics {
    pub frame_id: u64,
    pub state: String,
    pub mode: String,
    pub upscaler: Upscaler,
    pub requested_upscaler: Option<Upscaler>,
    pub active_upscaler: Upscaler,
    pub quality: MotionQuality,
    pub requested_quality: Option<MotionQuality>,
    pub requested_guidance_scale: Option<f32>,
    pub requested_jitter_mode: Option<JitterMode>,
    pub requested_debug_view: Option<DebugView>,
    pub jitter_mode: JitterMode,
    pub debug_view: DebugView,
    pub guidance_scale: f32,
    pub game_extent: [u32; 2],
    pub guidance_extent: [u32; 2],
    pub output_extent: [u32; 2],
    pub presentation_mode: String,
    pub window_mode: String,
    pub monitor: String,
    pub frame_delta_ms: f32,
    pub frame_delta_raw_ms: f32,
    pub frame_delta_validated_ms: f32,
    pub frame_delta_smoothed_ms: f32,
    pub reset_reason: String,
    pub motion_state: String,
    pub confidence_state: String,
    pub reactive_state: String,
    pub disocclusion_state: String,
    pub exposure_state: String,
    pub depth_state: String,
    pub composition_state: String,
    pub jitter_state: String,
    pub depth_semantics: String,
    pub capture_cpu_ms: f32,
    pub motion_cpu_ms: f32,
    pub guidance_cpu_ms: f32,
    pub reconstruction_cpu_ms: f32,
    pub overlay_cpu_ms: f32,
    pub capture_ms: f32,
    pub luma_ms: f32,
    pub pyramid_ms: f32,
    pub forward_flow_ms: f32,
    pub backward_flow_ms: f32,
    pub confidence_ms: f32,
    pub stats_ms: f32,
    pub scene_ms: f32,
    pub invalidate_ms: f32,
    pub motion_ms: f32,
    pub reactive_ms: f32,
    pub exposure_ms: f32,
    pub depth_ms: f32,
    pub guidance_ms: f32,
    pub guidance_total_ms: f32,
    pub reconstruction_ms: f32,
    pub overlay_ms: f32,
    pub full_injected_ms: f32,
    pub p95_ms: f32,
}

pub fn resolution_mode(game_extent: [u32; 2], output_extent: [u32; 2]) -> &'static str {
    if game_extent == output_extent {
        "Native AA"
    } else {
        "Virtual upscale"
    }
}

pub fn debug_view_label(view: DebugView) -> &'static str {
    match view {
        DebugView::Original => "Original",
        DebugView::Luminance => "Luminance",
        DebugView::Motion => "Motion",
        DebugView::Confidence => "Confidence",
        DebugView::Reconstructed => "Reconstructed",
        DebugView::History => "History",
        DebugView::Reactive => "Reactive",
        DebugView::Disocclusion => "Disocclusion",
        DebugView::Depth => "Depth",
        DebugView::Composition => "Composition",
        DebugView::Exposure => "Exposure",
    }
}

pub fn guidance_scale_request(value: f32) -> Option<f32> {
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
    diagnostics.requested_upscaler = None;
    diagnostics.requested_guidance_scale = None;
    diagnostics.requested_jitter_mode = None;
    diagnostics.requested_debug_view = None;
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
                    egui::ComboBox::from_label("Upscaler")
                        .selected_text(format_upscaler(diagnostics.upscaler))
                        .show_ui(ui, |ui| {
                            for upscaler in
                                [Upscaler::Reference, Upscaler::Fsr314, Upscaler::Off]
                            {
                                if ui
                                    .selectable_value(
                                        &mut diagnostics.upscaler,
                                        upscaler,
                                        format_upscaler(upscaler),
                                    )
                                    .changed()
                                {
                                    diagnostics.requested_upscaler =
                                        upscaler_request(diagnostics.active_upscaler, upscaler);
                                }
                            }
                        });
                    ui.label(format!(
                        "Active upscaler: {}",
                        format_upscaler(diagnostics.active_upscaler)
                    ));
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
                        "Guidance scale: {:.0}%",
                        diagnostics.guidance_scale * 100.0
                    ));
                    if diagnostics.game_extent[0] > 0 && diagnostics.output_extent[0] > 0 {
                        ui.label(format!(
                            "Game: {} x {} | Guidance: {} x {} | Output: {} x {}",
                            diagnostics.game_extent[0],
                            diagnostics.game_extent[1],
                            diagnostics.guidance_extent[0],
                            diagnostics.guidance_extent[1],
                            diagnostics.output_extent[0],
                            diagnostics.output_extent[1],
                        ));
                        ui.label(format!("Mode: {}", diagnostics.presentation_mode));
                        ui.label(format!(
                            "Window: {} | Monitor: {}",
                            diagnostics.window_mode, diagnostics.monitor
                        ));
                    }
                    egui::CollapsingHeader::new("Advanced")
                        .default_open(false)
                        .show(ui, |ui| {
                            let mut scale = diagnostics.guidance_scale;
                            if ui
                                .add(
                                    egui::Slider::new(&mut scale, 0.5..=1.0)
                                        .text("Guidance scale"),
                                )
                                .changed()
                            {
                                diagnostics.requested_guidance_scale = guidance_scale_request(scale);
                            }
                            ui.small(
                                "This does not reduce the game's rendering workload.",
                            );
                            let mut jitter_mode = diagnostics.jitter_mode;
                            egui::ComboBox::from_label("Capture jitter")
                                .selected_text(format_jitter_mode(jitter_mode))
                                .show_ui(ui, |ui| {
                                    for mode in [JitterMode::Off, JitterMode::ExperimentalHalton8] {
                                        if ui
                                            .selectable_value(
                                                &mut jitter_mode,
                                                mode,
                                                format_jitter_mode(mode),
                                            )
                                            .changed()
                                        {
                                            diagnostics.requested_jitter_mode = Some(mode);
                                        }
                                    }
                                });
                            ui.small("Experimental jitter resamples captured color only.");
                            let mut debug_view = diagnostics.debug_view;
                            egui::ComboBox::from_label("Debug view")
                                .selected_text(debug_view_label(debug_view))
                                .show_ui(ui, |ui| {
                                    for view in [
                                        DebugView::Original,
                                        DebugView::Luminance,
                                        DebugView::Motion,
                                        DebugView::Confidence,
                                        DebugView::Reconstructed,
                                        DebugView::History,
                                        DebugView::Reactive,
                                        DebugView::Disocclusion,
                                        DebugView::Depth,
                                        DebugView::Composition,
                                        DebugView::Exposure,
                                    ] {
                                        if ui
                                            .selectable_value(
                                                &mut debug_view,
                                                view,
                                                debug_view_label(view),
                                            )
                                            .changed()
                                        {
                                            diagnostics.requested_debug_view = Some(view);
                                        }
                                    }
                                });
                        });
                    ui.label(format!("Frame delta: {:.2} ms", diagnostics.frame_delta_ms));
                    ui.label(format!(
                        "Delta raw {:.2} | validated {:.2} | smoothed {:.2} ms",
                        diagnostics.frame_delta_raw_ms,
                        diagnostics.frame_delta_validated_ms,
                        diagnostics.frame_delta_smoothed_ms
                    ));
                    ui.label(format!("Reset: {}", diagnostics.reset_reason));
                    ui.label(format!(
                        "Signals: motion {} | confidence {} | reactive {} | disocclusion {}",
                        diagnostics.motion_state,
                        diagnostics.confidence_state,
                        diagnostics.reactive_state,
                        diagnostics.disocclusion_state
                    ));
                    ui.label(format!(
                        "Signals: exposure {} | depth {} ({}) | composition {} | jitter {}",
                        diagnostics.exposure_state,
                        diagnostics.depth_state,
                        diagnostics.depth_semantics,
                        diagnostics.composition_state,
                        diagnostics.jitter_state
                    ));
                    ui.label("Motion: current -> previous, pixels");
                    ui.label("Hue: direction | Brightness: magnitude");
                    ui.label(format!(
                        "GPU ms: capture {:.2} | guidance total {:.2} | reconstruction {:.2} | overlay {:.2}",
                        diagnostics.capture_ms,
                        diagnostics.guidance_total_ms,
                        diagnostics.reconstruction_ms,
                        diagnostics.overlay_ms
                    ));
                    ui.label(format!(
                        "Injected total: {:.2} ms",
                        diagnostics.full_injected_ms
                    ));
                    ui.label(format!(
                        "Passes ms: luma {:.2} | pyramid {:.2} | forward {:.2} | backward {:.2}",
                        diagnostics.luma_ms,
                        diagnostics.pyramid_ms,
                        diagnostics.forward_flow_ms,
                        diagnostics.backward_flow_ms
                    ));
                    ui.label(format!(
                        "Passes ms: confidence {:.2} | stats {:.2} | scene {:.2} | invalidate {:.2} | reactive {:.2} | exposure {:.2} | depth {:.2}",
                        diagnostics.confidence_ms,
                        diagnostics.stats_ms,
                        diagnostics.scene_ms,
                        diagnostics.invalidate_ms,
                        diagnostics.reactive_ms,
                        diagnostics.exposure_ms,
                        diagnostics.depth_ms
                    ));
                    ui.label(format!(
                        "CPU ms: capture {:.2} | motion {:.2} | guidance {:.2} | reconstruction {:.2} | overlay {:.2}",
                        diagnostics.capture_cpu_ms,
                        diagnostics.motion_cpu_ms,
                        diagnostics.guidance_cpu_ms,
                        diagnostics.reconstruction_cpu_ms,
                        diagnostics.overlay_cpu_ms
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
        requested_upscaler: diagnostics.requested_upscaler,
        requested_quality: diagnostics.requested_quality,
        requested_guidance_scale: diagnostics.requested_guidance_scale,
        requested_jitter_mode: diagnostics.requested_jitter_mode,
        requested_debug_view: diagnostics.requested_debug_view,
    }
}

pub fn upscaler_request(active: Upscaler, selected: Upscaler) -> Option<Upscaler> {
    (active != selected).then_some(selected)
}

fn format_upscaler(upscaler: Upscaler) -> &'static str {
    match upscaler {
        Upscaler::Reference => "Reference",
        Upscaler::Fsr314 => "FSR 3.1.4",
        Upscaler::Off => "Off",
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

fn format_jitter_mode(mode: JitterMode) -> &'static str {
    match mode {
        JitterMode::Off => "Off",
        JitterMode::ExperimentalHalton8 => "Experimental Halton 8",
    }
}

#[derive(Debug)]
pub struct OverlayFrame {
    pub pixels_per_point: f32,
    pub primitives: Vec<ClippedPrimitive>,
    pub textures_delta: TexturesDelta,
    pub requested_upscaler: Option<Upscaler>,
    pub requested_quality: Option<MotionQuality>,
    pub requested_guidance_scale: Option<f32>,
    pub requested_jitter_mode: Option<JitterMode>,
    pub requested_debug_view: Option<DebugView>,
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
        requested_upscaler: None,
        requested_quality: None,
        requested_guidance_scale: None,
        requested_jitter_mode: None,
        requested_debug_view: None,
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
    use super::{
        OverlayState, debug_view_label, guidance_scale_request, render_smoke_frame, resolution_mode,
    };
    use tuxscaling_config::{DebugView, Upscaler};

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
    fn accepts_only_guidance_scales_in_the_supported_range() {
        assert_eq!(guidance_scale_request(0.75), Some(0.75));
        assert_eq!(guidance_scale_request(0.49), None);
        assert_eq!(guidance_scale_request(1.01), None);
        assert_eq!(guidance_scale_request(f32::NAN), None);
    }

    #[test]
    fn labels_new_guidance_debug_views() {
        assert_eq!(debug_view_label(DebugView::Depth), "Depth");
        assert_eq!(debug_view_label(DebugView::Composition), "Composition");
        assert_eq!(debug_view_label(DebugView::Exposure), "Exposure");
    }

    #[test]
    fn selecting_fidelityfx_requests_only_the_upscaler() {
        assert_eq!(
            super::upscaler_request(Upscaler::Reference, Upscaler::Fsr314),
            Some(Upscaler::Fsr314)
        );
        assert_eq!(
            super::upscaler_request(Upscaler::Fsr314, Upscaler::Fsr314),
            None
        );
    }

    #[test]
    fn selecting_off_requests_disabled_upscaling() {
        assert_eq!(
            super::upscaler_request(Upscaler::Fsr314, Upscaler::Off),
            Some(Upscaler::Off)
        );
        assert_eq!(super::format_upscaler(Upscaler::Off), "Off");
    }
}
