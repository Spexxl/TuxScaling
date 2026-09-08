#![cfg(feature = "fidelityfx")]

#[test]
fn fidelityfx_input_shader_contains_deterministic_adapter_rules() {
    let shader = include_str!("../../../shaders/upscaler/fidelityfx_input.comp");
    assert!(shader.contains("confidence > 0.05"));
    assert!(shader.contains("max(reactive, disocclusion * (1.0 - confidence))"));
    assert!(shader.contains("max(composition, disocclusion)"));
    assert!(shader.contains("clamp(texelFetch(depth_image, pixel, 0).r, 0.0, 1.0)"));
    assert!(shader.contains(": 1.0;"));
}
