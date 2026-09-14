#![cfg(feature = "fidelityfx")]

#[test]
fn fidelityfx_input_shader_contains_deterministic_adapter_rules() {
    let shader = include_str!("../../../shaders/upscaler/fidelityfx_input.comp");
    assert!(shader.contains("params.use_estimated_motion != 0u"));
    assert!(shader.contains("float fsr_depth_value = 0.0"));
    assert!(shader.contains("float fsr_reactive_value = 0.0"));
    assert!(shader.contains("float fsr_composition_value = 0.0"));
    assert!(!shader.contains("disocclusion * (1.0 - confidence)"));
    assert!(!shader.contains("max(composition, disocclusion)"));
}
