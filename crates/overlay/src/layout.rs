pub const PANEL_SIZE: [f32; 2] = [420.0, 360.0];

pub fn initial_panel_geometry(output_pixels: [u32; 2], display_scale: f32) -> [f32; 4] {
    let scale = if display_scale.is_finite() && display_scale > 0.0 {
        display_scale
    } else {
        1.0
    };
    let screen = [
        output_pixels[0] as f32 / scale,
        output_pixels[1] as f32 / scale,
    ];
    let width = PANEL_SIZE[0].min((screen[0] - 16.0).max(1.0));
    let height = PANEL_SIZE[1].min((screen[1] - 16.0).max(1.0));
    let x = 200.0_f32.min((screen[0] - width).max(0.0));
    let y = 80.0_f32.min((screen[1] - height).max(0.0));
    [x, y, width, height]
}

#[cfg(test)]
mod tests {
    use super::initial_panel_geometry;

    #[test]
    fn compact_panel_stays_visible_at_common_outputs_and_display_scales() {
        for output in [[1280, 720], [1920, 1080], [3440, 1440]] {
            for scale in [1.0, 1.5, 2.0] {
                let [x, y, width, height] = initial_panel_geometry(output, scale);
                let screen = [output[0] as f32 / scale, output[1] as f32 / scale];
                assert!(x >= 0.0 && y >= 0.0);
                assert!(width > 0.0 && height > 0.0);
                assert!(x + width <= screen[0]);
                assert!(y + height <= screen[1]);
                assert!(width <= 420.0 && height <= 360.0);
            }
        }
        let [x, y, width, height] = initial_panel_geometry([1920, 1080], 1.0);
        assert_eq!([width, height], [420.0, 360.0]);
        assert!(x >= 160.0 && y >= 72.0);
    }
}
