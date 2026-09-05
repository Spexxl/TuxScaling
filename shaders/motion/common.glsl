layout(local_size_x=8, local_size_y=8) in;
layout(binding=0) uniform sampler2D color_frame;
layout(std430,binding=1) buffer Current { float current_luma[]; };
layout(std430,binding=2) buffer Previous { float previous_luma[]; };
layout(std430,binding=3) buffer Forward { vec4 forward_flow[]; };
layout(std430,binding=4) buffer Backward { vec4 backward_flow[]; };
layout(std430,binding=5) buffer Metadata { uint scene_cut; uint consistent; float distance_l1; uint reserved; };
layout(rgba8,binding=6) uniform image2D visualization;
layout(rg16f,binding=7) uniform image2D motion_image;
layout(r8,binding=8) uniform image2D confidence_image;
layout(push_constant) uniform Params {
    uint width; uint height; uint offset; uint grid_offset;
    uint parent_width; uint parent_height; uint parent_offset; uint parent_grid_offset;
    uint level; uint coarsest; uint valid; uint decode_srgb;
    uint mode; float scene_distance_threshold; float scene_consistency_threshold; uint direction;
} p;
