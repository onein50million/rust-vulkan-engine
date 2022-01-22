//from https://github.com/blender/blender/blob/594f47ecd2d5367ca936cf6fc6ec8168c2b360d0/source/blender/gpu/shaders/material/gpu_shader_material_map_range.glsl#L7
float map_range_linear(float value,
float fromMin,
float fromMax,
float toMin,
float toMax )
{
    float result = clamp(toMin + ((value - fromMin) / (fromMax - fromMin)) * (toMax - toMin), min(toMin,toMax), max(toMax,toMin));
    return result;
}