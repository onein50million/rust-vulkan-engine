//https://www.shadertoy.com/view/3d2GDW
//https://github.com/blender/blender/blob/master/source/blender/gpu/shaders/material/gpu_shader_material_tex_noise.glsl
//https://www.shadertoy.com/view/4djSRW
//https://github.com/blender/blender/blob/master/source/blender/gpu/shaders/material/gpu_shader_material_fractal_noise.glsl

const int NUM_RANDOM = 100;
const int NUM_LIGHTS = 1;
const int NUM_MODELS = 1000;
const float PI = 3.14159;

const int IS_CUBEMAP = 1;
const int IS_HIGHLIGHTED = 2;
const int IS_VIEWMODEL = 4;
const int IS_GLOBE = 8;
struct Light{
    vec4 position;
    vec4 color;
};
layout(binding = 0, std140) uniform UniformBufferObject {
    vec4 random[NUM_RANDOM];
    Light lights[NUM_LIGHTS];
    uint player_index;
    uint num_lights;
    uint map_mode;
    uint selected_province;
    vec2 mouse_position;
    vec2 screen_size;
    float time;
    float player_position_x;
    float player_position_y;
    float player_position_z;
} ubos;

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