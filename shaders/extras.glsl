//https://www.shadertoy.com/view/3d2GDW
//https://github.com/blender/blender/blob/master/source/blender/gpu/shaders/material/gpu_shader_material_tex_noise.glsl
//https://www.shadertoy.com/view/4djSRW
//https://github.com/blender/blender/blob/master/source/blender/gpu/shaders/material/gpu_shader_material_fractal_noise.glsl

const int NUM_RANDOM = 100;
const int NUM_LIGHTS = 1;
const int NUM_MODELS = 1000;
const int NUM_BONES_PER_BONESET = 256;
const int NUM_BONE_SETS = 256;
const float PI = 3.14159;

const int IS_CUBEMAP = 1;
const int IS_HIGHLIGHTED = 2;
const int IS_VIEWMODEL = 4;
const int IS_GLOBE = 8;
struct Light{
    vec4 position;
    vec4 color;
};

layout(push_constant) uniform PushConstants{
    mat4 model;
    int texture_index;
    uint bitfield; //32 bits, LSB is cubemap flag
    uint animation_frames; // u8 previous_frame, u8 next_frame, u16 UNORM progress
} pushConstant;


layout(binding = 0, std140) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec4 random[NUM_RANDOM];
    Light lights[NUM_LIGHTS];
    uint player_index;
    uint num_lights;
    uint map_mode;
    float exposure;
    vec2 mouse_position;
    vec2 screen_size;
    float time;
    float player_position_x;
    float player_position_y;
    float player_position_z;

} ubos;

struct Bone {
    mat4 matrix;
};

struct BoneSet {
    Bone bones[NUM_BONES_PER_BONESET];
};
layout(binding = 2) buffer ShaderStorageBufferObject {
    BoneSet bone_sets[NUM_BONE_SETS];
} ssbo;
