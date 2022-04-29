//https://www.shadertoy.com/view/3d2GDW
//https://github.com/blender/blender/blob/master/source/blender/gpu/shaders/material/gpu_shader_material_tex_noise.glsl
//https://www.shadertoy.com/view/4djSRW
//https://github.com/blender/blender/blob/master/source/blender/gpu/shaders/material/gpu_shader_material_fractal_noise.glsl


const int NUM_NOISE_OCTAVES = 5;


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
const int IS_VIEW_PROJ_MATRIX_IGNORED = 16;
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
    float input_tangent;
    float output_tangent;
    float _padding1;
    float _padding2;
    Bone bones[NUM_BONES_PER_BONESET];
};
layout(binding = 2) buffer ShaderStorageBufferObject {
    BoneSet bone_sets[NUM_BONE_SETS];
} ssbo;

float hash(float p) { p = fract(p * 0.011); p *= p + 7.5; p *= p + p; return fract(p); }
float hash(vec2 p) {vec3 p3 = fract(vec3(p.xyx) * 0.13); p3 += dot(p3, p3.yzx + 3.333); return fract((p3.x + p3.y) * p3.z); }


//https://www.shadertoy.com/view/4dS3Wd
float noise(vec3 x) {
    const vec3 step = vec3(110, 241, 171);

    vec3 i = floor(x);
    vec3 f = fract(x);
 
    // For performance, compute the base input to a 1D hash from the integer part of the argument and the 
    // incremental change to the 1D based on the 3D -> 1D wrapping
    float n = dot(i, step);

    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
               mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}

float fbm(vec3 x) {
	float v = 0.0;
	float a = 0.5;
	vec3 shift = vec3(100);
	for (int i = 0; i < NUM_NOISE_OCTAVES; ++i) {
		v += a * noise(x);
		x = x * 2.0 + shift;
		a *= 0.5;
	}
	return v;
}