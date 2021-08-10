#version 450

const int NUM_MODELS = 100;
const int NUM_RANDOM = 100;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model[NUM_MODELS];
    mat4 view[NUM_MODELS];
    mat4 proj[NUM_MODELS];
    float random[NUM_RANDOM];
    int player_index;
    int value2;
    int value3;
    int value4;
} ubos;

layout(binding = 1) uniform sampler2D texSampler[NUM_MODELS];

layout(push_constant) uniform PushConstants{
    int uniform_index;
    int texture_index;
    float constant;
} pushConstant;


layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler[pushConstant.texture_index], fragTexCoord) * pushConstant.constant;
}