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


layout(push_constant) uniform PushConstants{
    int uniform_index;
    int texture_index;
    float constant;

} pushConstant;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;


void main() {
    gl_Position = ubos.proj[pushConstant.uniform_index] * ubos.view[pushConstant.uniform_index] * ubos.model[pushConstant.uniform_index] * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;

}