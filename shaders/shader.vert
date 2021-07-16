#version 450

const int num_models = 100;


layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view[num_models];
    mat4 proj;
} ubos;

layout(push_constant) uniform PushConstants{
    int index;
} pushConstant;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;


void main() {
    gl_Position = ubos.proj * ubos.view[pushConstant.index] * ubos.model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;

}