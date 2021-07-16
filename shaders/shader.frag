#version 450

const int num_models = 100;


layout(binding = 1) uniform sampler2D texSampler[num_models];

layout(push_constant) uniform PushConstants{
    int uniform_index;
    int texture_index;
} pushConstant;


layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler[pushConstant.texture_index], fragTexCoord);
}