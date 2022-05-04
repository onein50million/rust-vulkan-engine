#version 450

layout(push_constant) uniform PushConstants{
    mat4 model_view_projection;
} pushConstant;

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 fragColor;


void main() {
    gl_Position = pushConstant.model_view_projection * vec4(position,1.0);
    fragColor = color;
}
