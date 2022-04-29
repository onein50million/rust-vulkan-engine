#version 450

layout(push_constant) uniform PushConstants{
    mat4 model_view_projection;
} pushConstant;

layout(location = 0) in vec3 position;

void main() {
    gl_Position = pushConstant.model_view_projection * vec4(position,1.0);
}
