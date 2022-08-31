#version 450

layout(push_constant) uniform PushConstants{
    mat4 model_view_projection;
} pushConstant;

layout(location = 0) in vec3 position;

layout(location = 0) flat out int VertexIndex;
layout(location = 1) out vec3 worldPosition;

void main() {
    gl_Position = pushConstant.model_view_projection * vec4(position,1.0);
    VertexIndex = gl_VertexIndex;
    worldPosition = position;
}
