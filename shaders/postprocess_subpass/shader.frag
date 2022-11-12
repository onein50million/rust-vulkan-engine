#version 450
layout (input_attachment_index = 0, binding = 0) uniform subpassInput rasterAlbedo;
layout (input_attachment_index = 0, binding = 1) uniform subpassInput rasterDepth;
layout (input_attachment_index = 0, binding = 2) uniform subpassInput rasterNormals;
layout (input_attachment_index = 0, binding = 3) uniform subpassInput rasterRoughMetalAo;


layout(binding = 2, std140) uniform UniformBufferObject {
    mat4 proj;
} ubos;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

layout(push_constant) uniform PushConstants{
    mat4 view_inverse;
    float time;
} pushConstant;


vec3 unproject_point(vec3 point){
    mat4 transform = ubos.proj;
    float inverse_denom = transform[2][3] / (point.z + transform[2][2]);
    return vec3(point.x * inverse_denom / transform[0][0],point.y * inverse_denom / transform[1][1],-inverse_denom);
}

const float PI = 3.14;
const float E = 2.71828;

//https://stackoverflow.com/a/32246825
vec3 depthWorldPosition(float depth, vec2 uv){
    float z = depth * 2.0 - 1.0;
    
    vec4 clip_space_position = vec4(uv * 2.0 - 1.0, z, 1.0);
    vec4 view_space_position = inverse(ubos.proj) * clip_space_position;

    // Perspective division
    view_space_position /= view_space_position.w;

    vec4 worldSpacePosition = pushConstant.view_inverse * view_space_position;

    return worldSpacePosition.xyz;
}

void main()
{
    vec3 scene_albedo = subpassLoad(rasterAlbedo).rgb;
    float scene_depth = subpassLoad(rasterDepth).r;
    vec3 world_position = depthWorldPosition(scene_depth, inUV);
    
    outFragColor = vec4(scene_albedo, 1.0);
}