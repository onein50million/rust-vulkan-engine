#version 450

layout (input_attachment_index = 0, binding = 0) uniform subpassInput rasterAlbedo;
layout (input_attachment_index = 0, binding = 1) uniform subpassInput rasterDepth;
layout (input_attachment_index = 0, binding = 2) uniform subpassInput rasterNormals;
layout (input_attachment_index = 0, binding = 3) uniform subpassInput rasterRoughMetalAo;
layout(binding = 4) uniform usampler3D voxelSDF;

layout(binding = 5, std140) uniform UniformBufferObject {
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
const int MAX_MARCHES = 500;
const float MAX_DISTANCE = 1000.0;

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
const float VOXEL_SCALE = 0.1;
//https://stackoverflow.com/a/49726668
float get_voxel_sdf(vec3 position){
    // position *= VOXEL_SCALE;
    vec3 sdf_size = textureSize(voxelSDF,0) * VOXEL_SCALE;
    // float box_sdf = length(max(abs(position) - sdf_size, 0.0));
    // if (box_sdf > 0.0){
    //     return box_sdf;
    // }
    return texture(voxelSDF, position/sdf_size).r * VOXEL_SCALE;
}

void main()
{
    vec3 scene_albedo = subpassLoad(rasterAlbedo).rgb;
    // outFragColor = vec4(scene_albedo, 1.0);
    // return;
    float scene_depth = subpassLoad(rasterDepth).r;
    vec3 world_position = depthWorldPosition(scene_depth, inUV);
    
    vec3 origin = pushConstant.view_inverse[3].xyz;
    vec3 screen_space_target = vec3(inUV.x * 2.0 - 1.0, inUV.y * 2.0 - 1.0, -1.0);
    vec4 v4_target = pushConstant.view_inverse * vec4(unproject_point(screen_space_target.xyz), 0.0);
    vec3 target = normalize(v4_target.xyz);
    origin.y *= -1.0;
    target.y *= -1.0;

    const float threshold = 1.0*VOXEL_SCALE;
    
    vec3 current_position = origin;
    for(int i = 0; i < MAX_MARCHES, distance(current_position, origin) < MAX_DISTANCE; i++){
        float light_strength = float(MAX_MARCHES - i) / float(MAX_MARCHES);
        float dist = get_voxel_sdf(current_position);

        if (dist < threshold){
            // outFragColor = vec4(vec3(1.0), 1.0);
            outFragColor = vec4((sin(current_position))*light_strength, 1.0);
            return;
        }
        dist = max(0.1*VOXEL_SCALE, dist - 0.5 * VOXEL_SCALE);
        current_position += target * 0.01;
    }

    outFragColor = vec4(vec3(0.0), 1.0);


    // int voxels[LENGTH][DEPTH][HEIGHT] = {0};
    // voxels[0][0][0] = 1;

    // vec3 sdf_size = textureSize(voxelSDF,0);


    // outFragColor = vec4(scene_albedo, 1.0);
}