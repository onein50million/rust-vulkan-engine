#version 450
#include "extras.glsl"


const uint MAX_UINT = 4294967295;





layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in uint inTextureType;
layout(location = 5) in uvec4 bone_indices;
layout(location = 6) in vec4 bone_weights;


layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec4 fragTangent;
layout(location = 3) out vec2 fragTexCoord;

layout(location = 4) out vec3 worldPosition;
layout(location = 5) out uint textureType;

mat4 mix(mat4 x, mat4 y, float a){
    return x * (1.0-a) + y * a;
}

//https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/gltfTutorial_007_Animations.md
mat4 cubic_spline(mat4 previous_matrix, float previous_tangent, mat4 next_matrix, float next_tangent, float interpolation_value){

    float t = interpolation_value;
    float t2 = t * t;
    float t3 = t2* t;

    return (2.0 * t3 - 3.0 * t2 + 1.0) * previous_matrix + (t3 - 2.0 * t2 + t) * previous_tangent + (-2.0 * t3 + 3.0 * t2) * next_matrix + (t3 - t2) * next_tangent;

}
float cubic_spline(float start, float previous_tangent, float end, float next_tangent, float interpolation_value){

    float t = interpolation_value;
    float t_2 = t * t;
    float t_3 = t_2* t;

    return (2.0 * t_3 - 3.0 * t_2 + 1.0) * start + (t_3 - 2.0 * t_2 + t) * previous_tangent + (-2.0 * t_3 + 3.0 * t_2) * end + (t_3 - t_2) * next_tangent;

}

void main() {


    float w = 1.0;

    uint previous_animation_frame = pushConstant.animation_frames >> 24;
    uint next_animation_frame = (pushConstant.animation_frames & 16711680) >> 16; //0b0000_0000_1111_1111_0000_0000_0000_0000



    float animation_progress = unpackUnorm2x16(pushConstant.animation_frames).x; //x because we want the least significant bits

    float previous_tangent = ssbo.bone_sets[previous_animation_frame].output_tangent;
    float next_tangent = ssbo.bone_sets[next_animation_frame].input_tangent;


    mat4 first_frame_bone_transform = mat4(0.0);
    float total_weight = 0.0;
    for(int i = 0; i < 4; i++){
        first_frame_bone_transform += ssbo.bone_sets[previous_animation_frame].bones[bone_indices[i]].matrix * bone_weights[i];
        total_weight += bone_weights[i];
    }
    if (total_weight == 0.0){
        first_frame_bone_transform = mat4(1.0);
    }


    mat4 second_frame_bone_transform = mat4(0.0);
    total_weight = 0.0;
    for(int i = 0; i < 4; i++){
        second_frame_bone_transform += ssbo.bone_sets[next_animation_frame].bones[bone_indices[i]].matrix * bone_weights[i];
        total_weight += bone_weights[i];
    }
    if (total_weight == 0.0){
        second_frame_bone_transform = mat4(1.0);
    }

    float cubic_animation_progress = cubic_spline(0.0, previous_tangent, 1.0, next_tangent, animation_progress);
    mat4 bone_transform = mix(first_frame_bone_transform, second_frame_bone_transform, cubic_animation_progress);
    vec4 out_position = ubos.proj * ubos.view * pushConstant.model * bone_transform * vec4(inPosition, w);
    gl_Position = out_position;

    mat3 transpose_inverse = mat3(transpose(inverse(pushConstant.model * bone_transform)));
    fragNormal = transpose_inverse * inNormal;

    worldPosition = (pushConstant.model * bone_transform * vec4(inPosition, 1.0)).xyz;
    fragPosition = (bone_transform * vec4(inPosition,1.0)).xyz;
    fragTexCoord = inTexCoord;
    fragTangent = inTangent;
    textureType = inTextureType;


}