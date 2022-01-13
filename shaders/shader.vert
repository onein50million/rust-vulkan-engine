#version 450
#include "extras.glsl"


layout(push_constant) uniform PushConstants{
    mat4 model;
    mat4 view;
    mat4 proj;
    int texture_index;
    float constant;
    int bitfield; //32 bits, LSB is cubemap flag
} pushConstant;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in uint inTextureType;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec4 fragTangent;
layout(location = 3) out vec2 fragTexCoord;

layout(location = 4) out vec3 worldPosition;
layout(location = 5) out uint textureType;



void main() {


    float w = 1.0;

    mat3 transpose_inverse = mat3(transpose(inverse(pushConstant.model)));
    vec4 out_position = pushConstant.proj * pushConstant.view * pushConstant.model * vec4(inPosition, w);
    gl_Position = out_position;
    fragNormal = transpose_inverse * inNormal;

    worldPosition = (pushConstant.model * vec4(inPosition, 1.0)).xyz;
    fragPosition = inPosition;
    fragTexCoord = inTexCoord;
    fragTangent = inTangent;
    textureType = inTextureType;


}