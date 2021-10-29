#version 450

const int NUM_MODELS = 1000;
const int NUM_RANDOM = 100;
const int NUM_LIGHTS = 100;

struct Light{
    vec4 position;
    vec4 color;
};

layout(binding = 0, std140) uniform UniformBufferObject {
    vec4 random[NUM_RANDOM];
    Light lights[NUM_LIGHTS];
    int player_index;
    int num_lights;
    int value3;
    int value4;
    vec2 mouse_ratio;
} ubos;

layout(push_constant) uniform PushConstants{
    mat4 model;
    mat4 view;
    mat4 proj;
    int texture_index;
    float constant;
} pushConstant;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec4 fragTangent;
layout(location = 3) out vec2 fragTexCoord;

layout(location = 4) out vec3 worldPosition;


void main() {
    mat3 transpose_inverse = mat3(transpose(inverse(pushConstant.model)));

    gl_Position = pushConstant.proj * pushConstant.view * pushConstant.model * vec4(inPosition, 1.0);
    fragNormal = transpose_inverse * inNormal;
    fragTexCoord = inTexCoord;
    fragPosition = inPosition;
    worldPosition = (pushConstant.model * vec4(inPosition, 1.0)).xyz;
//    fragTangent = vec4(transpose_inverse * inTangent.xyz, inTangent.w);
    fragTangent = inTangent;

}