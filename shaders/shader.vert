#version 450

const int NUM_MODELS = 100;
const int NUM_RANDOM = 100;

layout(binding = 0, std140) uniform UniformBufferObject {
    vec4 random[NUM_RANDOM];
    int player_index;
    int value2;
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
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 fragPosition;
layout(location = 3) out vec3 worldPosition;


void main() {

    gl_Position = pushConstant.proj * pushConstant.view * pushConstant.model * vec4(inPosition, 1.0);
    fragNormal = mat3(transpose(inverse(pushConstant.model))) * inNormal;
    fragTexCoord = inTexCoord;
    fragPosition = inPosition;
    worldPosition = (pushConstant.model * vec4(inPosition, 1.0)).xyz;
}