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
    int bitfield; //32 bits, LSB is cubemap flag
} pushConstant;


layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in uint inTextureType;
layout(location = 5) in float inElevation;
layout(location = 6) in float inAridity;
layout(location = 7) in float inPopulation;
layout(location = 8) in float inWarmTemp;
layout(location = 9) in float inColdTemp;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec4 fragTangent;
layout(location = 3) out vec2 fragTexCoord;

layout(location = 4) out vec3 worldPosition;
layout(location = 5) out uint textureType;
layout(location = 6) out float fragElevation;
layout(location = 7) out float fragAridity;
layout(location = 8) out float fragPopulation;
layout(location = 9) out float fragWarmTemp;
layout(location = 10) out float fragColdTemp;


void main() {

    mat3 transpose_inverse = mat3(transpose(inverse(pushConstant.model)));
    vec4 out_position = pushConstant.proj * pushConstant.view * pushConstant.model * vec4(inPosition, 1.0);
    gl_Position = out_position;
    fragNormal = transpose_inverse * inNormal;

    worldPosition = (pushConstant.model * vec4(inPosition, 1.0)).xyz;
    fragPosition = inPosition;
    fragTexCoord = inTexCoord;
    fragTangent = inTangent;
    textureType = inTextureType;
    fragElevation = inElevation;
    fragAridity = inAridity;
    fragPopulation = inPopulation;
    fragColdTemp = inColdTemp;
    fragWarmTemp = inWarmTemp;


}