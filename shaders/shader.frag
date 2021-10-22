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
    int bitfield; //32 bits, LSB is cubemap flag
} pushConstant;

const int CUBEMAP_FLAG = 1;


layout(binding = 1) uniform sampler2D texSampler[NUM_MODELS];
layout(binding = 3) uniform samplerCube cubemaps[NUM_MODELS];

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragPosition;
layout(location = 3) in vec3 worldPosition;

layout(location = 0) out vec4 outColor;

void main() {
    if ((pushConstant.bitfield&CUBEMAP_FLAG) > 0) {
        outColor = texture(cubemaps[pushConstant.texture_index], fragPosition) * pushConstant.constant;
    }else{

        vec4 albedo = texture(texSampler[pushConstant.texture_index], fragTexCoord) * pushConstant.constant;
        vec3 camera_location = inverse(pushConstant.view)[3].xyw;

        vec3 light_position = vec3(10.0,-1.0,0.0);
        float light_distance = length(light_position - worldPosition);
        vec3 light_color = vec3(100.0) * (1.0 / (light_distance * light_distance));
        vec3 light_direction = normalize(light_position - worldPosition);

        vec3 diffuse = max(dot(fragNormal,light_direction),0.0)*light_color;

        outColor = vec4((0.1+diffuse)*albedo.rgb,albedo.a);

    }
}