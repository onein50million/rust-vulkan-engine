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

const int IS_CUBEMAP = 1;
const int IS_HIGHLIGHTED = 2;
const int IS_VIEWMODEL = 4;


layout(binding = 1) uniform sampler2D texSampler[NUM_MODELS];
layout(binding = 3) uniform samplerCube cubemaps[NUM_MODELS];

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragPosition;
layout(location = 3) in vec3 worldPosition;

layout(location = 0) out vec4 outColor;

void main() {
    if ((pushConstant.bitfield&IS_CUBEMAP) > 0) {
        outColor = texture(cubemaps[pushConstant.texture_index], fragPosition) * pushConstant.constant;
    }else{
        if ((pushConstant.bitfield&IS_HIGHLIGHTED) > 0){
            outColor = vec4(0.9,0.1,0.1,1.0);
            return;
        }
        if ((pushConstant.bitfield&IS_VIEWMODEL) > 0){
            outColor = texture(texSampler[pushConstant.texture_index], fragTexCoord) * pushConstant.constant;
            return;
        }
        vec4 albedo = texture(texSampler[pushConstant.texture_index], fragTexCoord) * pushConstant.constant;
        vec3 camera_location = inverse(pushConstant.view)[3].xyw;

        vec3 light_position = vec3(10.0,-1.0,0.0);
        float light_distance = length(light_position - worldPosition);
        vec3 light_color = vec3(100.0) * (1.0 / (light_distance * light_distance));
        vec3 light_direction = normalize(light_position - worldPosition);

        vec3 sun_direction = normalize(vec3(1.0,-1.0,1.0));
        vec3 sun_color = vec3(1.0,0.6,0.7);
        vec3 total_light = vec3(0.0);
        total_light += dot(fragNormal,light_direction)*light_color * 0.0;
        total_light += dot(fragNormal,sun_direction)*sun_color;
        vec3 diffuse = total_light;

        outColor = vec4((0.1+diffuse)*albedo.rgb,albedo.a);

    }
}