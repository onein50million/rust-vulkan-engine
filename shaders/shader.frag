#version 450

const int NUM_MODELS = 1000;
const int NUM_RANDOM = 100;
const int NUM_LIGHTS = 4;


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

const int IS_CUBEMAP = 1;
const int IS_HIGHLIGHTED = 2;
const int IS_VIEWMODEL = 4;


const float SPECULAR_MULTIPLIER = 5.0;

layout(binding = 1) uniform sampler2D texSampler[NUM_MODELS];
layout(binding = 3) uniform samplerCube cubemaps[NUM_MODELS];
layout(binding = 4) uniform sampler2D normal_maps[NUM_MODELS];
layout(binding = 5) uniform sampler2D roughness_maps[NUM_MODELS];

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec4 fragTangent;
layout(location = 3) in vec2 fragTexCoord;

layout(location = 4) in vec3 worldPosition;

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
        vec3 camera_location = inverse(pushConstant.view)[3].xyz;
        vec3 view_direction = normalize(camera_location - worldPosition);

        vec3 tangent = fragTangent.xyz;
        vec3 bitangent = cross(fragNormal,tangent)*fragTangent.w;
        mat3 TBN = (mat3(
        normalize(vec3(pushConstant.model * vec4(tangent,0.0))),
        normalize(vec3(pushConstant.model * vec4(bitangent,0.0))),
        normalize(vec3(pushConstant.model * vec4(fragNormal,0.0)))
        ));

        vec3 normal = texture(normal_maps[pushConstant.texture_index], fragTexCoord).rgb;
        normal = 2.0*normal - 1.0;
        normal = normalize(TBN * normal);



        vec3 total_light = vec3(0.0);


        float roughness = texture(roughness_maps[pushConstant.texture_index], fragTexCoord).r;
        float shininess = 1.0 - roughness;
        //sun
        vec3 sun_direction = normalize(vec3(1.0,-1.0,1.0));
        vec3 sun_color = vec3(1.0,0.6,0.7) * 0.05;
        vec3 sun_halfway_direction = normalize(sun_direction + view_direction);
        float sun_specular_value = pow(max(dot(normal,sun_halfway_direction),0.0),1.0 + 255.0*shininess);
        total_light += max(dot(normal,sun_direction)*sun_color,0.0);
        total_light += SPECULAR_MULTIPLIER * sun_specular_value * sun_color;

        for (int i = 0; i< NUM_LIGHTS; i++){
            vec3 light_position = ubos.lights[i].position.xyz;
            float light_distance = length(light_position - worldPosition);
            vec3 light_color = ubos.lights[i].color.xyz * (1.0 / (light_distance * light_distance));
            vec3 light_direction = normalize(light_position - worldPosition);


            //diffuse
            total_light += max(dot(normal,light_direction)*light_color,0.0);

            //specular
            vec3 halfway_direction = normalize(light_direction + view_direction);
            float specular_value = pow(max(dot(normal,halfway_direction),0.0),1.0 + 255.0*shininess);
            total_light += SPECULAR_MULTIPLIER * specular_value * sun_color;

        }

        vec3 diffuse = total_light;
        vec3 incident = normalize(worldPosition - camera_location);
        vec3 environment_map = texture(cubemaps[0],reflect(incident, normal)).rgb;
        outColor = vec4((0.1+diffuse)*albedo.rgb + 0.1*environment_map*pow(shininess,2.0),albedo.a);
//        outColor = vec4(vec3(shininess),1.0);
//        outColor = vec4(texture(normal_maps[pushConstant.texture_index], fragTexCoord*100.0).rgb,1.0);
//        outColor = vec4(normal,1.0);
    }
}