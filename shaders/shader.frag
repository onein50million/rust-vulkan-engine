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

const float PI = 3.14159;


layout(binding = 1) uniform sampler2D texSampler[NUM_MODELS];
layout(binding = 3) uniform samplerCube cubemaps[NUM_MODELS];

layout(binding = 4) uniform sampler2D normal_maps[NUM_MODELS];
layout(binding = 5) uniform sampler2D rough_metal_ao_maps[NUM_MODELS];

layout(binding = 6) uniform samplerCube irradiance_map[NUM_MODELS];
layout(binding = 7) uniform sampler2D brdf_lut;
layout(binding = 8) uniform samplerCube environment_map[NUM_MODELS];
layout(binding = 9) uniform usampler2D cpu_images[NUM_MODELS];


layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec4 fragTangent;
layout(location = 3) in vec2 fragTexCoord;

layout(location = 4) in vec3 worldPosition;
layout(location = 5) flat in uint textureType;

layout(location = 0) out vec4 outColor;

const float EXPOSURE = 1.0/5.0;

vec3 fresnelSchlick(float cos_theta, vec3 normal_incidence){
    return normal_incidence + (1.0 - normal_incidence) * pow(clamp(1.0 - cos_theta, 0.0, 1.0),5.0);
}

float distributionGGX (vec3 normal, vec3 halfway, float roughness){
    float modified_roughness = pow(roughness,4);

    float normal_dot_halfway_squared = pow(max(dot(normal,halfway),0.0),2);

    float denominator = PI *  pow(normal_dot_halfway_squared * (modified_roughness-1.0)+ 1.0,2);

    return modified_roughness / denominator;
}

float geometrySchlickGGX(float normal_dot_view_direction, float roughness){
    float modified_roughness = pow(roughness+1.0,2)/8.0;

    return normal_dot_view_direction/(normal_dot_view_direction*(1.0-modified_roughness) + 1.0);
}

float geometrySmith(vec3 normal, vec3 view_direction, vec3 light_direction, float roughness){

    float normal_dot_view_direction = max(dot(normal,view_direction),0.0);
    float normal_dot_light_direction = max(dot(normal,light_direction),0.0);


    return geometrySchlickGGX(normal_dot_view_direction,roughness) * geometrySchlickGGX(normal_dot_light_direction,roughness);
}

void main() {
    if ((pushConstant.bitfield&IS_CUBEMAP) > 0) {
//        if(gl_FragCoord.x > 200){
//            outColor = vec4(texture(cubemaps[pushConstant.texture_index], fragPosition).rgb,1.0);
//        }else{
//            outColor = vec4(texture(irradiance_map[pushConstant.texture_index], fragPosition).rgb,1.0);
//        }
        outColor = vec4(texture(cubemaps[pushConstant.texture_index], fragPosition).rgb,1.0);

    }else{
        vec4 albedo = texture(texSampler[pushConstant.texture_index], fragTexCoord) * pushConstant.constant;
        if ((pushConstant.bitfield&IS_HIGHLIGHTED) > 0){
            outColor = vec4(0.9,0.1,0.1,1.0);
            return;
        }
        if ((pushConstant.bitfield&IS_VIEWMODEL) > 0){
//            uint cpu_image_value = texture(cpu_images[pushConstant.texture_index], fragTexCoord).r;
//            outColor = vec4(vec3(float(cpu_image_value)/255.0),1.0);
            outColor = albedo;
            return;
        }

        if (textureType == 1){
            albedo = vec4(vec3(texture(cpu_images[0], fragTexCoord).r/255.0),1.0);
        }
        float roughness = texture(rough_metal_ao_maps[pushConstant.texture_index], fragTexCoord).r;
        float metalness = texture(rough_metal_ao_maps[pushConstant.texture_index], fragTexCoord).g;
        float ambient_occlusion = texture(rough_metal_ao_maps[pushConstant.texture_index], fragTexCoord).b;

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
        normal = inverse(mat3(pushConstant.model)) * normal; //not sure why I need this

        vec3 normal_incidence = vec3(0.04);
        normal_incidence = mix(normal_incidence, albedo.rgb, metalness);

        vec3 total_light = vec3(0.0);
        for (int i = 0; i< NUM_LIGHTS; i++){
            vec3 light_position = ubos.lights[i].position.xyz;
            if (i == 0){
                light_position = camera_location;
            }
            float light_distance = length(light_position - worldPosition);
            float attenuation = 1.0/(light_distance * light_distance);
            vec3 radiance = ubos.lights[i].color.rgb * attenuation;
            vec3 light_direction = normalize(light_position - worldPosition);
            vec3 halfway_direction = normalize(view_direction + light_direction);

            vec3 fresnel = fresnelSchlick(max(dot(halfway_direction,view_direction),0.0),normal_incidence);

            float normal_dot_light_direction = max(dot(normal,light_direction),0.0);

            float normal_distribution = distributionGGX(normal, halfway_direction, roughness);
            float geometry = geometrySmith(normal, view_direction, light_direction, roughness);

            vec3 specular =
            (normal_distribution*geometry*fresnel)
            /(4.0*max(dot(normal,view_direction),0.0)
            * normal_dot_light_direction + 0.0001);

            vec3 reflection_ratio = fresnel;
            vec3 refraction_ratio = 1.0 - reflection_ratio;
            refraction_ratio *= 1.0 - metalness;

            total_light  += (refraction_ratio*albedo.rgb / PI + specular) * radiance * normal_dot_light_direction;

        }

        vec3 irradiance = texture(irradiance_map[0], normal).rgb;
        vec3 irradiance_reflection = fresnelSchlick(max(dot(normal,view_direction),0.0), normal_incidence);
        vec3 irradiance_refraction = 1.0 - irradiance_reflection;

        vec3 reflection = reflect(-view_direction, normal);
        vec3 prefilteredColor = textureLod(environment_map[0],reflection,roughness*6).rgb; //TODO set max reflection lod smarterly
        vec2 brdf = texture(brdf_lut,vec2(max(dot(normal,view_direction),0.0),roughness)).rg;
        vec3 specular = prefilteredColor * (irradiance_reflection * brdf.x + brdf.y);

        vec3 diffuse = irradiance * albedo.rgb;
        vec3 ambient = (irradiance_refraction * diffuse + specular) * ambient_occlusion * EXPOSURE;
        vec3 color = ambient + total_light;


        outColor = vec4(color, albedo.a);
//        outColor = vec4(vec3(textureType), 1.0);

    }
}