#version 450

const int NUM_MODELS = 1000;
const int NUM_RANDOM = 100;
const int NUM_LIGHTS = 1;


struct Light{
    vec4 position;
    vec4 color;
};
struct SampleSet{ //set of samples
    vec4 albedo;
    float roughness;
    float metalness;
    float ambient_occlusion;
    vec3 normal;
};


layout(binding = 0, std140) uniform UniformBufferObject {
    vec4 random[NUM_RANDOM];
    Light lights[NUM_LIGHTS];
    int player_index;
    int num_lights;
    int map_mode;
    int value4;
    vec2 mouse_ratio;
    float time;
    float b;
    float c;
    float d;
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
const int IS_GLOBE = 8;

const int DEEP_WATER_OFFSET = 1;
const int SHALLOW_WATER_OFFSET = 2;
const int FOLIAGE_OFFSET = 3;
const int DESERT_OFFSET = 4;
const int MOUNTAIN_OFFSET = 5;
const int SNOW_OFFSET = 6;


const float PI = 3.14159;

struct Ray{
    vec3 origin;
    vec3 direction;
};


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
layout(location = 6) in float fragElevation;
layout(location = 7) in float fragAridity;
layout(location = 8) in float fragPopulation;
layout(location = 9) in float fragWarmTemp;
layout(location = 10) in float fragColdTemp;

layout(location = 0) out vec4 outColor;

const float EXPOSURE = 1.0/0.5;

//this is broken in some way
//float map(float value, float value_min, float value_max, float target_min, float target_max) {
//    float clamped_value = clamp(value, min(target_min,target_max), max(target_max,target_min));
//    return clamp(target_min + (clamped_value - value_min) * (target_max - target_min) / (value_max - value_min), min(target_min,target_max), max(target_max,target_min));
//}

//from https://github.com/blender/blender/blob/594f47ecd2d5367ca936cf6fc6ec8168c2b360d0/source/blender/gpu/shaders/material/gpu_shader_material_map_range.glsl#L7
float map_range_linear(float value,
float fromMin,
float fromMax,
float toMin,
float toMax )
{
    float result = clamp(toMin + ((value - fromMin) / (fromMax - fromMin)) * (toMax - toMin), min(toMin,toMax), max(toMax,toMin));
    return result;
}

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

vec4 triplanar_sample(in sampler2D in_sampler, vec3 position, vec3 normal, float scale, vec3 offset){
    vec4 color_x = texture(in_sampler, (position.zy + offset.zy) * scale);
    vec4 color_y = texture(in_sampler, (position.xz + offset.xz) * scale);
    vec4 color_z = texture(in_sampler, (position.xy + offset.xy) * scale);

    float blend_sharpness = 5.0;
    vec3 blend_weight = pow(abs(normal), vec3(blend_sharpness));
    blend_weight /= dot(blend_weight, vec3(1.0));

    return color_x * blend_weight.x + color_y * blend_weight.y + color_z * blend_weight.z;

}

vec3 triplanar_normal(in sampler2D in_sampler, vec3 position,vec3 world_normal,vec3 object_normal, float scale,vec3 offset){
    vec3 tangent_normal_x = texture(in_sampler, (position.zy + offset.zy) * scale).xyz * 2.0 - 1.0;
    vec3 tangent_normal_y = texture(in_sampler, (position.xz + offset.xz) * scale).xyz * 2.0 - 1.0;
    vec3 tangent_normal_z = texture(in_sampler, (position.xy + offset.xy) * scale).xyz * 2.0 - 1.0;

    float blend_sharpness = 5.0;
//    vec3 blend_weight = pow(abs(world_normal), vec3(blend_sharpness));
    vec3 blend_weight = pow(abs(object_normal), vec3(blend_sharpness));
    blend_weight /= dot(blend_weight, vec3(1.0));

//    normal_x.z *= sign(world_normal).x;
//    normal_y.z *= sign(world_normal).y;
//    normal_z.z *= sign(world_normal).z;

    tangent_normal_x = vec3(tangent_normal_x.xy + world_normal.zy,(tangent_normal_x.z) * world_normal.x);
    tangent_normal_y = vec3(tangent_normal_y.xy + world_normal.xz,(tangent_normal_y.z) * world_normal.y);
    tangent_normal_z = vec3(tangent_normal_z.xy + world_normal.xy,(tangent_normal_z.z) * world_normal.z);

    return normalize(vec3(
    tangent_normal_x.zyx * blend_weight.x + tangent_normal_y.xzy * blend_weight.y +tangent_normal_z.xyz * blend_weight.z
    ));
}

SampleSet load_sample_set(int texture_offset, vec3 offset){
    float scale = 1.0/6000000;
    vec3 normal = triplanar_normal(normal_maps[pushConstant.texture_index+texture_offset],fragPosition, normalize(worldPosition),normalize(fragPosition),scale, offset).rgb;
    //    vec3 normal = normalize(fragNormal);
    vec4 albedo = triplanar_sample(texSampler[pushConstant.texture_index+texture_offset],fragPosition, normalize(fragPosition),scale, offset);
    albedo.a = 1.0;
    float roughness = triplanar_sample(rough_metal_ao_maps[pushConstant.texture_index+texture_offset],fragPosition, normalize(fragPosition),scale, offset).r;
    float metalness = triplanar_sample(rough_metal_ao_maps[pushConstant.texture_index+texture_offset],fragPosition, normalize(fragPosition),scale, offset).g;
    float ambient_occlusion = triplanar_sample(rough_metal_ao_maps[pushConstant.texture_index+texture_offset],fragPosition, normalize(fragPosition),scale, offset).b;

    return SampleSet(albedo,roughness,metalness,ambient_occlusion,normal);
}


void main() {
    if ((pushConstant.bitfield&IS_CUBEMAP) > 0) {
        //        if(gl_FragCoord.x > 200){
        //            outColor = vec4(texture(cubemaps[pushConstant.texture_index], fragPosition).rgb,1.0);
        //        }else{
        //            outColor = vec4(texture(irradiance_map[pushConstant.texture_index], fragPosition).rgb,1.0);
        //        }
        outColor = vec4(texture(cubemaps[pushConstant.texture_index], fragPosition).rgb, 1.0);
    }
    else{
        vec4 albedo;
        float roughness;
        float metalness;
        float ambient_occlusion;
        vec3 normal;

        vec3 tangent;
        vec3 bitangent;
        mat3 TBN;

        if ((pushConstant.bitfield&IS_GLOBE) > 0){ //triplanar mapping
            float current_temperature = mix(fragColdTemp,fragWarmTemp, sin(ubos.time));
            current_temperature -= 30.0 * map_range_linear(fragElevation, 500.0,2000.0,0.0,1.0);
            if(ubos.map_mode == 0){
//                vec3 water_shift = vec3(ubos.time * 0.001, sin(ubos.time)*0.00001, cos(ubos.time*0.0001));
                vec3 water_shift = vec3(0.0);
                SampleSet deep_water_set = load_sample_set(DEEP_WATER_OFFSET,water_shift);
                SampleSet shallow_water_set = load_sample_set(SHALLOW_WATER_OFFSET,vec3(0.0));
                SampleSet foliage_set = load_sample_set(FOLIAGE_OFFSET,vec3(0.0));
                SampleSet desert_set = load_sample_set(DESERT_OFFSET,vec3(0.0));
                SampleSet mountain_set = load_sample_set(MOUNTAIN_OFFSET,vec3(0.0));
                SampleSet snow_set = load_sample_set(SNOW_OFFSET,vec3(0.0));



                float deep_water_weight = 0.0;
                float shallow_water_weight = 0.0;
                float foliage_weight = 0.0;
                float desert_weight = 0.0;
                float mountain_weight = 0.0;
                float snow_weight = 0.0;

                float steepness = clamp(1.0 - dot(fragNormal, normalize(fragPosition)),0.0, 1.0);
                if (fragElevation < 0.0){
                    if (current_temperature > -5.0){
                        deep_water_weight = map_range_linear(fragElevation, 0.0, -100.0, 0.0,1.0);
                        shallow_water_weight = 1.0 - deep_water_weight;
                    }else{
                        snow_weight = 1.0;
                    }

                }else{
                    float grass_ratio = map_range_linear(fragElevation,0.0, 2000.0,1.0, 0.0);
                    foliage_weight = map_range_linear(fragAridity, 0.0, 0.5, 0.0, 1.0) * grass_ratio;
                    desert_weight = 1.0 - foliage_weight;
                    mountain_weight = map_range_linear(fragElevation,1500.0, 2000.0,0.0, 1.0);

                    float coldness = map_range_linear(current_temperature, -5.0, 3.0, 1.0, 0.0); //How far below zero
                    snow_weight = map_range_linear(steepness, 0.0, 0.001, 0.0, 10.0)*fragAridity * coldness;
                }
//                float coldness = map(current_temperature, -1.0, 0.0, 1.0, 0.0); //How far below zero



                SampleSet sample_sets[] = {
                deep_water_set,
                shallow_water_set,
                foliage_set,
                desert_set,
                mountain_set,
                snow_set
                };

                float weights[] = {
                deep_water_weight,
                shallow_water_weight,
                foliage_weight,
                desert_weight,
                mountain_weight,
                snow_weight
                };

                float weight_sum = 0.0;
                for (int i = 0; i < weights.length(); i++){
                    weight_sum += weights[i];
                }

                if(weight_sum <= 0.1){
                    deep_water_weight = 0.0;
                    shallow_water_weight = 0.0;
                    foliage_weight = 0.0;
                    desert_weight = 0.0;
                    snow_weight = 0.0;
                    mountain_weight = 1.0;
                    weight_sum = 1.0;
                }


                float blends[] = {
                deep_water_weight/weight_sum,
                shallow_water_weight/weight_sum,
                foliage_weight/weight_sum,
                desert_weight/weight_sum,
                mountain_weight/weight_sum,
                snow_weight/weight_sum
                };


                albedo = vec4(0.0);
                roughness = 0.0;
                metalness = 0.0;
                ambient_occlusion = 0.0;
                normal = vec3(0.0);

                for (int i = 0; i < sample_sets.length(); i++){
                    albedo += sample_sets[i].albedo * blends[i];
                    roughness += sample_sets[i].roughness * blends[i];
                    metalness += sample_sets[i].metalness * blends[i];
                    ambient_occlusion += sample_sets[i].ambient_occlusion * blends[i];
                    normal += sample_sets[i].normal * blends[i];
                }
//                outColor = albedo;
//                return;
            }
            else if(ubos.map_mode == 1){ //Elevation
                outColor = vec4(vec3(fragElevation/10000.0),1.0);
                return;
            }else if(ubos.map_mode == 2){ //Aridity
                outColor = vec4(vec3(fragAridity/2.5),1.0);
                return;
            }else if(ubos.map_mode == 3){ //Population
                outColor = vec4(vec3(fragPopulation/100000.0),1.0);
                return;
            }else if(ubos.map_mode == 4){ //Temperature
                outColor = vec4(vec3(current_temperature / 60.0),1.0);
                return;
            }

        }
        else{
            albedo = texture(texSampler[pushConstant.texture_index], fragTexCoord) * pushConstant.constant;
            roughness = texture(rough_metal_ao_maps[pushConstant.texture_index], fragTexCoord).r;
            metalness = texture(rough_metal_ao_maps[pushConstant.texture_index], fragTexCoord).g;
            ambient_occlusion = texture(rough_metal_ao_maps[pushConstant.texture_index], fragTexCoord).b;
            normal = texture(normal_maps[pushConstant.texture_index], fragTexCoord).rgb;

            tangent = fragTangent.xyz;
            bitangent = cross(fragNormal,tangent)*fragTangent.w;
            TBN = (mat3(
            normalize(vec3(pushConstant.model * vec4(tangent,0.0))),
            normalize(vec3(pushConstant.model * vec4(bitangent,0.0))),
            normalize(vec3(pushConstant.model * vec4(fragNormal,0.0)))
            ));

            normal = 2.0*normal - 1.0;
            normal = normalize(TBN * normal);
            normal = inverse(mat3(pushConstant.model)) * normal; //not sure why I need this

        }

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


        vec3 camera_location = inverse(pushConstant.view)[3].xyz;
        vec3 view_direction = normalize(camera_location - worldPosition);


        if ((pushConstant.bitfield&IS_GLOBE) > 0){

        }





        vec3 normal_incidence = vec3(0.04);
        normal_incidence = mix(normal_incidence, albedo.rgb, metalness);

        vec3 total_light = vec3(0.0);
        for (int i = 0; i< NUM_LIGHTS + 1; i++){
            vec3 light_position = ubos.lights[i].position.xyz;
            if (i == 0){
                light_position = camera_location;
            }
            float light_distance = length(light_position - worldPosition);
            float attenuation = 1.0/(light_distance * light_distance);

            vec3 radiance = ubos.lights[i].color.rgb * attenuation;

            vec3 light_direction = normalize(light_position - worldPosition);
            if (i == 0){ //sun
                radiance = ubos.lights[i].color.rgb;
                light_direction = normalize(vec3(sin(ubos.time),0.0,cos(ubos.time)));
            }

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
    }
}