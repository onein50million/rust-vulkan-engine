#version 450
#include "extras.glsl"
#include "map.glsl"

struct SampleSet{ //set of samples
    vec4 albedo;
    float roughness;
    float metalness;
    float ambient_occlusion;
    vec3 normal;
};

const float RADIUS = 6378137.0;

const int DEEP_WATER_OFFSET = 1;
const int SHALLOW_WATER_OFFSET = 2;
const int FOLIAGE_OFFSET = 3;
const int DESERT_OFFSET = 4;
const int MOUNTAIN_OFFSET = 5;
const int SNOW_OFFSET = 6;
const int DATA_OFFSET = 7;

const mat3 CORRECTION_MATRIX = mat3(1.0000000,  0.0000000,  0.0000000,
0.0000000,  0.0000000, 1.0000000,
0.0000000,  -1.0000000,  0.0000000);


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

layout(location = 0) out vec4 outColor;

const float EXPOSURE = 0.5;


vec3 unproject_point(vec3 point){
    mat4 transform = ubos.proj;
    float inverse_denom = transform[2][3] / (point.z + transform[2][2]);
    return vec3(point.x * inverse_denom / transform[0][0],point.y * inverse_denom / transform[1][1],-inverse_denom);
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

    float blend_sharpness = 50.0;
    vec3 blend_weight = pow(abs(normal), vec3(blend_sharpness));
    blend_weight /= dot(blend_weight, vec3(1.0));

    return color_x * blend_weight.x + color_y * blend_weight.y + color_z * blend_weight.z;

}

vec3 triplanar_normal(in sampler2D in_sampler, vec3 position,vec3 world_normal,vec3 object_normal, float scale,vec3 offset){
    vec3 tangent_normal_x = texture(in_sampler, (position.zy + offset.zy) * scale).xyz * 2.0 - 1.0;
    vec3 tangent_normal_y = texture(in_sampler, (position.xz + offset.xz) * scale).xyz * 2.0 - 1.0;
    vec3 tangent_normal_z = texture(in_sampler, (position.xy + offset.xy) * scale).xyz * 2.0 - 1.0;

    float blend_sharpness = 50.0;
    vec3 blend_weight = pow(abs(object_normal), vec3(blend_sharpness));
    blend_weight /= dot(blend_weight, vec3(1.0));

    tangent_normal_x = vec3(tangent_normal_x.xy + world_normal.zy,(tangent_normal_x.z) * world_normal.x);
    tangent_normal_y = vec3(tangent_normal_y.xy + world_normal.xz,(tangent_normal_y.z) * world_normal.y);
    tangent_normal_z = vec3(tangent_normal_z.xy + world_normal.xy,(tangent_normal_z.z) * world_normal.z);

    return normalize(vec3(
    tangent_normal_x.zyx * blend_weight.x + tangent_normal_y.xzy * blend_weight.y +tangent_normal_z.xyz * blend_weight.z
    ));
}

SampleSet load_sample_set(int texture_offset, vec3 offset, float scale_modifier){
    float scale = 1.0/RADIUS;
    scale *= scale_modifier;
    vec3 normal = triplanar_normal(normal_maps[pushConstant.texture_index+texture_offset],worldPosition, fragNormal,fragNormal,scale, offset).rgb;
    vec4 albedo = triplanar_sample(texSampler[pushConstant.texture_index+texture_offset],worldPosition, normal, scale, offset);
    albedo.a = 1.0;
    float roughness = triplanar_sample(rough_metal_ao_maps[pushConstant.texture_index+texture_offset],worldPosition, normal, scale, offset).r;
    float metalness = triplanar_sample(rough_metal_ao_maps[pushConstant.texture_index+texture_offset],worldPosition, normal, scale, offset).g;
    float ambient_occlusion = triplanar_sample(rough_metal_ao_maps[pushConstant.texture_index+texture_offset],worldPosition, normal, scale, offset).b;

    return SampleSet(albedo,roughness,metalness,ambient_occlusion,normal);
}


void main() {


    vec4 albedo;
    float roughness;
    float metalness;
    float ambient_occlusion;
    vec3 normal;

    vec3 tangent;
    vec3 bitangent;
    mat3 TBN;

    float shadow_amount = 0.0;


    vec3 player_position = vec3(
    ubos.player_position_x,
    ubos.player_position_y,
    ubos.player_position_z
    );

    if ((pushConstant.bitfield&IS_CUBEMAP) > 0) {
        // vec3 view_direction = (inverse(ubos.view) * vec4(gl_FragCoord.xy,1.0,0.0)).xyz;
        vec3 scren_space_direction = vec3(fragTexCoord*2.0 - 1.0,1.0);
        vec3 view_direction = CORRECTION_MATRIX * normalize((inverse(ubos.view) * vec4(unproject_point(scren_space_direction),0.0)).xyz);

        outColor = vec4(texture(cubemaps[pushConstant.texture_index], view_direction).rgb, 1.0);
        // outColor = vec4(view_direction, 1.0);
        return;
    }
    else if ((pushConstant.bitfield&IS_GLOBE) > 0){ //triplanar mapping

        SampleSet deep_water_set = load_sample_set(DEEP_WATER_OFFSET,vec3(0.0),1.0);
        SampleSet shallow_water_set = load_sample_set(SHALLOW_WATER_OFFSET,vec3(0.0),1.0);
        SampleSet foliage_set = load_sample_set(FOLIAGE_OFFSET,vec3(0.0),1.0);
        SampleSet desert_set = load_sample_set(DESERT_OFFSET,vec3(0.0),1.0);
        SampleSet mountain_set = load_sample_set(MOUNTAIN_OFFSET,vec3(0.0),10.0);
        SampleSet snow_set = load_sample_set(SNOW_OFFSET,vec3(0.0),1.0);
        SampleSet data_set = load_sample_set(DATA_OFFSET,vec3(0.0),1.0);

        float deep_water_weight = 0.0;
        float shallow_water_weight = 0.0;
        float foliage_weight = 0.0;
        float desert_weight = 0.0;
        float mountain_weight = 0.0;
        float snow_weight = 0.0;


        float latitude_factor = clamp(pow((abs(asin(normalize(fragPosition).y)) / PI) + 0.6, 64.0), 0.0, 1.0);
        float aridity = clamp(pow((1.0 - abs(asin(normalize(fragPosition).y)) / PI) + 0.01, 32.0),0.0, 1.0);
        float temperature_factor = (data_set.albedo.r - latitude_factor)*2.0 - 1.0 + 0.5;
        // float shifted_elevation = fragElevation + (fbm(normalize(fragPosition)*10.0) * 2.0 - 1.0 )* 5000.0;
        float shifted_elevation = fragElevation;
        float snow_level = 5000.0 + temperature_factor * 3000.0;
        // outColor = vec4(vec3(shifted_elevation),1.0);
        // return;


        if (shifted_elevation < 0.0){
            deep_water_weight = map_range_linear(shifted_elevation, 0.0, -3000.0, 0.0,1.0);
            if (snow_level < -0.8){
                snow_weight = 1.0 - deep_water_weight;
            }
            else{
                shallow_water_weight = 1.0 - deep_water_weight;
            }
        }else{
        float grass_ratio = map_range_linear(shifted_elevation,0.0, 2500.0,1.0 , 0.2);
        mountain_weight = map_range_linear(shifted_elevation,2500.0, 3000.0,0.0, 1.0);

        snow_weight = map_range_linear(shifted_elevation,snow_level,snow_level + 2000.0,0.0, 100.0);
        desert_weight = clamp(temperature_factor + aridity,0.0,1.0);
        foliage_weight = clamp(grass_ratio - mountain_weight - desert_weight,0.0,1.0);

        }

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
        clamp(deep_water_weight/weight_sum,0.0,1.0),
        clamp(shallow_water_weight/weight_sum,0.0,1.0),
        clamp(foliage_weight/weight_sum,0.0,1.0),
        clamp(desert_weight/weight_sum,0.0,1.0),
        clamp(mountain_weight/weight_sum,0.0,1.0),
        clamp(snow_weight/weight_sum,0.0,1.0)
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

        vec3 modified_player_position = player_position;
        modified_player_position.y = (player_position.y - worldPosition.y)*0.05 + worldPosition.y;
        shadow_amount = clamp(mix(1.0,0.0,pow(distance(modified_player_position, worldPosition)*2.0 + 0.1,5.0)),0.0,1.0);
    }
    else {

        albedo = texture(texSampler[pushConstant.texture_index], fragTexCoord);
        roughness = texture(rough_metal_ao_maps[pushConstant.texture_index], fragTexCoord).r;
        metalness = texture(rough_metal_ao_maps[pushConstant.texture_index], fragTexCoord).g;
        ambient_occlusion = texture(rough_metal_ao_maps[pushConstant.texture_index], fragTexCoord).b;
        normal = texture(normal_maps[pushConstant.texture_index], fragTexCoord).rgb;


        tangent = fragTangent.xyz;
        bitangent = cross(fragNormal, tangent)*fragTangent.w;
        TBN = (mat3(
        normalize(vec3(pushConstant.model * vec4(tangent, 0.0))),
        normalize(vec3(pushConstant.model * vec4(bitangent, 0.0))),
        normalize(vec3(pushConstant.model * vec4(fragNormal, 0.0)))
        ));

        normal = 2.0*normal - 1.0;
        normal = normalize(TBN * normal);
        normal = inverse(mat3(pushConstant.model)) * normal;//not sure why I need this
    }
    vec3 sun_direction = normalize(vec3(sin(ubos.time.x), 0.0, cos(ubos.time.x)));

    if ((pushConstant.bitfield&IS_HIGHLIGHTED) > 0){
        outColor = vec4(0.9,0.1,0.1,1.0);
        return;
    }
    if ((pushConstant.bitfield&IS_VIEWMODEL) > 0){
        outColor = albedo;
        return;
    }

    if (textureType == 1){
        albedo = vec4(vec3(texture(cpu_images[0], fragTexCoord).r/255.0),1.0);
    }


    vec3 camera_location = inverse(ubos.view)[3].xyz;
    vec3 view_direction = normalize(camera_location - worldPosition);

    vec3 normal_incidence = vec3(0.04);
    normal_incidence = mix(normal_incidence, albedo.rgb, metalness);

    vec3 total_light = vec3(0.0);
    for (int i = 0; i< NUM_LIGHTS; i++){
        vec3 light_position = ubos.lights[i].position.xyz;
        // if (i == 0){
        //     light_position = camera_location;
        // }
        float light_distance = length(light_position - worldPosition);
        float attenuation = 1.0/(light_distance * light_distance);

        vec3 radiance = ubos.lights[i].color.rgb * attenuation;

        vec3 light_direction = normalize(light_position - worldPosition);
        if (i == 0){ //sun
            radiance = ubos.lights[i].color.rgb;
            light_direction = sun_direction;
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
    irradiance = vec3(1.0) - exp(-irradiance * ubos.exposure);

    vec3 irradiance_reflection = fresnelSchlick(max(dot(normal,view_direction),0.0), normal_incidence);
    vec3 irradiance_refraction = 1.0 - irradiance_reflection;

    vec3 reflection = reflect(-view_direction, normal);
    vec3 prefilteredColor = textureLod(environment_map[0],reflection,int(roughness*9.0)).rgb; //TODO set max reflection lod smarterly
    prefilteredColor = vec3(1.0) - exp(-prefilteredColor * ubos.exposure);

    vec2 brdf = texture(brdf_lut,vec2(max(dot(normal,view_direction),0.0),roughness)).xy;
    vec3 specular = prefilteredColor * (irradiance_reflection * brdf.x + brdf.y);
    vec3 diffuse = irradiance * albedo.rgb;
    diffuse *= 1.0 - shadow_amount;
    specular *= 1.0 - shadow_amount;
    vec3 ambient = (irradiance_refraction * diffuse + specular) * ambient_occlusion;
    vec3 color = ambient + total_light;

    outColor = vec4(color, albedo.a);
    // outColor = vec4(vec3(fbm(normalize(fragPosition))), 1.0);

}