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
layout(binding = 11) uniform sampler3D image_3ds[NUM_MODELS]; //animated textures and whatnot


layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec4 fragTangent;
layout(location = 3) in vec2 fragTexCoord;

layout(location = 4) in vec3 worldPosition;
layout(location = 5) flat in uint textureType;

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

vec2 triplanar_uv(vec3 position, vec3 normal, float scale, vec3 offset){
    vec2 color_x = vec2((position.zy + offset.zy) * scale);
    vec2 color_y = vec2((position.xz + offset.xz) * scale);
    vec2 color_z = vec2((position.xy + offset.xy) * scale);

    float blend_sharpness = 50.0;
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

vec4 sample_3d(sampler3D in_sampler, vec2 uv, float w){
    vec3 texture_size = textureSize(in_sampler, 0);
    return texture(in_sampler, vec3(uv, ((w * (texture_size.z-1))+0.5) / (texture_size.z) ));
}

vec4 sample_3d_nearest(sampler3D in_sampler, vec2 uv, float w){
    vec3 texture_size = textureSize(in_sampler, 0);
    return texture(in_sampler, vec3(uv, (round(w * (texture_size.z - 1))+0.5) / (texture_size.z) ));
}

vec3 rgb(float r, float g, float b){
    return vec3(r/255.0,g/255.0,b/255.0);
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
    float year_ratio = ubos.time.x;
    vec3 sun_direction = normalize(vec3(sin(year_ratio*PI*2.0), 0.0, cos(year_ratio*PI*2.0)));

    if ((pushConstant.bitfield&IS_CUBEMAP) > 0) {
        // vec3 view_direction = (inverse(ubos.view) * vec4(gl_FragCoord.xy,1.0,0.0)).xyz;
        vec3 scren_space_direction = vec3(fragTexCoord*2.0 - 1.0,1.0);
        vec3 view_direction = CORRECTION_MATRIX * normalize((inverse(ubos.view) * vec4(unproject_point(scren_space_direction),0.0)).xyz);

        outColor = vec4(texture(cubemaps[ubos.cubemap_index], view_direction).rgb, 1.0);
        // outColor = vec4(view_direction, 1.0);
        return;
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
        if (ubos.lights[i].color.a < 0.5){
            continue;
        } 
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

    vec3 irradiance = texture(irradiance_map[ubos.cubemap_index], normal).rgb;
    // irradiance = vec3(1.0) - exp(-irradiance * ubos.exposure);
    irradiance = vec3(1.0) - exp(-irradiance * 1.0);

    vec3 irradiance_reflection = fresnelSchlick(max(dot(normal,view_direction),0.0), normal_incidence);
    vec3 irradiance_refraction = 1.0 - irradiance_reflection;

    vec3 reflection = reflect(-view_direction, normal);
    vec3 prefilteredColor = textureLod(environment_map[ubos.cubemap_index],reflection,int(roughness*9.0)).rgb; //TODO set max reflection lod smarterly
    // prefilteredColor = vec3(1.0) - exp(-prefilteredColor * ubos.exposure);
    prefilteredColor = vec3(1.0) - exp(-prefilteredColor * 1.0);

    vec2 brdf = texture(brdf_lut,vec2(max(dot(normal,view_direction),0.0),roughness)).xy;
    vec3 specular = prefilteredColor * (irradiance_reflection * brdf.x + brdf.y);
    vec3 diffuse = irradiance * albedo.rgb;
    diffuse *= 1.0 - shadow_amount;
    specular *= 1.0 - shadow_amount;
    vec3 ambient = (irradiance_refraction * diffuse + specular) * ambient_occlusion;
    vec3 color = ambient + total_light;

    // outColor = vec4(normal, 1.0);
    outColor = vec4(color, albedo.a);
    // outColor = albedo;
    // outColor = vec4(vec3(fbm(normalize(fragPosition))), 1.0);

}