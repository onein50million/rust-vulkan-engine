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
// layout(binding = 10) uniform samplerCube planet_textures[NUM_PLANET_TEXTURES];
layout(binding = 11) uniform sampler3D image_3ds[NUM_MODELS]; //animated textures and whatnot


layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec4 fragTangent;
layout(location = 3) in vec2 fragTexCoord;

layout(location = 4) in vec3 worldPosition;
layout(location = 5) flat in uint textureType;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outRoughMetalAo;


vec3 unproject_point(vec3 point){
    mat4 transform = ubos.proj;
    float inverse_denom = transform[2][3] / (point.z + transform[2][2]);
    return vec3(point.x * inverse_denom / transform[0][0],point.y * inverse_denom / transform[1][1],-inverse_denom);
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
    float scale = 1.0;
    scale *= scale_modifier;
    vec3 normal = triplanar_normal(normal_maps[pushConstant.texture_index+texture_offset],fragPosition, normalize(worldPosition),normalize(fragPosition),scale, offset).rgb;
    vec4 albedo = triplanar_sample(texSampler[pushConstant.texture_index+texture_offset],fragPosition, normalize(fragPosition), scale, offset);
    albedo.a = 1.0;
    float roughness = triplanar_sample(rough_metal_ao_maps[pushConstant.texture_index+texture_offset],fragPosition, normalize(fragPosition), scale, offset).r;
    float metalness = triplanar_sample(rough_metal_ao_maps[pushConstant.texture_index+texture_offset],fragPosition, normalize(fragPosition), scale, offset).g;
    float ambient_occlusion = triplanar_sample(rough_metal_ao_maps[pushConstant.texture_index+texture_offset],fragPosition, normalize(fragPosition), scale, offset).b;

    return SampleSet(albedo,roughness,metalness,ambient_occlusion,normal);
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
    outAlbedo = vec4(vec3(0.0),1.0);
    outNormal = vec4(vec3(0.0),1.0);
    outRoughMetalAo = vec4(vec3(0.0),1.0);


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
        vec3 scren_space_direction = vec3(fragTexCoord*2.0 - 1.0,1.0);
        vec3 view_direction = CORRECTION_MATRIX * normalize((inverse(ubos.view) * vec4(unproject_point(scren_space_direction),0.0)).xyz);

        outAlbedo.rgb = texture(cubemaps[ubos.cubemap_index], view_direction).rgb;
        return;
    }
    else if ((pushConstant.bitfield&IS_GLOBE) > 0){ //triplanar mapping
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
    outAlbedo.rgb = albedo.rgb;
    outNormal.xyz = normal;
    outRoughMetalAo.rgb = vec3(roughness, metalness, ambient_occlusion);
}