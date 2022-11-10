#version 450
// #include "extras.glsl"
#include "map.glsl"
#include "planet/elevation.glsl"

struct SampleSet{ //set of samples
    vec4 albedo;
    float roughness;
    float metalness;
    float ambient_occlusion;
    vec3 normal;
};

const float RADIUS = 6378137.0;

const int NUM_PLANET_TEXTURES = 6;
const int NORMAL_TEXTURE_INDEX = 0;
const int ELEVATION_TEXTURE_INDEX = 1;
const int ARIDITY_TEXTURE_INDEX = 2;
const int COLD_TEXTURE_INDEX = 3;
const int WARM_TEXTURE_INDEX = 4;
const int WATER_TEXTURE_INDEX = 5;



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
layout(binding = 10) uniform samplerCube planet_textures[NUM_PLANET_TEXTURES];
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

// vec3 triplanar_normal(in sampler2D in_sampler, vec3 position,vec3 world_normal,vec3 object_normal, float scale,vec3 offset){
//     vec3 tangent_normal_x = texture(in_sampler, (position.zy + offset.zy) * scale).xyz * 2.0 - 1.0;
//     vec3 tangent_normal_y = texture(in_sampler, (position.xz + offset.xz) * scale).xyz * 2.0 - 1.0;
//     vec3 tangent_normal_z = texture(in_sampler, (position.xy + offset.xy) * scale).xyz * 2.0 - 1.0;

//     float blend_sharpness = 50.0;
//     vec3 blend_weight = pow(abs(object_normal), vec3(blend_sharpness));
//     blend_weight /= dot(blend_weight, vec3(1.0));

//     tangent_normal_x = vec3(tangent_normal_x.xy + world_normal.zy,(tangent_normal_x.z) * world_normal.x);
//     tangent_normal_y = vec3(tangent_normal_y.xy + world_normal.xz,(tangent_normal_y.z) * world_normal.y);
//     tangent_normal_z = vec3(tangent_normal_z.xy + world_normal.xy,(tangent_normal_z.z) * world_normal.z);

//     return normalize(vec3(
//     tangent_normal_x.zyx * blend_weight.x + tangent_normal_y.xzy * blend_weight.y +tangent_normal_z.xyz * blend_weight.z
//     ));
// }

// vec3 triplanar_normal(in sampler2D in_sampler, vec3 position, vec3 worldPos ,vec3 worldNormal, float scale,vec3 offset){
//     // Basic Swizzle
//     // Triplanar uvs
//     vec2 uvX = worldPos.zy; // x facing plane
//     vec2 uvY = worldPos.xz; // y facing plane
//     vec2 uvZ = worldPos.xy; // z facing plane
//     // Tangent space normal maps
//     vec3 tnormalX = texture(in_sampler, uvX).xyz * 2.0 - 1.0;
//     vec3 tnormalY = texture(in_sampler, uvY).xyz * 2.0 - 1.0;
//     vec3 tnormalZ = texture(in_sampler, uvZ).xyz * 2.0 - 1.0;
//     // Get the sign (-1 or 1) of the surface normal
//     vec3 axisSign = sign(worldNormal);
//     // Flip tangent normal z to account for surface normal facing
//     tnormalX.z *= axisSign.x;
//     tnormalY.z *= axisSign.y;
//     tnormalZ.z *= axisSign.z;

//     float blend_sharpness = 50.0;
//     vec3 blend = pow(abs(normalize(position)), vec3(blend_sharpness));
//     blend /= dot(blend, vec3(1.0));


//     // Swizzle tangent normals to match world orientation and triblend
//     return worldNormal = normalize(
//         tnormalX.zyx * blend.x +
//         tnormalY.xzy * blend.y +
//         tnormalZ.xyz * blend.z
//         );
// }

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

SampleSet load_sample_set(int texture_offset, vec3 offset, float scale_modifier){
    float scale = 1.0/RADIUS;
    scale *= scale_modifier;
    // vec3 normal = triplanar_normal(normal_maps[pushConstant.texture_index+texture_offset],fragPosition, fragNormal,fragNormal,scale, offset).rgb;
    // vec3 normal = triplanar_normal(normal_maps[pushConstant.texture_index+texture_offset],fragPosition, worldPosition,normalize(worldPoeition),scale, offset).rgb;
    vec3 normal = triplanar_normal(normal_maps[pushConstant.texture_index+texture_offset],fragPosition, normalize(worldPosition),normalize(fragPosition),scale, offset).rgb;
    // vec4 albedo = triplanar_sample(texSampler[pushConstant.texture_index+texture_offset],fragPosition, normal, scale, offset);
    // albedo.a = 1.0;
    // float roughness = triplanar_sample(rough_metal_ao_maps[pushConstant.texture_index+texture_offset],fragPosition, normal, scale, offset).r;
    // float metalness = triplanar_sample(rough_metal_ao_maps[pushConstant.texture_index+texture_offset],fragPosition, normal, scale, offset).g;
    // float ambient_occlusion = triplanar_sample(rough_metal_ao_maps[pushConstant.texture_index+texture_offset],fragPosition, normal, scale, offset).b;

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
    else if ((pushConstant.bitfield&IS_GLOBE) > 0){ //triplanar mapping

        float elevation = texture(planet_textures[ELEVATION_TEXTURE_INDEX], normalize(fragPosition)).r;
        mat3 transpose_inverse = mat3(transpose(inverse(pushConstant.model)));
        // normal = (inverse(pushConstant.model) * vec4(texture(planet_textures[NORMAL_TEXTURE_INDEX], normalize(fragPosition)).xyz, 0.0)).xyz;
        // normal = texture(planet_textures[NORMAL_TEXTURE_INDEX], normalize(fragPosition)).xyz;
        normal = transpose_inverse * (texture(planet_textures[NORMAL_TEXTURE_INDEX], normalize(fragPosition)).xyz);
        float aridity = texture(planet_textures[ARIDITY_TEXTURE_INDEX], normalize(fragPosition)).r;
        float cold_temp = texture(planet_textures[COLD_TEXTURE_INDEX], normalize(fragPosition)).r;
        float warm_temp = texture(planet_textures[WARM_TEXTURE_INDEX], normalize(fragPosition)).r;
        // float water_ratio = map_range_linear(texture(planet_textures[WATER_TEXTURE_INDEX], normalize(fragPosition)).r, 10000.0, 20000.0, 1.0, 0.0);
        float water_sdf = texture(planet_textures[WATER_TEXTURE_INDEX], normalize(fragPosition)).r;
        // float water_sdf_normalzied = map_range_linear(water_sdf, -2747820.25, 1079892.5, -1.0, 1.0);
        
        vec3 offset = vec3(
            hash(ubos.time),
            hash(ubos.time + 69.0),
            hash(ubos.time + 420.0)
        );
        float current_temperature = mix(cold_temp,warm_temp, (sin(year_ratio*PI* 2.0) + 1.0) / 2.0) + fbm((normalize(fragPosition) + offset * 100.0) * 2.0 - 1.0) * 10.0;
        float coldness = map_range_linear(current_temperature, -10.0, 3.0, 1.0, 0.0); //How far below zero
        float steepness = 1.0 - clamp(dot(normal, normalize(worldPosition)),0.0, 1.0);
        steepness = map_range_linear(steepness, 0.0, 0.001, 0.0, 1.0);

        // outColor = vec4(normal, 1.0);
        // return;

        if(ubos.map_mode <= 1){

            SampleSet deep_water_set = load_sample_set(DEEP_WATER_OFFSET,vec3(0.0),1.0);
            SampleSet shallow_water_set = load_sample_set(SHALLOW_WATER_OFFSET,vec3(0.0),1.0);
            SampleSet foliage_set = load_sample_set(FOLIAGE_OFFSET,vec3(0.0),1.0);
            SampleSet desert_set = load_sample_set(DESERT_OFFSET,vec3(0.0),10.0);
            SampleSet mountain_set = load_sample_set(MOUNTAIN_OFFSET,vec3(0.0),100.0);
            SampleSet snow_set = load_sample_set(SNOW_OFFSET,vec3(0.0),10.0);
            SampleSet data_set = load_sample_set(DATA_OFFSET,vec3(0.0),1.0);

            float deep_water_weight = 0.0;
            float shallow_water_weight = 0.0;
            float foliage_weight = 0.0;
            float desert_weight = 0.0;
            float mountain_weight = 0.0;
            float snow_weight = 0.0;


            // current_temperature -= 30.0 * map_range_linear(elevation, 500.0,2000.0,0.0,1.0);


            if (water_sdf < 3000.0){
                shallow_water_weight = map_range_linear(water_sdf, -15000.0, -20000, 1.0, 0.0);
                deep_water_weight = 1.0 - shallow_water_weight;
            }else if(water_sdf < 6000.0){
                desert_weight = 1.0;
            }
            else{
                float num_indices = 4.0;
                float index = mod(water_sdf / 25000.0, num_indices + 1.0);

                // if (index < 1.0){
                //     foliage_weight = 1.0;
                // }else if (index < 2.0){
                //     desert_weight = 1.0;
                // }else if (index < 3.0){
                //     snow_weight = 1.0;
                // }else if (index < 4.0){
                //     mountain_weight = 1.0;
                // }

                // foliage_weight = smoothstep(0.0, 1.0 / num_indices, index);
                // desert_weight = smoothstep(1.0 / num_indices, 2.0 / num_indices, index);
                // mountain_weight = smoothstep(2.0 / num_indices, 3.0 / num_indices, index);
                // snow_weight = smoothstep(3.0 / num_indices, 4.0 / num_indices, index);

                float grass_ratio = map_range_linear(elevation,0.0, 1500.0,1.0 , 0.2);
                mountain_weight = map_range_linear(elevation,1500.0, 3000.0,0.0, 1.0);
                foliage_weight = clamp((1.0 - steepness*0.01) *  map_range_linear(aridity, 0.0, 0.5, 0.1, 1.0) * grass_ratio - mountain_weight,0.0,1.0);
                desert_weight = clamp(1.0 - foliage_weight - mountain_weight,0.0,1.0);

                snow_weight = (1.0 - steepness)*aridity * coldness * 1.0;


                foliage_weight = 1.0;
            }
            
            // deep_water_weight = (map_range_linear(elevation, 76.0, -100.0, 0.0,1.0)) * water_ratio;
            // deep_water_weight = water_ratio;
            // shallow_water_weight = (1.0 - deep_water_weight);

            // float grass_ratio = map_range_linear(elevation,0.0, 1500.0,1.0 , 0.2);
            // mountain_weight = map_range_linear(elevation,1500.0, 3000.0,0.0, 1.0) * land_ratio;
            // foliage_weight = clamp((1.0 - steepness*0.01) *  map_range_linear(aridity, 0.0, 0.5, 0.1, 1.0) * grass_ratio - mountain_weight,0.0,1.0) * land_ratio;
            // desert_weight = clamp(1.0 - foliage_weight - mountain_weight,0.0,1.0) * land_ratio;

            // snow_weight = (1.0 - steepness)*aridity * coldness * 1.0;

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
            // outColor = vec4(albedo.rgb, 1.0);
            // return;
        }
        else if (ubos.map_mode == 1){
            // outColor = vec4(vec3(water_ratio),1.0);
            // return;
        }
        // else if(ubos.map_mode == 1){ //Paper/globe map
        //     // albedo = vec4(1.0);
        //     SampleSet data_set = load_sample_set(DATA_OFFSET,vec3(0.0),1.0);
        //     // float aridity = clamp(pow((abs(asin(normalize(fragPosition).y)) / PI) + 0.01, 32.0),0.0, 1.0);
        //     // float latitude_factor = clamp(pow((abs(asin(normalize(fragPosition).y)) / PI) + 0.6, 64.0), 0.0, 1.0);
        //     // float temperature_factor = (data_set.albedo.r - latitude_factor)*2.0 - 1.0 + 0.5;

        //     roughness = 0.95;
        //     ambient_occlusion = 1.0;
        //     metalness = 0.0;
        //     vec3 mountain_color = vec3(0.187820772300678, 0.223227957316809, 0.371237680474149);
        //     vec3 dark_stain_color = vec3(0.701101891932973, 0.479320183100827, 0.258182852921596);
        //     vec3 light_stain_color = vec3(0.896269353374266, 0.814846572216101, 0.723055128921969);
        //     vec3 ground_color = vec3(0.982250550333117, 0.921581856277295, 0.83879901174074);
        //     vec3 grass_color = vec3(0.287440837726917,0.491020849847836,0.287440837726917);
        //     vec3 tree_color = vec3(0.114435373826974, 0.165132194501668, 0.119538427988346);
        //     // vec3 water_color = vec3(0.47353149614801, 0.590618840919337, 0.708375779891687);
        //     vec3 water_color = vec3(0.00749903204322618, 0.168269400189691, 0.356400144145944);
        //     vec3 river_color = vec3(0.00749903204322618, 0.168269400189691, 0.456400144145944);
        //     float scale = 100.0/RADIUS;

        //     vec3 waves_color = mix(water_color, vec3(1.0), smoothstep(0.9,0.95,sin(elevation*-0.03 + ubos.time * 50.0)));
        //     water_color = mix(waves_color, water_color, smoothstep(-500.0, -600.0, elevation));
        //     float latitude = asin(normalize(fragPosition).y);
        //     float longitude = atan(normalize(fragPosition).z, normalize(fragPosition).x);

        //     float normal_ratio = atan(normal.z, normal.x) / (2.0 * PI) + 0.5;
        //     float sun_ratio = atan(sun_direction.z, sun_direction.x) / (2.0 * PI) + 0.5;
        //     float light_ratio = sun_ratio - normal_ratio;

        //     vec4 mountain_image = texture(image_3ds[0], vec3(longitude*60.0, latitude*-60.0, light_ratio));
        //     mountain_color = mix(mountain_color, mountain_image.rgb, mountain_image.a);

        //     // vec4 grass_image = texture(image_3ds[1], vec3(longitude*100.0, latitude*-100.0, (smoothstep(0.499, 0.501, fbm(normal*50.0))+0.5) / 2.0 ));
        //     vec4 grass_image = sample_3d_nearest(image_3ds[1], vec2(longitude*100.0, latitude*-100.0), smoothstep(0.499, 0.501, fbm(normalize(fragPosition)*50.0)));
        //     // vec4 grass_image = texture(image_3ds[1], vec3(longitude*100.0, latitude*-100.0, 0.25));
        //     grass_color = mix(grass_color, grass_image.rgb, grass_image.a);

        //     float steepness = clamp(acos(dot(normal, normalize(worldPosition)))*10.0, 0.0, 1.0);
        //     float tree_elevaton_factor = smoothstep(2600.0, 2500.0, elevation);
        //     tree_elevaton_factor = mix(tree_elevaton_factor, 0.0, smoothstep(500.0,490.0, elevation));
        //     float tree_factor = mix(0.0, 1.0 - steepness,tree_elevaton_factor);

        //     // vec4 tree_image = texture(image_3ds[2], vec3(longitude*100.0, latitude*-100.0, 0.5));
        //     vec4 tree_image = sample_3d(image_3ds[2], vec2(longitude*100.0, latitude*-100.0), smoothstep(0.0, 1.0, smoothstep(0.5,1.0,tree_factor)));
        //     tree_color = mix(tree_color, tree_image.rgb, tree_image.a);
            
        //     vec4 ground_image = sample_3d(image_3ds[3], vec2(longitude*100.0, latitude*-100.0), (0.9*sin(ubos.time*100.0 + fbm(ubos.time*2.0)*500.0*(fbm(normalize(fragPosition)* 1.0) * 2.0 - 1.0)) + 1.0)/2.0);
        //     ground_color = mix(ground_color, ground_image.rgb, ground_image.a);


        //     vec3 stain_color = mix(dark_stain_color, light_stain_color, fbm(normalize(fragPosition)*10.0));

        //     vec3 map_color = mix(mountain_color, ground_color, smoothstep(3000.0, 2990.0, elevation));
        //     map_color = mix(map_color, grass_color, smoothstep(1000.0, 990.0, elevation));
        //     map_color = mix(map_color, tree_color, smoothstep(0.499, 0.501, tree_factor));
            
        //     float river_distort_modifier = 10.0;
        //     float river_width = (0.0001 + 0.01 * fbm(vec2(latitude,longitude) * 100.0)) * (smoothstep(3000.0, 1000.0, elevation)) + smoothstep(2600.0, 3000.0, elevation) * 0.1 * smoothstep(3000.0, 2600.0, elevation);
        //     vec3 distorted_river_coords = vec3(sfbm(normalize(fragPosition) * river_distort_modifier + 128.0), sfbm(normalize(fragPosition) * river_distort_modifier+ 841.0), sfbm(normalize(fragPosition) * river_distort_modifier + 699.0));
        //     vec3 distorted_river_wavy_coords = vec3(sfbm(normalize(fragPosition)*river_distort_modifier + 128.0*0.99999), sfbm(normalize(fragPosition)*river_distort_modifier + 841.0*0.99999), sfbm(normalize(fragPosition)*river_distort_modifier + 699.0*0.99999));
        //     vec3 river_offset = mix(
        //         normalize(fragPosition) * 10.00,
        //         normalize(fragPosition) * 10.00 + mix(
        //             1.0 * normalize(distorted_river_coords),
        //             1.0 * normalize(distorted_river_wavy_coords),
        //             0.5 + sin(ubos.time*100.0)*0.01),
        //         0.05
        //     );
        //     float river_factor = smoothstep(river_width, river_width - 0.001, voronoi3d(river_offset)).r;

        //     map_color = mix(map_color, river_color, river_factor);
        //     map_color = mix(map_color, water_color, smoothstep(0.0, -10.0, elevation));

            
        //     vec3 final_color = mix(stain_color, map_color, clamp(fbm(normalize(fragPosition)*10.0 + 100.0) + 0.8, 0.0, 1.0));
        //     albedo = vec4(final_color,1.0);


        // }
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