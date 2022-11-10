#version 450
#include "../planet/atmosphere.glsl"
layout (input_attachment_index = 0, binding = 0) uniform subpassInput previousPassColor;
layout (input_attachment_index = 0, binding = 1) uniform subpassInput previousPassDepth;


layout(binding = 2, std140) uniform UniformBufferObject {
    mat4 proj;
} ubos;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

layout(push_constant) uniform PushConstants{
    mat4 view_inverse;
    float time;
} pushConstant;


vec3 unproject_point(vec3 point){
    mat4 transform = ubos.proj;
    float inverse_denom = transform[2][3] / (point.z + transform[2][2]);
    return vec3(point.x * inverse_denom / transform[0][0],point.y * inverse_denom / transform[1][1],-inverse_denom);
}

const float PI = 3.14;
const float E = 2.71828;

float linearize_depth(float d,float zNear,float zFar)
{
    return zNear * zFar / (zFar + d * (zNear - zFar));
}

// const float STANDARD_DEVIATION = 0.3;
// const float GAUSSIAN_RADIUS = 0.05;
// const float GAUSSIAN_SAMPLES = 32.0;
// float gaussian_blur(vec2 sample_position){
//     return (1.0 / (2.0 * PI * STANDARD_DEVIATION * STANDARD_DEVIATION)) * pow(E,-((sample_position.x * sample_position.x + sample_position.y * sample_position.y)/(2.0*PI*PI)));
// }

// const float SUN_ANGULAR_SIZE = 1.0 - 0.99127335;
const float SUN_ANGULAR_SIZE = 0.1;


float hash(float n) { return fract(sin(n) * 1e4); }

void main()
{

    vec3 scene_color = subpassLoad(previousPassColor).rgb;
    float scene_depth = subpassLoad(previousPassDepth).r;

    vec2 screen_ratio = inUV;
    vec3 origin = pushConstant.view_inverse[3].xyz;

    vec3 screen_space_target = vec3(screen_ratio.x * 2.0 - 1.0, screen_ratio.y * 2.0 - 1.0, -1.0);
    vec4 v4_target = pushConstant.view_inverse * vec4(unproject_point(screen_space_target.xyz), 0.0);
    vec3 target = normalize(v4_target.xyz);
    origin.y *= -1.0;
    target.y *= -1.0;

    float year_ratio = pushConstant.time;
    vec3 sun_direction = normalize(vec3(sin(year_ratio*PI*2.0), 0.0, cos(year_ratio*PI*2.0)));
    // the color to use, w is the scene depth
    vec4 skylight_color = vec4(0.0, 0.0, 0.0, 1e12);
    
    // add a sun, if the angle between the ray direction and the light direction is small enough, color the pixels white
    skylight_color.rgb = vec3(dot(target, sun_direction) > 0.9998 ? 3.0 : 0.0);
    // vec2 sun_screen_position = (pushConstant.view_inverse * vec4(unproject_point(sun_direction.xyz), 0.0)).xy * 2.0 - 1.0;
    
    
    // // vec2 sun_screen_position;
    // // {
    // //     vec3 forward = (pushConstant.view_inverse * vec4(unproject_point(vec3(0.0,0.0,1.0)), 1.0)).xyz;
    // //     float denom = dot(forward,sun_direction);
    // //     vec3 plane_origin = origin + forward * 1.0;
    // //     float t = dot(plane_origin-origin,-forward) / denom;
    // //     vec3 hit_position = origin + sun_direction * t;
    // //     sun_screen_position = (ubos.proj * inverse(pushConstant.view_inverse) * vec4(hit_position, 1.0)).xy;
    // // }
    
    // vec2 sun_screen_position = ((ubos.proj * inverse(pushConstant.view_inverse) * vec4(sun_direction * 1e9, 1.0)).xy + 1.0) / 2.0;
    // vec2 sun_offset = inUV - sun_screen_position;
    // float sun_rotation = atan(sun_offset.y,sun_offset.x);
    // float radius = mix(SUN_ANGULAR_SIZE * 0.9, SUN_ANGULAR_SIZE * 1.1, smoothstep(-0.1, 0.1, sin(sun_rotation*30.0)));
    // // radius *= max(hash((sun_rotation + pushConstant.time*0.1)*0.0003),SUN_ANGULAR_SIZE * 0.9);

    // skylight_color.rgb = (1.0 - vec3(smoothstep(radius * 0.9, radius,pow(acos(dot(target, sun_direction)),1.0)))) * vec3(0.99,0.99,0.8);
    // if (distance(sun_screen_position, inUV) < 0.1){
    //     skylight_color.rgb = vec3(1.0, 0.0, 0.0);
    // }
    // skylight_color.rgb = (1.0 - vec3(smoothstep(radius * 0.9, radius,pow(distance(sun_screen_position, inUV),1.0)))) * vec3(0.99,0.99,0.4);
    // skylight_color.rgb = (1.0 - vec3(distance(sun_screen_position, inUV))) * vec3(0.99,0.99,0.4);
    // skylight_color.rgb = vec3(smoothstep(radius * 0.99, radius,pow(dot(target, sun_direction),32.0)));

    // for (float x = -GAUSSIAN_RADIUS; x < GAUSSIAN_RADIUS; x+= (GAUSSIAN_RADIUS * 2.0) / GAUSSIAN_SAMPLES){
    //     for (float y = -GAUSSIAN_RADIUS; y < GAUSSIAN_RADIUS; y+= (GAUSSIAN_RADIUS * 2.0) / GAUSSIAN_SAMPLES){
    //         float aspect_ratio = ubos.proj[1][1] / ubos.proj[0][0];
    //         vec2 sample_position = vec2(x,y);
    //         sample_position.y *= aspect_ratio;
    //         vec2 gaussian_screen_ratio = inUV + sample_position;
    //         vec3 gaussian_screen_space_target = vec3(gaussian_screen_ratio.x * 2.0 - 1.0, gaussian_screen_ratio.y * 2.0 - 1.0, -1.0);
    //         vec4 gaussian_v4_target = pushConstant.view_inverse * vec4(unproject_point(gaussian_screen_space_target.xyz), 0.0);
    //         vec3 gaussian_target = normalize(gaussian_v4_target.xyz);
    //         gaussian_target.y *= -1.0;
    //         skylight_color.rgb += vec3(dot(gaussian_target, sun_direction) > 0.9998 ? 3.0 : 0.0) * gaussian_blur(sample_position);
    //     }
    // }
    // get where the ray intersects the planet
    vec2 planet_intersect = ray_sphere_intersect(origin - PLANET_POS, target, PLANET_RADIUS); 
    
    // if the ray hit the planet, set the max distance to that ray
    if (0.0 < planet_intersect.y) {
    	skylight_color.w = max(planet_intersect.x, 0.0);
        
        // sample position, where the pixel is
        vec3 sample_pos = origin + (target * planet_intersect.x) - PLANET_POS;
        
        // and the surface normal
        vec3 surface_normal = normalize(sample_pos);
        
        // get the color of the sphere
        // skylight_color.xyz = vec3(0.0, 0.25, 0.05); 
        
        // get wether this point is shadowed, + how much light scatters towards the camera according to the lommel-seelinger law
        vec3 N = surface_normal;
        vec3 V = -target;
        vec3 L = sun_direction;
        float dotNV = max(1e-6, dot(N, V));
        float dotNL = max(1e-6, dot(N, L));
        float shadow = dotNL / (dotNL + dotNV);
        
        // apply the shadow
        // skylight_color.xyz *= shadow;
        skylight_color.xyz = scene_color * mix(1.0, 10.0, shadow); 

        // apply skylight
        skylight_color.xyz += clamp(skylight(sample_pos, surface_normal, sun_direction, vec3(0.0)), 0.0, 1.0);
    }
    // col = color.rgb;
    vec3 col = vec3(0.0);

    col += calculate_scattering(
    	origin,				// the position of the camera
        target, 					// the camera vector (ray direction of this pixel)
        linearize_depth(scene_depth,1000000.0, 20000000.0), 						// max dist, essentially the scene depth
        // 1.0/0.0, 						// max dist, essentially the scene depth
        skylight_color.rgb,						// scene color, the color of the current pixel being rendered
        sun_direction,						// light direction
        vec3(40.0),						// light intensity, 40 looks nice
        PLANET_POS,						// position of the planet
        PLANET_RADIUS,                  // radius of the planet in meters
        ATMOS_RADIUS,                   // radius of the atmosphere in meters
        RAY_BETA,						// Rayleigh scattering coefficient
        MIE_BETA,                       // Mie scattering coefficient
        ABSORPTION_BETA,                // Absorbtion coefficient
        AMBIENT_BETA,					// ambient scattering, turned off for now. This causes the air to glow a bit when no light reaches it
        G,                          	// Mie preferred scattering direction
        HEIGHT_RAY,                     // Rayleigh scale height
        HEIGHT_MIE,                     // Mie scale height
        HEIGHT_ABSORPTION,				// the height at which the most absorption happens
        ABSORPTION_FALLOFF,				// how fast the absorption falls off from the absorption height 
        PRIMARY_STEPS, 					// steps in the ray direction 
        LIGHT_STEPS 					// steps in the light direction
    );

    // apply exposure, removing this makes the brighter colors look ugly
    // you can play around with removing this
    col = 1.0 - exp(-col);

    // if (length(col) > 0.0){
    //     outFragColor = vec4(col,1.0);
    // }else{
    //     outFragColor = vec4(scene_color,1.0);
    // }
    // col = mix(scene_color, col, length(col));

    outFragColor = vec4(mix(
        scene_color,
        col,
        length(col)
    ), 1.0);
}