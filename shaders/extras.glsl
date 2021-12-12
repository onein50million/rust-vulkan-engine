//https://www.shadertoy.com/view/3d2GDW
//https://github.com/blender/blender/blob/master/source/blender/gpu/shaders/material/gpu_shader_material_tex_noise.glsl
//https://www.shadertoy.com/view/4djSRW
const int NUM_RANDOM = 100;
const int NUM_LIGHTS = 1;
const int NUM_MODELS = 100;
const float PI = 3.14159;

const float PLANET_SIZE = 6371000.0;
const float ATMOSPHERE_ELEVATION = 120000.0;
const float ATMOSPHERE_SIZE = PLANET_SIZE + ATMOSPHERE_ELEVATION;
const float CLOUD_SIZE = PLANET_SIZE + ATMOSPHERE_ELEVATION * 0.5;

struct Light{
    vec4 position;
    vec4 color;
};
layout(binding = 0, std140) uniform UniformBufferObject {
    vec4 random[NUM_RANDOM];
    Light lights[NUM_LIGHTS];
    int player_index;
    int num_lights;
    int map_mode;
    int value4;
    vec4 mouse_position;
    vec4 time;
    mat4 planet_model_matrix;
} ubos;


vec4 random_vec4_offset(float p)
{
    vec4 p4 = fract(vec4(p) * vec4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);

}
vec4 mod289(vec4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0; }

float mod289(float x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0; }

vec4 permute(vec4 x) {
    return mod289(((x*34.0)+1.0)*x);
}

float permute(float x) {
    return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
    return 1.79284291400159 - 0.85373472095314 * r;
}

float taylorInvSqrt(float r)
{
    return 1.79284291400159 - 0.85373472095314 * r;
}

vec4 grad4(float j, vec4 ip)
{
    const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
    vec4 p,s;

    p.xyz = floor( fract (vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
    p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
    s = vec4(lessThan(p, vec4(0.0)));
    p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www;

    return p;
}

    // (sqrt(5) - 1)/4 = F4, used once below
    #define F4 0.309016994374947451

float snoise(vec4 sample_point, float distortion)
{
    const vec4  C = vec4( 0.138196601125011,  // (5 - sqrt(5))/20  G4
    0.276393202250021,  // 2 * G4
    0.414589803375032,  // 3 * G4
    -0.447213595499958); // -1 + 4 * G4

    // First corner
    vec4 i  = floor(sample_point + dot(sample_point, vec4(F4)) );
    vec4 x0 = sample_point -   i + dot(i, C.xxxx);

    // Other corners

    // Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
    vec4 i0;
    vec3 isX = step( x0.yzw, x0.xxx );
    vec3 isYZ = step( x0.zww, x0.yyz );
    //  i0.x = dot( isX, vec3( 1.0 ) );
    i0.x = isX.x + isX.y + isX.z;
    i0.yzw = 1.0 - isX;
    //  i0.y += dot( isYZ.xy, vec2( 1.0 ) );
    i0.y += isYZ.x + isYZ.y;
    i0.zw += 1.0 - isYZ.xy;
    i0.z += isYZ.z;
    i0.w += 1.0 - isYZ.z;

    // i0 now contains the unique values 0,1,2,3 in each channel
    vec4 i3 = clamp( i0, 0.0, 1.0 );
    vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );
    vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );

    //  x0 = x0 - 0.0 + 0.0 * C.xxxx
    //  x1 = x0 - i1  + 1.0 * C.xxxx
    //  x2 = x0 - i2  + 2.0 * C.xxxx
    //  x3 = x0 - i3  + 3.0 * C.xxxx
    //  x4 = x0 - 1.0 + 4.0 * C.xxxx
    vec4 x1 = x0 - i1 + C.xxxx;
    vec4 x2 = x0 - i2 + C.yyyy;
    vec4 x3 = x0 - i3 + C.zzzz;
    vec4 x4 = x0 + C.wwww;

    // Permutations
    i = mod289(i);
    float j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x);
    vec4 j1 = permute( permute( permute( permute (
    i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))
    + i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))
    + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))
    + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));

    // Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
    // 7*7*6 = 294, which is close to the ring size 17*17 = 289.
    vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;

    vec4 p0 = grad4(j0,   ip);
    vec4 p1 = grad4(j1.x, ip);
    vec4 p2 = grad4(j1.y, ip);
    vec4 p3 = grad4(j1.z, ip);
    vec4 p4 = grad4(j1.w, ip);

    // Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    p4 *= taylorInvSqrt(dot(p4,p4));

    // Mix contributions from the five corners
    vec3 m0 = max(0.5 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
    vec2 m1 = max(0.5 - vec2(dot(x3,x3), dot(x4,x4)            ), 0.0);
    vec3  m02 = m0 * m0;
    vec2 m12 = m1 * m1;
    vec3 m04 = m02 * m02;
    vec2 m14 = m12 * m12;
    vec3 pdotx0 = vec3(dot(p0,x0), dot(p1,x1), dot(p2,x2));
    vec2 pdotx1 = vec2(dot(p3,x3), dot(p4,x4));

    return 109.319 * (  dot(m02*m02, vec3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 )))
    + dot(m12*m12, vec2( dot( p3, x3 ), dot( p4, x4 ) ) ) ) ;

}

float get_noise(vec4 sample_point, float distortion){
    vec4 distorted_point = sample_point;
    if (distortion != 0.0) {
        distorted_point += vec4(snoise(distorted_point + random_vec4_offset(0.0), 0.0) * distortion,
        snoise(distorted_point + random_vec4_offset(1.0), 0.0) * distortion,
        snoise(distorted_point + random_vec4_offset(2.0), 0.0) * distortion,
        snoise(distorted_point + random_vec4_offset(3.0), 0.0) * distortion);
    }

    return snoise(distorted_point, 0.0);
}

//TODO: put this into a texture rather than sampling it twice(once in frag shader, once in compute shader)
//TODO: maybe even compute it less frequently
float cloudAt(vec3 point){
    vec4 sample_point = vec4(point/PLANET_SIZE,ubos.time.x*100.0)*1.0;
    float noise = smoothstep(0.1, 0.5,get_noise(sample_point, 10.5));
//    noise += smoothstep(0.4, 0.9,get_noise(sample_point*0.1 + random_vec4_offset(69.0), 10.5));
    return clamp(noise,0.0,1.0);
}