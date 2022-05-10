#include "../extras.glsl"
float shift_elevation(vec3 normal, float elevation){
    float shifted_elevation = elevation + (fbm(normal*10.0) * 2.0 - 1.0 ) * 5000.0;
    return shifted_elevation;
}