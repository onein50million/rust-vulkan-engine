#include "../extras.glsl"
float shift_elevation(vec3 normal, float elevation){
    // float shifted_elevation = elevation + (fbm(normal*10.0) * 2.0 - 1.0 ) * 5000.0;
    float shifted_elevation = elevation + (sfbm(normal*10.0)) * 3000.0;
    // float shifted_elevation = elevation + snoise(normal*50.0) * 1000.0 + snoise(normal*100.0)*100.0;
    // float shifted_elevation = elevation;
    return shifted_elevation;
}