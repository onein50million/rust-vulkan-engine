#version 450
#include "../planet/elevation.glsl"

layout(location = 0) out float outColor;

layout(location = 0) in float fragElevation;
layout(location = 1) in vec3 fragPosition;


void main() {
    float shifted_elevation = shift_elevation(normalize(fragPosition), fragElevation);
    outColor = shifted_elevation;
}
