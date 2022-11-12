#version 450

layout(location = 0) out float outColor;

layout(location = 0) in float fragElevation;
layout(location = 1) in vec3 fragPosition;


void main() {
    outColor = fragElevation;
}
