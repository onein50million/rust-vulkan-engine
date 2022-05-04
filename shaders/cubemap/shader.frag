#version 450

layout(location = 0) out float outColor;

layout(location = 0) in float fragElevation;


void main() {
    outColor = fragElevation;
}
