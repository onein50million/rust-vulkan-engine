#version 450

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform usampler2D font_texture;


void main() {
    outColor = vec4(1.0);
}
