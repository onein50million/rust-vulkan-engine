#version 450
#include "extras.glsl"

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D font_texture;


void main() {
    outColor = texture(font_texture,fragTexCoord).r * fragColor;
}
