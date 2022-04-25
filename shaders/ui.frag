
#version 450
#include "extras.glsl"

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform usampler2D font_texture;

// const vec4 FONT_COLOR = vec4(0.0,0.0,0.0,1.0);

void main() {
    // outColor = texture(font_texture,fragTexCoord).r * fragColor;
    // outColor = fragColor;
    // outColor = vec3(texture(font_texture,fragTexCoord).r,1.0);
    // outColor = mix(fragColor, FONT_COLOR, texture(font_texture,fragTexCoord).r);
    // outColor = vec4(vec3(1.0),texture(font_texture,fragTexCoord).r);
    outColor = fragColor * texture(font_texture, fragTexCoord).rrrr/255.0;
    // outColor = fragColor;
    // outColor = vec4(texture(font_texture, fragTexCoord).rrr,1.0);
    // outColor = vec4(fragTexCoord, 0.0, 1.0);
}
