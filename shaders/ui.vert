#version 450
#include "extras.glsl"


layout(location = 0) in vec2 uiPosition;
layout(location = 1) in vec2 uiUV;
layout(location = 2) in vec4 uiColor;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec4 fragColor;

void main() {
    gl_Position = vec4(vec3((uiPosition/ubos.screen_size) * 2.0 - 1.0, 0.0), 1.0);
    fragPosition = gl_Position.xyz;
    fragTexCoord = uiUV;
    fragColor = uiColor;
}
