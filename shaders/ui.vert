#version 450

const int NUM_RANDOM = 100;
const int NUM_LIGHTS = 1;
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
    vec2 mouse_position;
    vec2 screen_size;
    vec4 time;
    mat4 planet_model_matrix;
} ubos;


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
