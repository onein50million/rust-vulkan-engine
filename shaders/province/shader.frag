#version 450
#include "colors.glsl"

layout(location = 0) out vec4 outColor;
layout(location = 0) flat in int VertexIndex;
layout(location = 1) in vec3 worldPosition;

const uint SELECT_FLAG = 1; 
const uint TARGET_FLAG = 2; 

struct VertexData{
    uint flags;
    uint nation_index;
};


const int NUM_CHARS_IN_NAME = 64;
struct NationData{
    mat4 name_matrix;
    uint[NUM_CHARS_IN_NAME/4] name_string; //4 packed u8 in an uint, each representing a character with A starting at 0
    uint name_length;
};

layout(std430, binding = 0) buffer VertexDataArray{
    VertexData data[];
}vertexData;

layout(std430, binding = 1) buffer NationDataArray{
    NationData data[];
}nationData;

layout(binding = 2) uniform sampler2D font_image;

layout(push_constant) uniform PushConstants{
    mat4 model_view_projection;
} pushConstant;

// const float RADIUS = 6378137.0;
// mat4 inverse_ortho(){
//     float left = -RADIUS*0.1;
//     float right = RADIUS*0.1;
//     float bottom = -RADIUS*0.1;
//     float top = RADIUS*0.1;
//     float near = -RADIUS*2.0;
//     float far = RADIUS*2.0;

//     return mat4(
//         (right - left) / 2.0,
//         0,
//         0,
//         0,
//         0,
//         (top-bottom)/2.0,
//         0,
//         0,
//         0,
//         0,
//         (far-near)/-2.0,
//         0,
//         (left+right)/2.0,
//         (top+bottom) / 2.0,
//         -(far+near)/2.0,
//         1.0
//     );
// }


// mat4 look_at(vec3 point){
//     vec3 up = vec3(0.0,-1.0,0.0);
//     vec3 zaxis = normalize(point);
//     vec3 xaxis = cross(up, zaxis);
//     vec3 yaxis = cross(zaxis, xaxis);

//     return mat4(
//         xaxis, 0.0,
//         yaxis, 0.0,
//         zaxis, 0.0,
//         vec3(0.0), 1.0
//     );
// }

// vec2 unproject_point(vec3 point){
//     vec3 nation_point = nationData.data[32].position;
//     // vec3 nation_point = nationData.data[vertexData.data[VertexIndex].nation_index].position;
//     mat4 proj = inverse(inverse_ortho()) * look_at(nation_point);
//     // mat4 proj = pushConstant.model_view_projection;

//     vec4 unprojected_point = proj * vec4(point, 1.0);
//     return unprojected_point.xy / unprojected_point.w;
// }
const float FONT_WIDTH = 64.0;
const float FONT_HEIGHT = 128.0;
const float NUM_ROWS = 1.0;
void main() {
    if ((vertexData.data[VertexIndex].flags & SELECT_FLAG) > 0){
        outColor = vec4(0.0, 0.5, 0.0, 1.0);
    }else if ((vertexData.data[VertexIndex].flags & TARGET_FLAG) > 0){
        outColor = vec4(0.5, 0.0, 0.0, 1.0);
    }else{
        outColor = COLORS[vertexData.data[VertexIndex].nation_index % COLORS.length()];
    }

    NationData nationData = nationData.data[vertexData.data[VertexIndex].nation_index];
    vec4 unprojected_point = nationData.name_matrix * vec4(worldPosition, 1.0);
    vec2 unprojected = unprojected_point.xy / unprojected_point.w;
    if (unprojected.x > -1.0 && unprojected.x < 1.0 && unprojected.y > -1.0 && unprojected.y < 1.0){

        vec2 normalized_position = (unprojected + 1.0)/2.0;

        uint current_letter_index = int(normalized_position.x * nationData.name_length);
        uint current_letter = (nationData.name_string[current_letter_index/4] >> ((current_letter_index%4) * 8)) & 255;
        vec2 atlas_size = textureSize(font_image, 0);

        float half_pixel = 0.5 / atlas_size.x;
        vec2 character_size = vec2(FONT_WIDTH, FONT_HEIGHT) / atlas_size;
        vec2 sample_position = vec2(
            -half_pixel + (current_letter + mod(normalized_position.x * nationData.name_length, 1.0)) * character_size.x,
            normalized_position.y * NUM_ROWS
        );

        vec3 font_color = vec3(0.05);
        if(current_letter != 255){
            outColor = vec4(mix(outColor.rgb, font_color, texture(font_image, sample_position).r), 1.0);
        }
    }



}
