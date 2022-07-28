#version 450
#include "colors.glsl"

layout(location = 0) out vec4 outColor;
layout(location = 0) flat in int VertexIndex;

const uint SELECT_FLAG = 1; 
const uint TARGET_FLAG = 2; 

struct VertexData{
    uint flags;
    uint nation_index;
};

layout(std430, binding = 0) buffer VertexDataArray{
    VertexData data[];
}vertexData;

void main() {
    if ((vertexData.data[VertexIndex].flags & SELECT_FLAG) > 0){
        outColor = vec4(0.0, 0.5, 0.0, 1.0);
    }else if ((vertexData.data[VertexIndex].flags & TARGET_FLAG) > 0){
        outColor = vec4(0.5, 0.0, 0.0, 1.0);
    }else{
        outColor = COLORS[vertexData.data[VertexIndex].nation_index % COLORS.length()];
    }
}
