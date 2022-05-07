#version 450

layout(location = 0) out vec4 outColor;

layout(location = 0) in vec4 fragColor;


void main() {
    // if (fragColor.a < 0.9){
    //     discard;
    // }
    outColor = fragColor;
    // outColor.a = pow(outColor.a, 32);
}
