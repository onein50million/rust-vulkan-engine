#version 450
#extension GL_EXT_multiview : require
layout(push_constant) uniform PushConstants{
    mat4 model_view_projection;
} pushConstant;

layout(location = 0) in vec3 position;
layout(location = 1) in float elevation;

layout(location = 0) out float fragElevation;

mat4 x_rotation(float angle){
	return mat4(
		vec4(1.0,0.0,0.0,0.0),
		vec4(0.0,cos(angle),sin(angle),0.0),
		vec4(0.0,-sin(angle),cos(angle),0.0),
		vec4(0.0,0.0,0.0,1.0)
	);
}

mat4 y_rotation(float angle){
	return mat4(
		vec4(cos(angle),0.0,-sin(angle),0.0),
		vec4(0.0,1.0,0.0,0.0),
		vec4(sin(angle),0.0,cos(angle),0.0),
		vec4(0.0,0.0,0.0,1.0)
	);
}

mat4 z_rotation(float angle){
	return mat4(
		vec4(cos(angle), sin(angle),0.0,0.0),
		vec4(-sin(angle),cos(angle),0.0,0.0),
		vec4(0.0,0.0,1.0,0.0),
		vec4(0.0,0.0,0.0,1.0)
	);
}

// mat4 axis_angle(vec3 axis, float angle){
// 	return mat4(
// 		vec4((1.0 - cos(angle))*axis.x*axis.x+cos(angle),(1.0-cos(angle))*axis.x*axis.y+axis.z*sin(angle), (1.0 - cos(angle))*axis.z*axis.x - axis.y*sin(angle), 0.0),
// 		vec4((1.0 - cos(angle))*axis.x*axis.y-axis.z*sin(angle),(1.0-cos(angle))*axis.y*axis.y+cos(angle), (1.0 - cos(angle))*axis.y*axis.z + axis.x*sin(angle), 0.0),
// 		vec4((1.0 - cos(angle))*axis.z*axis.x+axis.y*sin(angle),(1.0-cos(angle))*axis.y*axis.z-axis.x*sin(angle), (1.0 - cos(angle))*axis.z*axis.z + cos(angle), 0.0),
// 		vec4(0.0,0.0,0.0,1.0)
// 	);
// }

// mat4 translate(vec3 amount){
// 	return mat4(
// 		vec4(1.0,0.0,0.0,0.0),
// 		vec4(0.0,1.0,0.0,0.0),
// 		vec4(0.0,0.0,1.0,0.0),
// 		vec4(amount.x,amount.y,amount.z,1.0)
// 	);
// }

/*

    if (gl_GlobalInvocationID.z == 0)
    { normal = vec3(1.0f, -uv.y, -uv.x); }
    else if (gl_GlobalInvocationID.z == 1)
    { normal = vec3(-1.0f, -uv.y, uv.x); }
    else if (gl_GlobalInvocationID.z == 2)
    { normal = vec3(uv.x, 1.0f, uv.y); }
    else if (gl_GlobalInvocationID.z == 3)
    { normal = vec3(uv.x, -1.0f, -uv.y); }
    else if (gl_GlobalInvocationID.z == 4)
    { normal = vec3(uv.x, -uv.y, 1.0f); }
    else if (gl_GlobalInvocationID.z == 5)
    { normal = vec3(-uv.x, -uv.y, -1.0f); }


*/


// const mat4[] CAMERA_MATRICES = {
//     mat4(1.0),
//     mat4(1.0),
//     mat4(1.0),
//     mat4(1.0),
//     mat4(1.0),
//     mat4(1.0)
// };

const float ROOT2_OVER_2 = 0.707106781;

const float LEFT = -6371000.0 * ROOT2_OVER_2;
const float RIGHT = 6371000.0 * ROOT2_OVER_2;
const float TOP = -6371000.0 * ROOT2_OVER_2;
const float BOTTOM = 6371000.0 * ROOT2_OVER_2;
const float NEAR = 3.0 * 6371000.0;
const float FAR = 10000.0;


// const mat4 PROJECTION = mat4(
//     2.0/ (RIGHT - LEFT),0.0,0.0, 0.0,
//     0.0,2.0 / (TOP - BOTTOM),0.0, 0.0,
//     0.0,0.0,-2.0 / (FAR - NEAR), 0.0,
//     -(RIGHT + LEFT)/(RIGHT - LEFT),-(TOP + BOTTOM) / (TOP - BOTTOM),-(FAR + NEAR)/ (FAR - NEAR), 1.0
// );

const mat4 PROJECTION = mat4(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, -1.0, -1.0,
    0.0,0.0, -2000.0, 0.0
);

const float PI = 3.1416;
const float TAU = 2.0 * PI;

const vec3[] ROTATIONS = {
    vec3(0.0, -PI/2.0, 0.0),
    vec3(0.0, PI/2.0, 0.0),
    vec3(-PI/2.0, 0.0, 0.0),
    vec3(PI/2.0, 0.0, 0.0),
    vec3(0.0, 0.0, 0.0),
    vec3(0.0, PI, 0.0),
};

void main() {
    
    vec3 rotation = ROTATIONS[gl_ViewIndex];
    mat4 view = inverse(x_rotation(rotation.x) * y_rotation(rotation.y) * z_rotation(rotation.z));

    gl_Position = PROJECTION * view * vec4(position,1.0);
    fragElevation = elevation;
    // fragElevation = float(gl_ViewIndex);
}
