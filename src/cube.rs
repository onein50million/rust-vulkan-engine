use crate::support::Vertex;
use cgmath::{Vector2, Vector3};

/*

   0   3

   1   2

4   7

5   6

*/

pub(crate) const POSITIVE_X_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3 {
            x: 1.0,
            y: -1.0,
            z: -1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 0.0, y: 1.0 },
    },
    Vertex {
        position: Vector3 {
            x: 1.0,
            y: 1.0,
            z: -1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 1.0/6.0, y: 1.0 },
    },
    Vertex {
        position: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 {x: 1.0/6.0, y: 0.0},
    },
    Vertex {
        position: Vector3 {
            x: 1.0,
            y: -1.0,
            z: 1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 0.0, y: 0.0 },
    },
];


pub(crate) const NEGATIVE_X_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3 {
            x: -1.0,
            y: -1.0,
            z: 1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 2.0/6.0, y: 0.0 },
    },
    Vertex {
        position: Vector3 {
            x: -1.0,
            y: 1.0,
            z: 1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 1.0/6.0, y: 0.0 },
    },
    Vertex {
        position: Vector3 {
            x: -1.0,
            y: 1.0,
            z: -1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 1.0/6.0, y: 1.0 },
    },
    Vertex {
        position: Vector3 {
            x: -1.0,
            y: -1.0,
            z: -1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 {
            x: 2.0 / 6.0,
            y: 1.0,
        },
    },
];



pub(crate) const POSITIVE_Y_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 3.0/6.0, y: 1.0 },
    },
    Vertex {
        position: Vector3 {
            x: 1.0,
            y: 1.0,
            z: -1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 3.0/6.0, y: 0.0 },
    },
    Vertex {
        position: Vector3 {
            x: -1.0,
            y: 1.0,
            z: -1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 2.0/6.0, y: 0.0 },
    },
    Vertex {
        position: Vector3 {
            x: -1.0,
            y: 1.0,
            z: 1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 2.0/6.0, y: 1.0 },
    },
];

pub(crate) const NEGATIVE_Y_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3 {
            x: -1.0,
            y: -1.0,
            z: 1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 3.0/6.0, y: 0.0 },
    },
    Vertex {
        position: Vector3 {
            x: -1.0,
            y: -1.0,
            z: -1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 3.0/6.0, y: 1.0 },
    },
    Vertex {
        position: Vector3 {
            x: 1.0,
            y: -1.0,
            z: -1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 4.0/6.0, y: 1.0 },
    },
    Vertex {
        position: Vector3 {
            x: 1.0,
            y: -1.0,
            z: 1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 4.0/6.0, y: 0.0 },
    },
];



pub(crate) const POSITIVE_Z_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3 {
            x: 1.0,
            y: -1.0,
            z: 1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 5.0/6.0, y: 1.0 },
    },
    Vertex {
        position: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 5.0/6.0, y: 0.0 },
    },
    Vertex {
        position: Vector3 {
            x: -1.0,
            y: 1.0,
            z: 1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 4.0/6.0, y: 0.0 },
    },
    Vertex {
        position: Vector3 {
            x: -1.0,
            y: -1.0,
            z: 1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 4.0/6.0, y: 1.0 },
    },
];

pub(crate) const NEGATIVE_Z_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3 {
            x: -1.0,
            y: -1.0,
            z: -1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 5.0/6.0, y: 0.0 },
    },
    Vertex {
        position: Vector3 {
            x: -1.0,
            y: 1.0,
            z: -1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 5.0/6.0, y: 1.0 },
    },
    Vertex {
        position: Vector3 {
            x: 1.0,
            y: 1.0,
            z: -1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 6.0/6.0, y: 1.0 },
    },
    Vertex {
        position: Vector3 {
            x: 1.0,
            y: -1.0,
            z: -1.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        texture_coordinate: Vector2 { x: 6.0/6.0, y: 0.0 },
    },
];



//test
pub(crate) const QUAD_INDICES: [u32; 6] = [
    0, 1, 2, 2, 3, 0,
];


pub(crate) const CUBE_INDICES: [u32; 36] = [
    0, 1, 2, 2, 3, 0, //front face
    4, 5, 6, 6, 7, 4, //right face
    8, 9, 10, 10, 11, 8, //back face
    12, 13, 14, 14, 15, 12, //left face
    16, 17, 18, 18, 19, 16, //top face
    20, 21, 22, 22, 23, 20, //bottom face
];

// pub(crate) const SKYBOX_BACKPLANE_INDICES: [u32; 36] = [
//     //front face
//     0, 1, 2, 2, 3, 0, //right face
//     3, 7, 2, 2, 6, 7, //back face
//     4, 5, 6, 6, 7, 4, //left face
//     4, 5, 1, 1, 0, 4, //top face
//     4, 0, 3, 3, 7, 4, //bottom face
//     1, 2, 6, 6, 5, 1,
// ];
