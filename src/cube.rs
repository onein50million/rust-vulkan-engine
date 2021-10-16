use crate::support::Vertex;
use nalgebra::{Vector2, Vector3};

/*

   0   3

   1   2

4   7

5   6

*/

pub(crate) const NEGATIVE_X_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3::new(-1.0, -1.0, 1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(0.0, 0.0),
    },
    Vertex {
        position: Vector3::new(-1.0, 1.0, 1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(0.0, 1.0),
    },
    Vertex {
        position: Vector3::new(-1.0, 1.0, -1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(1.0, 1.0),
    },
    Vertex {
        position: Vector3::new(-1.0, -1.0, -1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(1.0, 1.0),
    },
];

pub(crate) const POSITIVE_X_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3::new(1.0, -1.0, -1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(0.0, 0.0),
    },
    Vertex {
        position: Vector3::new(1.0, 1.0, -1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(0.0, 1.0),
    },
    Vertex {
        position: Vector3::new(1.0, 1.0, 1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(1.0, 1.0),
    },
    Vertex {
        position: Vector3::new(1.0, -1.0, 1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(1.0, 0.0),
    },
];

pub(crate) const NEGATIVE_Y_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3::new(-1.0, -1.0, 1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(0.0, 0.0),
    },
    Vertex {
        position: Vector3::new(-1.0, -1.0, -1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(0.0, 1.0),
    },
    Vertex {
        position: Vector3::new(1.0, -1.0, -1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(1.0, 1.0),
    },
    Vertex {
        position: Vector3::new(1.0, -1.0, 1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(1.0, 0.0),
    },
];

pub(crate) const POSITIVE_Y_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3::new(1.0, 1.0, 1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(1.0, 1.0),
    },
    Vertex {
        position: Vector3::new(1.0, 1.0, -1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(1.0, 0.0),
    },
    Vertex {
        position: Vector3::new(-1.0, 1.0, -1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(0.0, 0.0),
    },
    Vertex {
        position: Vector3::new(-1.0, 1.0, 1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(0.0, 1.0),
    },
];

pub(crate) const NEGATIVE_Z_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3::new(-1.0, -1.0, -1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(0.0, 0.0),
    },
    Vertex {
        position: Vector3::new(-1.0, 1.0, -1.0),
        color: Vector3::new(1.0, 1.0, 1.0),

        texture_coordinate: Vector2::new(0.0, 1.0),
    },
    Vertex {
        position: Vector3::new(1.0, 1.0, -1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(1.0, 1.0),
    },
    Vertex {
        position: Vector3::new(1.0, -1.0, -1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(1.0, 0.0),
    },
];

pub(crate) const POSITIVE_Z_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3::new(1.0, -1.0, 1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(1.0, 0.0),
    },
    Vertex {
        position: Vector3::new(1.0, 1.0, 1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(1.0, 1.0),
    },
    Vertex {
        position: Vector3::new(-1.0, 1.0, 1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(0.0, 1.0),
    },
    Vertex {
        position: Vector3::new(-1.0, -1.0, 1.0),
        color: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: Vector2::new(0.0, 0.0),
    },
];

pub(crate) const QUAD_INDICES: [u32; 6] = [0, 1, 2, 2, 3, 0];

// pub(crate) const SKYBOX_BACKPLANE_INDICES: [u32; 36] = [
//     //front face
//     0, 1, 2, 2, 3, 0, //right face
//     3, 7, 2, 2, 6, 7, //back face
//     4, 5, 6, 6, 7, 4, //left face
//     4, 5, 1, 1, 0, 4, //top face
//     4, 0, 3, 3, 7, 4, //bottom face
//     1, 2, 6, 6, 5, 1,
// ];
