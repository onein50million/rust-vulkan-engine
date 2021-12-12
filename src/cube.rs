use crate::support::Vertex;
use nalgebra::{Vector2, Vector3};


pub(crate) const FULLSCREEN_QUAD_VERTICES: [Vertex; 4] = [
    Vertex::new(Vector3::new(1.0, -1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(1.0, 0.0)),
    Vertex::new(Vector3::new(1.0, 1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(1.0, 1.0)),
    Vertex::new(Vector3::new(-1.0, 1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 1.0)),
    Vertex::new(Vector3::new(-1.0, -1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
];

pub(crate) const POSITIVE_X_VERTICES: [Vertex; 4] = [
    Vertex::new(Vector3::new(1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(1.0, 1.0, -1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(1.0, 1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(1.0, -1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
];

pub(crate) const NEGATIVE_X_VERTICES: [Vertex; 4] = [
    Vertex::new(Vector3::new(-1.0, -1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(-1.0, 1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(-1.0, 1.0, -1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(-1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
];

pub(crate) const POSITIVE_Y_VERTICES: [Vertex; 4] = [
    Vertex::new(Vector3::new(1.0, 1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(1.0, 1.0, -1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(-1.0, 1.0, -1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(-1.0, 1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
];

pub(crate) const NEGATIVE_Y_VERTICES: [Vertex; 4] = [
    Vertex::new(Vector3::new(-1.0, -1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(-1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(1.0, -1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
];

pub(crate) const POSITIVE_Z_VERTICES: [Vertex; 4] = [
    Vertex::new(Vector3::new(1.0, -1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(1.0, 1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(-1.0, 1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(-1.0, -1.0, 1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
];

pub(crate) const NEGATIVE_Z_VERTICES: [Vertex; 4] = [
    Vertex::new(Vector3::new(-1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(-1.0, 1.0, -1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(1.0, 1.0, -1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
    Vertex::new(Vector3::new(1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0), Vector2::new(0.0, 0.0)),
];

pub(crate) const QUAD_INDICES: [u32; 6] = [0, 1, 2, 2, 3, 0];

pub(crate) const CUBEMAP_INDICES: [u32; 36] = [
    //front face
    0, 1, 2, 2, 3, 0, //right face
    4, 5, 6, 6, 7, 4, //back face
    8, 9, 10, 10, 11, 8, //left face
    12, 13, 14, 14, 15, 12, //top face
    16, 17, 18, 18, 19, 16, //bottom face
    20, 21, 22, 22, 23, 20,
];
