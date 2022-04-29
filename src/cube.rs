use crate::support::Vertex;
use nalgebra::{Vector2, Vector3};

pub(crate) const FULLSCREEN_QUAD_VERTICES: [Vertex; 4] = [
    Vertex::new(
        Vector3::new(1.0, -1.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
        Vector2::new(1.0, 0.0),
    ),
    Vertex::new(
        Vector3::new(1.0, 1.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
        Vector2::new(1.0, 1.0),
    ),
    Vertex::new(
        Vector3::new(-1.0, 1.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
        Vector2::new(0.0, 1.0),
    ),
    Vertex::new(
        Vector3::new(-1.0, -1.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
        Vector2::new(0.0, 0.0),
    ),
];

pub(crate) const QUAD_INDICES: [u32; 6] = [0, 1, 2, 2, 3, 0];
