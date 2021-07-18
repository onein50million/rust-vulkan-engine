use cgmath::{Vector3,Vector2};
use crate::support::Vertex;


pub(crate) const SKYBOX_BACKPLANE_VERTICES: [Vertex;4] = [
    Vertex {
        position: Vector3 {
            x: -1.0,
            y: -1.0,
            z: 0.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0
        },
        texture_coordinate: Vector2 { x: 0.0, y: 0.0 }
    }, Vertex {
        position: Vector3 {
            x: -1.0,
            y: 1.0,
            z: 0.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0
        },
        texture_coordinate: Vector2 { x: 0.0, y: 1.0 }
    }, Vertex {
        position: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 0.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0
        },
        texture_coordinate: Vector2 { x: 1.0, y: 1.0 }
    }, Vertex {
        position: Vector3 {
            x: 1.0,
            y: -1.0,
            z: 0.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0
        },
        texture_coordinate: Vector2 { x: 1.0, y: 0.0 }
    }
];
pub(crate) const SKYBOX_BACKPLANE_INDICES: [u32;6] = [0,1,2,2,3,0];
