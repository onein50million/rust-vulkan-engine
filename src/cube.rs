use crate::support::Vertex;
use nalgebra::{Vector3};

/*

   0   3

   1   2

4   7

5   6

*/
mod texture_points{

    pub(crate) mod x{
        pub(crate) mod positive{ //RIGHT
            pub(crate) const BOTTOM_LEFT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(1.0/6.0, 0.0);
            pub(crate) const BOTTOM_RIGHT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(1.0/6.0, 1.0);
            pub(crate) const TOP_LEFT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(0.0/6.0, 0.0);
            pub(crate) const TOP_RIGHT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(0.0/6.0, 1.0);

        }
        pub(crate) mod negative{ //LEFT
            pub(crate) const BOTTOM_LEFT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(1.0/6.0, 1.0);
            pub(crate) const BOTTOM_RIGHT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(1.0/6.0, 0.0);
            pub(crate) const TOP_LEFT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(2.0/6.0, 1.0);
            pub(crate) const TOP_RIGHT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(2.0/6.0, 0.0);

        }
    }
    pub(crate) mod y{
        pub(crate) mod positive{ //BOTTOM
            pub(crate) const BOTTOM_LEFT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(5.0/6.0, 0.0);
            pub(crate) const BOTTOM_RIGHT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(6.0/6.0, 0.0);
            pub(crate) const TOP_LEFT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(5.0/6.0, 1.0);
            pub(crate) const TOP_RIGHT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(6.0/6.0, 1.0);

        }
        pub(crate) mod negative{//TOP

            pub(crate) const BOTTOM_LEFT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(4.0/6.0, 0.0);
            pub(crate) const BOTTOM_RIGHT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(5.0/6.0, 0.0);
            pub(crate) const TOP_LEFT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(4.0/6.0, 1.0);
            pub(crate) const TOP_RIGHT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(5.0/6.0, 1.0);

        }
    }
    pub(crate) mod z{
        pub(crate) mod positive{//BACK
            pub(crate) const BOTTOM_LEFT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(3.0/6.0, 1.0);
            pub(crate) const BOTTOM_RIGHT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(2.0/6.0, 1.0);
            pub(crate) const TOP_LEFT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(3.0/6.0, 0.0);
            pub(crate) const TOP_RIGHT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(2.0/6.0, 0.0);
        }
        pub(crate) mod negative{//FRONT
            pub(crate) const BOTTOM_LEFT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(3.0/6.0, 0.0);
            pub(crate) const BOTTOM_RIGHT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(4.0/6.0, 0.0);
            pub(crate) const TOP_LEFT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(3.0/6.0, 1.0);
            pub(crate) const TOP_RIGHT: nalgebra::Vector2<f32> = nalgebra::Vector2::new(4.0/6.0, 1.0);

        }
    }


}


pub(crate) const POSITIVE_X_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3::new(1.0, -1.0, -1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::x::positive::TOP_RIGHT,
    },
    Vertex {
        position: Vector3::new(1.0, 1.0, -1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::x::positive::BOTTOM_RIGHT,
    },
    Vertex {
        position: Vector3::new(1.0, 1.0, 1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::x::positive::BOTTOM_LEFT,
    },
    Vertex {
        position: Vector3::new(1.0, -1.0, 1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::x::positive::TOP_LEFT,
    },
];


pub(crate) const NEGATIVE_X_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3::new(-1.0, -1.0, 1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::x::negative::TOP_RIGHT,
    },
    Vertex {
        position: Vector3::new(-1.0, 1.0, 1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::x::negative::BOTTOM_RIGHT,
    },
    Vertex {
        position: Vector3::new(-1.0, 1.0, -1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::x::negative::BOTTOM_LEFT,
    },
    Vertex {
        position: Vector3::new(-1.0, -1.0, -1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::x::negative::TOP_LEFT,
    },
];


pub(crate) const POSITIVE_Y_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3::new(1.0, 1.0, 1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::y::positive::BOTTOM_LEFT,
    },
    Vertex {
        position: Vector3::new(1.0, 1.0, -1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::y::positive::TOP_LEFT
    },
    Vertex {
        position: Vector3::new(-1.0, 1.0, -1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::y::positive::TOP_RIGHT,
    },
    Vertex {
        position: Vector3::new(-1.0, 1.0, 1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::y::positive::BOTTOM_RIGHT,
    },
];

pub(crate) const NEGATIVE_Y_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3::new(-1.0, -1.0, 1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::y::negative::BOTTOM_LEFT,
    },
    Vertex {
        position: Vector3::new(-1.0, -1.0, -1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::y::negative::TOP_LEFT,
    },
    Vertex {
        position: Vector3::new(1.0, -1.0, -1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::y::negative::TOP_RIGHT,
    },
    Vertex {
        position: Vector3::new(1.0, -1.0, 1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::y::negative::BOTTOM_RIGHT,
    },
];

pub(crate) const POSITIVE_Z_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3::new(1.0, -1.0, 1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::z::positive::BOTTOM_LEFT,
    },
    Vertex {
        position: Vector3::new(1.0, 1.0, 1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::z::positive::TOP_LEFT,
    },
    Vertex {
        position: Vector3::new(-1.0, 1.0, 1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::z::positive::TOP_RIGHT,
    },
    Vertex {
        position: Vector3::new(-1.0, -1.0, 1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::z::positive::BOTTOM_RIGHT,
    },
];

pub(crate) const NEGATIVE_Z_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vector3::new(-1.0, -1.0, -1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::z::negative::BOTTOM_LEFT,
    },
    Vertex {
        position: Vector3::new(-1.0, 1.0, -1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::z::negative::TOP_LEFT,
    },
    Vertex {
        position: Vector3::new(1.0, 1.0, -1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::z::negative::TOP_RIGHT,
    },
    Vertex {
        position: Vector3::new(1.0, -1.0, -1.0),
        normal: Vector3::new(1.0, 1.0, 1.0),
        texture_coordinate: texture_points::z::negative::BOTTOM_RIGHT,
    },
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