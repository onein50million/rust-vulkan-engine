use crate::Vertex;
use nalgebra::{Vector2, Vector3, Vector4};
use parry3d_f64::shape::TriMesh;
use std::convert::TryInto;
use std::ops::{Div, Sub};

pub(crate) const WORLD_SIZE_X: usize = 10;
pub(crate) const WORLD_SIZE_Z: usize = 10;
pub(crate) const WORLD_SIZE_Y: usize = 10;

const SURFACE_LEVEL: f32 = 0.0;

fn map_range_linear(value: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    let result = f32::clamp(
        to_min + ((value - from_min) / (from_max - from_min)) * (to_max - to_min),
        f32::min(to_min, to_max),
        f32::max(to_max, to_min),
    );
    return result;
}

#[derive(Copy, Clone)]
struct Voxel {
    strength: f32,
}

#[derive(Copy, Clone)]
struct Cube {
    voxels: [Voxel; 8],
}

impl Into<u8> for Cube {
    fn into(self) -> u8 {
        let mut output = 0;
        for (i, voxel) in self.voxels.iter().enumerate() {
            output += ((voxel.strength > SURFACE_LEVEL) as u8) << i;
        }
        return output;
    }
}

pub(crate) struct World {
    voxels: Vec<Voxel>,
    pub(crate) collision: Option<TriMesh>,
}

impl World {
    fn get_voxel(&self, x: usize, z: usize, y: usize) -> &Voxel {
        return &self.voxels[x * WORLD_SIZE_Y * WORLD_SIZE_Z + z * WORLD_SIZE_Y + y];
    }
    fn get_mut_voxel(&mut self, x: usize, z: usize, y: usize) -> &mut Voxel {
        return &mut self.voxels[x * WORLD_SIZE_Y * WORLD_SIZE_Z + z * WORLD_SIZE_Y + y];
    }

    pub(crate) fn new_random() -> Self {
        let rng = fastrand::Rng::new();

        let mut voxels = Self {
            voxels: vec![Voxel { strength: 0.0 }; WORLD_SIZE_X * WORLD_SIZE_Z * WORLD_SIZE_Y],
            collision: None,
        };
        for x_index in 0..(WORLD_SIZE_X - 1) {
            for z_index in 1..WORLD_SIZE_Z {
                for y_index in 0..(WORLD_SIZE_Y - 1) {
                    // voxels.get_mut_voxel(x_index, z_index, y_index).strength = (rng.f32() - 0.5) * 10.0;
                    if y_index > 4 {
                        voxels.get_mut_voxel(x_index, z_index, y_index).strength = 2.0;
                    } else if rng.bool() {
                        voxels.get_mut_voxel(x_index, z_index, y_index).strength = -0.5;
                    }
                }
            }
        }
        // let voxels = Box::new(voxels.try_into().unwrap());
        return voxels;
    }

    pub(crate) fn generate_mesh(&self) -> Vec<Vertex> {
        let mut output = vec![];
        for x_index in 0..(WORLD_SIZE_X - 1) {
            for z_index in 1..WORLD_SIZE_Z {
                for y_index in 0..(WORLD_SIZE_Y - 1) {
                    // dbg!(self.get_voxel(x_index,z_index,y_index).drawn);
                    let cube = Cube {
                        voxels: [
                            *self.get_voxel(x_index, z_index, y_index + 1),
                            *self.get_voxel(x_index + 1, z_index, y_index + 1),
                            *self.get_voxel(x_index + 1, z_index - 1, y_index + 1),
                            *self.get_voxel(x_index, z_index - 1, y_index + 1),
                            *self.get_voxel(x_index, z_index, y_index),
                            *self.get_voxel(x_index + 1, z_index, y_index),
                            *self.get_voxel(x_index + 1, z_index - 1, y_index),
                            *self.get_voxel(x_index, z_index - 1, y_index),
                        ],
                    };

                    let cube_index: u8 = cube.into();
                    let edges = crate::edge_triangulation::TABLE[cube_index as usize];

                    assert_eq!(edges.len() % 3, 0);

                    for edge in edges {
                        let x = match edge {
                            0 => map_range_linear(
                                SURFACE_LEVEL,
                                cube.voxels[0].strength,
                                cube.voxels[1].strength,
                                0.0,
                                1.0,
                            ),
                            1 => 1.0,
                            2 => map_range_linear(
                                SURFACE_LEVEL,
                                cube.voxels[3].strength,
                                cube.voxels[2].strength,
                                0.0,
                                1.0,
                            ),
                            3 => 0.0,
                            4 => map_range_linear(
                                SURFACE_LEVEL,
                                cube.voxels[4].strength,
                                cube.voxels[5].strength,
                                0.0,
                                1.0,
                            ),
                            5 => 1.0,
                            6 => map_range_linear(
                                SURFACE_LEVEL,
                                cube.voxels[7].strength,
                                cube.voxels[6].strength,
                                0.0,
                                1.0,
                            ),
                            7 => 0.0,
                            8 => 0.0,
                            9 => 1.0,
                            10 => 1.0,
                            11 => 0.0,
                            &_ => {
                                panic!()
                            }
                        };
                        let y = match edge {
                            0 => 1.0,
                            1 => 1.0,
                            2 => 1.0,
                            3 => 1.0,
                            4 => 0.0,
                            5 => 0.0,
                            6 => 0.0,
                            7 => 0.0,
                            8 => map_range_linear(
                                SURFACE_LEVEL,
                                cube.voxels[4].strength,
                                cube.voxels[0].strength,
                                0.0,
                                1.0,
                            ),
                            9 => map_range_linear(
                                SURFACE_LEVEL,
                                cube.voxels[5].strength,
                                cube.voxels[1].strength,
                                0.0,
                                1.0,
                            ),
                            10 => map_range_linear(
                                SURFACE_LEVEL,
                                cube.voxels[6].strength,
                                cube.voxels[2].strength,
                                0.0,
                                1.0,
                            ),
                            11 => map_range_linear(
                                SURFACE_LEVEL,
                                cube.voxels[7].strength,
                                cube.voxels[3].strength,
                                0.0,
                                1.0,
                            ),
                            &_ => {
                                panic!()
                            }
                        };
                        let z = match edge {
                            0 => 1.0,
                            1 => map_range_linear(
                                SURFACE_LEVEL,
                                cube.voxels[2].strength,
                                cube.voxels[1].strength,
                                0.0,
                                1.0,
                            ),
                            2 => 0.0,
                            3 => map_range_linear(
                                SURFACE_LEVEL,
                                cube.voxels[3].strength,
                                cube.voxels[0].strength,
                                0.0,
                                1.0,
                            ),
                            4 => 1.0,
                            5 => map_range_linear(
                                SURFACE_LEVEL,
                                cube.voxels[6].strength,
                                cube.voxels[5].strength,
                                0.0,
                                1.0,
                            ),
                            6 => 0.0,
                            7 => map_range_linear(
                                SURFACE_LEVEL,
                                cube.voxels[7].strength,
                                cube.voxels[4].strength,
                                0.0,
                                1.0,
                            ),
                            8 => 1.0,
                            9 => 1.0,
                            10 => 0.0,
                            11 => 0.0,
                            &_ => {
                                panic!()
                            }
                        };
                        let position = Vector3::new(
                            x_index as f32 + x,
                            y_index as f32 + y,
                            z_index as f32 + z,
                        );
                        let normal = Vector3::new(0.0, 0.0, 1.0);
                        let tangent = Vector4::new(0.0, 1.0, 0.0, 1.0);
                        let texture_coordinate = Vector2::zeros();
                        let texture_type = 0;

                        output.push(Vertex {
                            position,
                            normal,
                            tangent,
                            texture_coordinate,
                            texture_type,
                            bone_indices: Vector4::zeros(),
                            bone_weights: Vector4::zeros(),
                        })
                    }
                }
            }
        }

        for triangle_index in (0..output.len()).step_by(3) {
            for i in 0..3 {
                let edge1 = output[triangle_index + i].position
                    - output[triangle_index + ((i + 1) % 3)].position;
                let edge2 = output[triangle_index + i].position
                    - output[triangle_index + ((i + 2) % 3)].position;
                let normal = edge2.cross(&edge1);
                output[triangle_index + i].normal =
                    normal.try_normalize(0.1).unwrap_or(Vector3::zeros())
            }
        }

        // for vertex in &mut output {
        //     vertex.normal = vertex.normal.normalize();
        // }

        return output;
    }
}