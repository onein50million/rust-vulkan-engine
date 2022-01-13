use nalgebra::{Vector2, Vector3, Vector4};
use crate::Vertex;

const WORLD_SIZE_X:usize = 100;
const WORLD_SIZE_Z:usize = 100;
const WORLD_SIZE_Y:usize = 3;


#[derive(Copy,Clone)]
struct Voxel{
    drawn: bool
}

struct Cube{
    voxels: [Voxel; 8]
}

impl Into<u8> for Cube{
    fn into(self) -> u8 {
        let mut output = 0;
        for (i,voxel) in self.voxels.iter().enumerate(){
            output += (voxel.drawn as u8) << i;
        }
        return output
    }
}

pub(crate) struct World{
    voxels: [
        [[Voxel; WORLD_SIZE_Y]; WORLD_SIZE_Z]; WORLD_SIZE_X //voxels[x][z][y]
    ]
}

impl World{
    pub(crate) fn new_random() -> Self{
        let mut voxels = [[[Voxel{ drawn: false }; WORLD_SIZE_Y]; WORLD_SIZE_Z]; WORLD_SIZE_X];
        let mut rng = fastrand::Rng::new();
        for x_index in 0..(WORLD_SIZE_X - 1){
            for z_index in 1..WORLD_SIZE_Z{
                for y_index in 0..(WORLD_SIZE_Y - 1) {
                    // voxels[x_index][z_index][y_index].drawn = true
                    voxels[x_index][z_index][y_index].drawn = y_index == 0 || (voxels[x_index][z_index][y_index-1].drawn && rng.bool())
                }
            }
        }
        return Self{voxels};
    }

    pub(crate) fn generate_mesh(&self) -> Vec<Vertex>{
        let mut output = vec![];
        for x_index in 0..(WORLD_SIZE_X - 1){
            for z_index in 1..WORLD_SIZE_Z{
                for y_index in 0..(WORLD_SIZE_Y - 1){


                    // let fallback = Voxel{ drawn: false };
                    //
                    // let bottom_back_left;
                    // let bottom_back_right;
                    // let bottom_front_left;
                    // let bottom_front_right;
                    // let top_back_left;
                    // let top_back_right;
                    // let top_front_left;
                    // let top_front_right;
                    //
                    // bottom_back_left = self.voxels[x_index][z_index][y_index];
                    // if (0..WORLD_SIZE).contains(&(x_index + 1)){
                    //     bottom_back_right = self.voxels[x_index + 1][z_index][y_index];
                    // }else{
                    //     bottom_back_right = fallback
                    // }
                    //
                    // if (0..WORLD_SIZE).contains(&(z_index - 1)){
                    //     bottom_front_left = self.voxels[x_index][z_index - 1][y_index];
                    // }else{
                    //     bottom_front_left = fallback
                    // }
                    // if (0..WORLD_SIZE).contains(&(z_index - 1)) && (0..WORLD_SIZE).contains(&(x_index + 1)) {
                    //     bottom_front_right = self.voxels[x_index + 1][z_index - 1][y_index];
                    // }else{
                    //     bottom_front_right = fallback
                    // }
                    // if (0..WORLD_SIZE).contains(&(y_index + 1)){
                    //     top_back_left = self.voxels[x_index][z_index][y_index + 1];
                    //     top_back_right = self.voxels[x_index + 1][z_index][y_index + 1];
                    //     top_front_left = self.voxels[x_index][z_index - 1][y_index + 1];
                    //     top_front_right = self.voxels[x_index + 1][z_index - 1][y_index + 1];
                    //
                    // }else{
                    //     top_back_left = fallback;
                    //     top_back_right = fallback;
                    //     top_front_left = fallback;
                    //     top_front_right = fallback;
                    // }
                    //
                    //
                    // let cube =  Cube{
                    //     bottom_back_left,
                    //     bottom_back_right,
                    //     bottom_front_left,
                    //     bottom_front_right,
                    //     top_back_left,
                    //     top_back_right,
                    //     top_front_left,
                    //     top_front_right
                    // };

                    // let cube = Cube{
                    //      voxels: [self.voxels[x_index][z_index][y_index],
                    //      self.voxels[x_index + 1][z_index][y_index],
                    //      self.voxels[x_index][z_index - 1][y_index],
                    //      self.voxels[x_index + 1][z_index - 1][y_index],
                    //      self.voxels[x_index][z_index][y_index + 1],
                    //      self.voxels[x_index + 1][z_index][y_index + 1],
                    //      self.voxels[x_index][z_index - 1][y_index + 1],
                    //     self.voxels[x_index + 1][z_index - 1][y_index + 1]]
                    // };

                    let cube = Cube{
                        voxels: [self.voxels[x_index][z_index][y_index + 1],
                            self.voxels[x_index + 1][z_index][y_index + 1],
                            self.voxels[x_index + 1][z_index - 1][y_index + 1],
                            self.voxels[x_index][z_index - 1][y_index + 1],
                            self.voxels[x_index][z_index][y_index ],
                            self.voxels[x_index + 1][z_index][y_index ],
                            self.voxels[x_index + 1][z_index - 1][y_index ],
                            self.voxels[x_index][z_index - 1][y_index],]
                    };

                    let cube: u8 = cube.into();
                    let edges = crate::edge_triangulation::TABLE[cube as usize];

                    assert_eq!(edges.len() % 3, 0);

                    for edge in edges{
                        let x = match edge{
                            0 => 0.5,
                            1 => 1.0,
                            2 => 0.5,
                            3 => 0.0,
                            4 => 0.5,
                            5 => 1.0,
                            6 => 0.5,
                            7 => 0.0,
                            8 => 0.0,
                            9 => 1.0,
                            10 => 1.0,
                            11 => 0.0,
                            &_ => {panic!()}
                        };
                        let y = match edge{
                            0 => 1.0,
                            1 => 1.0,
                            2 => 1.0,
                            3 => 1.0,
                            4 => 0.0,
                            5 => 0.0,
                            6 => 0.0,
                            7 => 0.0,
                            8 => 0.5,
                            9 => 0.5,
                            10 => 0.5,
                            11 => 0.5,
                            &_ => {panic!()}
                        };
                        let z= match edge{
                            0 => 1.0,
                            1 => 0.5,
                            2 => 0.0,
                            3 => 0.5,
                            4 => 1.0,
                            5 => 0.5,
                            6 => 0.0,
                            7 => 0.5,
                            8 => 1.0,
                            9 => 1.0,
                            10 => 0.0,
                            11 => 0.0,
                            &_ => {panic!()}
                        };
                        let position = Vector3::new(x_index as f32 + x, y_index as f32 + y, z_index as f32 + z);
                        let normal = Vector3::new(0.0,0.0,1.0);
                        let tangent = Vector4::new(0.0,1.0,0.0,1.0);
                        let texture_coordinate = Vector2::zeros();
                        let texture_type = 0;

                        output.push(Vertex{
                            position,
                            normal,
                            tangent,
                            texture_coordinate,
                            texture_type
                        }
                        )
                    }

                }
            }
        }


        for triangle_index in (0..output.len()).step_by(3) {
            let edge1 = output[triangle_index]
                .position
                - output[triangle_index + 1]
                .position;
            let edge2 = output[triangle_index]
                .position
                - output[triangle_index + 2]
                .position;
            let weighted_normal = edge1.cross(&edge2);
            for i in 0..3 {
                output[triangle_index + i]
                    .normal += weighted_normal
            }
        }

        for vertex in &mut output{
            vertex.normal = vertex.normal.normalize()
        }

        return output;

    }
}