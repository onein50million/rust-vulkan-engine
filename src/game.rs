use std::borrow::BorrowMut;
use crate::marching_cubes::{World, WORLD_SIZE_X, WORLD_SIZE_Y, WORLD_SIZE_Z};
use crate::renderer::*;
use nalgebra::{
    Isometry3, Matrix4, Point3, Rotation3, Translation3, UnitQuaternion, Vector2, Vector3,
};
use parry3d_f64::query::contact;
use parry3d_f64::shape::{Capsule, TriMesh};
use std::convert::TryInto;
use std::mem::size_of;
use std::option::Option::None;

use std::time::Instant;
use slotmap::{new_key_type, SlotMap};
use winit::window::Window;

use bincode::{Decode, Encode};

/*
Some random ideas:

player has states, ie crouching, aiming, maybe a bitfield-like object?

compile time gltf parsing


Game object types:
Server object: calculated purely on server, client just copies data from server
Client predicted: Uses client side prediction to reduce latency
Client only (Animations, particles)
 */


mod directions {
    use nalgebra::Vector3;
    use std::f64::consts::FRAC_1_SQRT_2;

    pub(crate) const UP: Vector3<f64> = Vector3::new(0.0, -1.0, 0.0);
    pub(crate) const DOWN: Vector3<f64> = Vector3::new(0.0, 1.0, 0.0);
    pub(crate) const LEFT: Vector3<f64> = Vector3::new(-1.0, 0.0, 0.0);
    pub(crate) const RIGHT: Vector3<f64> = Vector3::new(1.0, 0.0, 0.0);
    pub(crate) const FORWARDS: Vector3<f64> = Vector3::new(0.0, 0.0, 1.0);
    pub(crate) const BACKWARDS: Vector3<f64> = Vector3::new(0.0, 0.0, -1.0);

    pub(crate) const ISOMETRIC_DOWN: Vector3<f64> =
        Vector3::new(-FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2);
    pub(crate) const ISOMETRIC_UP: Vector3<f64> = Vector3::new(FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2);
    pub(crate) const ISOMETRIC_RIGHT: Vector3<f64> =
        Vector3::new(-FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2);
    pub(crate) const ISOMETRIC_LEFT: Vector3<f64> =
        Vector3::new(FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2);
}

new_key_type!{
    pub struct GameObjectKey;
}


pub(crate) mod server{
    use std::collections::hash_map::ValuesMut;
    use std::convert::TryInto;
    use std::net::SocketAddr;
    use std::time::Instant;
    use fastrand::f64;
    use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector3};
    use parry3d_f64::query::contact;
    use parry3d_f64::shape::{Capsule, TriMesh};
    use slotmap::SlotMap;
    use crate::game::{directions, GameObjectKey};
    use crate::marching_cubes::{World, WORLD_SIZE_Y};
    use crate::support::Inputs;

    pub enum ObjectType {
        None,
        Grip(bool),
        Player(Player),
    }
    pub struct GameObject {
        pub position: Vector3<f64>,
        pub rotation: UnitQuaternion<f64>,
        pub object_type: ObjectType
    }
    impl GameObject {
        pub fn new(object_type: ObjectType) -> Self {
            return Self {
                position: Vector3::new(0.0, 0.0, 0.0),
                rotation: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.0),
                object_type
            };
        }

    }
    struct Game {
        game_start: Instant,
        game_objects: SlotMap<GameObjectKey, GameObject>,
        last_frame_instant: Instant,
        world: World,
    }

    pub struct Player {
        velocity: Vector3<f64>,
        last_rotation: UnitQuaternion<f64>,
        inputs: Inputs
    }
    impl Player{
        fn process(&mut self, delta_time: f64, world_isometry: Isometry3<f64>, world: &World, position: &mut Vector3<f64>, rotation: &mut UnitQuaternion<f64>){

            let friction = Vector3::new(10.0, 0.1, 10.0);
            let acceleration = 100.0;
            if self.inputs.left_click {
                position.y = -2.0 * (WORLD_SIZE_Y as f64);
                self.velocity.y = 0.0;
                self.inputs.left_click = false;
            }

            self.velocity += (self.inputs.up * directions::ISOMETRIC_UP
                + self.inputs.down * directions::ISOMETRIC_DOWN
                + self.inputs.left * directions::ISOMETRIC_LEFT
                + self.inputs.right * directions::ISOMETRIC_RIGHT)
                .try_normalize(0.1)
                .unwrap_or(Vector3::zeros())
                * acceleration
                * delta_time;

            *rotation = if self.velocity.magnitude() > 0.1 {
                UnitQuaternion::face_towards(&self.velocity, &Vector3::new(0.0, 1.0, 0.0))
            } else {
                self.last_rotation
            };

            self.velocity -= self.velocity.component_mul(&friction) * delta_time.min(1.0);

            *position += self.velocity * delta_time;

            let player_capsule = Capsule::new(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, -Player::HEIGHT.y, 0.0),
                0.5,
            );

            let player_isometry =
                Isometry3::from_parts(Translation3::from(*position), *rotation);

            let contact_result = contact(
                &world_isometry,
                world.collision.as_ref().unwrap(),
                &player_isometry,
                &player_capsule,
                10.0,
            )
                .unwrap();

            if contact_result.is_none() || contact_result.unwrap().dist > 0.1 {
                self.velocity.y += 9.8 * delta_time
            } else {
                self.velocity.y = 0.0;
                position.y = contact_result.unwrap().point1.y;
            }
        }
    }
    impl Player {
        const HEIGHT: Vector3<f64> = Vector3::new(0.0, 1.7, 0.0);
    }

    impl Game {
        pub fn new() -> Self {
            let mut world = World::new_random();

            let game = Self{
                game_start: Instant::now(),
                game_objects: SlotMap::with_key(),
                last_frame_instant: Instant::now(),
                world,
            };
            game
        }

        pub fn process(&mut self){
            let delta_time = self.last_frame_instant.elapsed().as_secs_f64();
            self.last_frame_instant = std::time::Instant::now();


            for object in self.game_objects.values_mut(){
                match &mut object.object_type{
                    ObjectType::None => {}
                    ObjectType::Grip(_) => {}
                    ObjectType::Player(player) => {player.process(delta_time, Isometry3::identity(),&self.world, &mut object.position, &mut object.rotation)
                    }
                }
            }

        }

        // pub fn add_player(&mut self){
        //     self.game_objects.insert(GameObject::new(ObjectType::Player(Player{
        //         last_rotation: UnitQuaternion::identity(),
        //         velocity: Vector3::zeros(),
        //     })))
        // }


    }

}
pub mod client{
    use nalgebra::{UnitQuaternion, Vector2, Vector3};
    use slotmap::SlotMap;
    use crate::game::GameObjectKey;
    use crate::marching_cubes::World;
    use crate::renderer::VulkanData;
    use crate::support::Inputs;

    pub struct AnimationHandler {
        pub index: usize,
        pub previous_frame: usize,
        pub next_frame: usize,
        frame_count: usize,
        frame_rate: f64,
        pub progress: f64,
    }
    impl AnimationHandler {
        fn new(index: usize, frame_count: usize) -> Self {
            Self {
                index,
                previous_frame: 0,
                next_frame: 1,
                frame_count,
                frame_rate: 60.0,
                progress: 0.0,
            }
        }
        fn process(&mut self, delta_time: f64) {
            self.progress += delta_time * self.frame_rate;
            if self.progress > 1.0 {
                self.progress = 0.0;
                self.previous_frame = (self.previous_frame + 1) % self.frame_count;
                self.next_frame = (self.next_frame + 1) % self.frame_count;
            }
        }
        fn switch_animation(&mut self, vulkan_data: &VulkanData, render_object_index: usize, animation_index: usize){
            self.previous_frame = 0;
            self.next_frame = 1;
            self.frame_count = vulkan_data.objects[render_object_index].get_animation_length(animation_index);
            self.progress = 0.0
        }
    }

    pub enum ObjectType {
        None,
        Grip(bool),
        Player,
    }
    pub struct GameObject {
        pub position: Vector3<f64>,
        pub rotation: UnitQuaternion<f64>,
        pub object_type: ObjectType,
        pub render_object_index: usize,
        pub animation_handler: Option<AnimationHandler>,
    }

    pub struct Game {
        pub game_objects: SlotMap<GameObjectKey, GameObject>,
        pub world: World,
        pub inputs: Inputs,
        pub mouse_position: Vector2<f64>,
    }
    impl Game{
        pub fn new() -> Self{
            let world =  World::new_random();

            Self{
                game_objects: SlotMap::with_key(),
                world,
                inputs: Inputs::new(),
                mouse_position: Vector2::zeros()
            }
        }
        pub fn process(&mut self, delta_time: f64){
            //Do stuff
        }
    }

}

pub mod sendable{
    use nalgebra::{UnitQuaternion, Vector3};
    use crate::game::GameObjectKey;
    use crate::marching_cubes::Voxel;
    use serde::{Serialize, Deserialize};
    use slotmap::SlotMap;

    #[derive(Serialize, Deserialize, Clone, Debug)]
    pub struct World{
        voxels: Vec<Voxel>,
    }


    #[derive(Serialize, Deserialize, Copy,Clone, Debug)]
    pub enum ObjectType {
        None,
        Grip,
        Player,
    }

    #[derive(Serialize, Deserialize, Copy, Clone, Debug)]
    pub struct GameObject{
        pub position: Vector3<f64>,
        pub rotation: UnitQuaternion<f64>,
        pub object_type: ObjectType
    }


    #[derive(Serialize, Deserialize, Clone)]
    pub struct Game {
        world: Option<World>,
        game_objects: SlotMap<GameObjectKey, GameObject>,
    }

}




