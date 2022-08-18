/*
Some random ideas:

player has states, ie crouching, aiming, maybe a bitfield-like object?

compile time gltf parsing


Game object types:
Server object: calculated purely on server, client just copies data from server
Client predicted: Uses client side prediction to reduce latency
Client only (Animations, particles)
 */

pub const PHYSICS_TICKRATE: f64 = 60.0;

pub mod directions {
    use nalgebra::Vector3;

    pub const UP: Vector3<f64> = Vector3::new(0.0, 1.0, 0.0);
    pub const DOWN: Vector3<f64> = Vector3::new(0.0, -1.0, 0.0);
    pub const LEFT: Vector3<f64> = Vector3::new(-1.0, 0.0, 0.0);
    pub const RIGHT: Vector3<f64> = Vector3::new(1.0, 0.0, 0.0);
    pub const FORWARDS: Vector3<f64> = Vector3::new(0.0, 0.0, 1.0);
    pub const BACKWARDS: Vector3<f64> = Vector3::new(0.0, 0.0, -1.0);
}

use std::f64::consts::E;
use std::iter;
use std::ops::IndexMut;
use std::{f64::consts::PI, ops::Index};
use std::time::Instant;

use crate::network::NETWORK_TICKRATE;
use crate::renderer::VulkanData;
use crate::support::{Inputs, FRAMERATE_TARGET};
use bincode::{Decode, Encode};
use float_ord::FloatOrd;
use lyon_tessellation::geom::euclid::num::Zero;
use nalgebra::{Matrix4, Perspective3, Point3, Translation3, UnitQuaternion, Vector2, Vector3, Rotation3};
// use rapier3d_f64::prelude::{IntegrationParameters, PhysicsPipeline, IslandManager, BroadPhase, NarrowPhase, ImpulseJointSet, MultibodyJointSet, CCDSolver, RigidBodySet, ColliderSet, EventHandler, PhysicsHooks, RigidBodyHandle};

pub struct AnimationHandler {
    pub index: usize,
    pub previous_frame: usize,
    pub next_frame: usize,
    frame_count: usize,
    frame_rate: f64,
    pub progress: f64,
}
impl AnimationHandler {
    pub fn new(index: usize, frame_count: usize) -> Self {
        Self {
            index,
            previous_frame: 0,
            next_frame: 1,
            frame_count,
            frame_rate: 60.0,
            progress: 0.0,
        }
    }
    pub fn process(&mut self, delta_time: f64) {
        self.progress += delta_time * self.frame_rate;
        if self.progress > 1.0 {
            self.progress = 0.0;
            self.previous_frame = (self.previous_frame + 1) % self.frame_count;
            self.next_frame = (self.next_frame + 1) % self.frame_count;
            // dbg!(self.previous_frame, self.next_frame);
        }
    }
    fn switch_animation(
        &mut self,
        vulkan_data: &VulkanData,
        render_object_index: usize,
        animation_index: usize,
    ) {
        self.previous_frame = 0;
        self.next_frame = 1;
        self.frame_count =
            vulkan_data.objects[render_object_index].get_animation_length(animation_index);
        self.progress = 0.0
    }
}

pub struct Camera {
    pub latitude: f64,
    pub longitude: f64,
    pub targeted_player: Option<PlayerKey>,
}
impl Camera {
    const OFFSET: Vector3<f64> = Vector3::new(3.0, -1.0, 0.0);
    fn new() -> Self {
        Self {
            latitude: 0.0,
            longitude: 0.0,
            targeted_player: None,
        }
    }
    fn get_rotation(&self, target_position: Vector3<f64>) -> UnitQuaternion<f64> {
        return UnitQuaternion::face_towards(&(self.get_position(target_position) - target_position), &Vector3::new(0.0, 1.0, 0.0));
    }
    fn get_position(&self, target_position: Vector3<f64>) -> Vector3<f64> {
        return 
            UnitQuaternion::from_euler_angles(0.0, self.longitude, 0.0)
            * UnitQuaternion::from_euler_angles(0.0, 0.0, self.latitude)
            * Self::OFFSET + target_position;
    }
    pub fn get_view_matrix(&self, target_position: Vector3<f64>) -> Matrix4<f64> {
        (Matrix4::from(Translation3::from(self.get_position(target_position)))
            * self.get_rotation(target_position).to_homogeneous())
        .try_inverse()
        .unwrap()
    }
    pub fn get_direction(&self, target_position: Vector3<f64>) -> Vector3<f64>{
        return self.get_rotation(target_position).transform_vector(&Vector3::new(0.0, 0.0, -1.0));
    }
}

pub struct GameObject {
    pub position: Vector3<f64>,
    pub rotation: UnitQuaternion<f64>,
    pub render_object_index: usize,
    pub animation_handler: Option<AnimationHandler>,
}
impl GameObject {
    pub fn get_transform(&self) -> Matrix4<f64> {
        return Matrix4::from(Translation3::from(self.position)) * self.rotation.to_homogeneous();
    }
}


//index 0 is newest, 1 is second newest, 2 is third newest, etc
pub struct CircularBuffer<T, const L: usize>{
    buffer: Box<[T]>,
    start: usize,
}
impl<T,const L: usize> CircularBuffer<T, L>{
    pub fn new<F:FnMut()-> T>(callback: F) -> Self{
        let buffer = iter::repeat_with(callback).take(L).collect();
        Self { buffer, start: 0 }
    }
    pub fn add_value(&mut self, value: T){
        self.start += 1;
        if self.start >= L{
            self.start = 0;
        }
        self.buffer[self.start] = value;
    }
}
impl<T,const L: usize> Index<usize> for CircularBuffer<T, L>{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index > L{
            eprintln!("Trying to access out of bounds in network buffer, clamping");
        }
        let index = index.min(L);
        &self.buffer[(self.start as i64 - index as i64).rem_euclid(L as i64 ) as usize]
    }
}
impl<T,const L: usize> IndexMut<usize> for CircularBuffer<T, L>{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index > L{
            eprintln!("Trying to access out of bounds in network buffer, clamping");
        }
        let index = index.min(L);
        &mut self.buffer[(self.start as i64 - index as i64).rem_euclid(L as i64 ) as usize]
    }
}

pub struct ClientPlayer{
    pub game_object: GameObject,
    pub last_server_player_state: PlayerState,
    pub last_client_input_from_server: f64,
    pub input_buffer: Option<CircularBuffer<Inputs, INPUT_BUFFER_LEN>>, //Only the local player has an input buffer
    extra_sim_time: f64,

}
impl ClientPlayer{
    pub fn new(render_object_index: usize, vulkan_data: &VulkanData) -> Self{
        Self{
            game_object: GameObject {
                position: Vector3::zeros(),
                rotation: UnitQuaternion::identity(),
                render_object_index,
                animation_handler: Some(AnimationHandler::new(2, vulkan_data.objects[render_object_index].animations[2].frame_count)),

                // animation_handler: None
            },
            last_server_player_state: PlayerState {
                position: Vector3::zeros(),
                velocity: Vector3::zeros(),
                is_jumping: false,
            },
            last_client_input_from_server: 0.0,
            input_buffer: None,
            extra_sim_time: 0.0,

        }
    }
    // pub fn update_gameobject(&mut self, client_start: &Instant){
    //     let mut new_state = self.last_server_player_state;
    //     let mut time_to_simulate = client_start.elapsed().as_secs_f64() - self.last_client_input_from_server;
    //     let mut ticks_behind = NetworkTick::new((time_to_simulate / (1.0 / FRAMERATE_TARGET)) as usize);
    //     while time_to_simulate > 1.0/FRAMERATE_TARGET{
    //         let input_at_time = match &self.input_buffer{
    //             Some(input_buffer) => input_buffer[ticks_behind],
    //             None => Inputs::new(),
    //         };

    //         simulate_new_player_state(&mut new_state, &input_at_time, 1.0/FRAMERATE_TARGET);
    //         ticks_behind.val -= 1;
    //         time_to_simulate -= 1.0/FRAMERATE_TARGET;
    //     }
    //     self.game_object.position = new_state.position;
    //     if let Some(input_buffer) = &self.input_buffer{
    //         self.game_object.rotation = UnitQuaternion::face_towards(&(input_buffer[0].camera_direction.component_mul(&Vector3::new(1.0,0.0,1.0))), &directions::UP)
    //     }
    // }

    pub fn update_gameobject(&mut self, client_start: &Instant){
        let mut new_state = self.last_server_player_state;

        let default_inputs = Inputs::new();
        let inputs = match &self.input_buffer{
            Some(inputs) => SimulationInput::Buffer(inputs),
            None => SimulationInput::SingleInput(&default_inputs),
        };

        simulate_new_player_state(&mut new_state, inputs, client_start.elapsed().as_secs_f64() - self.last_client_input_from_server, &mut self.extra_sim_time);
        self.game_object.position = new_state.position;
        if let Some(input_buffer) = &self.input_buffer{
            self.game_object.rotation = UnitQuaternion::face_towards(&(input_buffer[0].camera_direction.component_mul(&Vector3::new(1.0,0.0,1.0))), &directions::UP)
        }
    }

} 
const INPUT_BUFFER_LEN: usize = (4.0 * PHYSICS_TICKRATE) as usize; // 4 seconds of input buffer
pub struct ClientGame {
    pub inputs: Inputs,
    pub mouse_position: Vector2<f64>,
    pub camera: Camera,
    pub players: PlayerMap<ClientPlayer>,
}
impl ClientGame {
    pub fn new() -> Self {
        Self {
            inputs: Inputs::new(),
            camera: Camera::new(),
            mouse_position: Vector2::zero(),
            players: PlayerMap::new(),
        }
    }
}

safe_index::new! {
    #[derive(Encode, Decode)]
    NetworkTick
}

#[derive(Encode,Decode, Debug, Clone, Copy)]
pub struct PlayerState{
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub is_jumping: bool,
}
pub struct PlayerObject {
    pub player_state: PlayerState,
    pub inputs: Inputs,
    pub last_input_time: Option<f64>,
}

safe_index::new! {
    #[derive(Encode, Decode)]
    PlayerKey,
    map: PlayerMap
}

// pub struct PhysicsState{
//     gravity: Vector3<f64>,
//     integration_parameters: IntegrationParameters,
//     physics_pipeline: PhysicsPipeline,
//     island_manager: IslandManager,
//     broad_phase: BroadPhase,
//     narrow_phase: NarrowPhase,
//     pub rigid_body_set: RigidBodySet,
//     collider_set: ColliderSet,
//     impulse_joint_set: ImpulseJointSet,
//     multibody_joint_set: MultibodyJointSet,
//     ccd_solver: CCDSolver,

// }
// impl PhysicsState{
//     pub fn new() -> Self{
//         let gravity = Vector3::new(0.0, -9.81, 0.0);
//         let integration_parameters = IntegrationParameters::default();
//         let physics_pipeline = PhysicsPipeline::new();
//         let island_manager = IslandManager::new();
//         let broad_phase = BroadPhase::new();
//         let narrow_phase = NarrowPhase::new();
//         let rigid_body_set = RigidBodySet::new();
//         let collider_set = ColliderSet::new();
//         let impulse_joint_set = ImpulseJointSet::new();
//         let multibody_joint_set = MultibodyJointSet::new();
//         let ccd_solver = CCDSolver::new();

//         Self{
//             gravity,
//             integration_parameters,
//             physics_pipeline,
//             island_manager,
//             broad_phase,
//             narrow_phase,
//             rigid_body_set,
//             collider_set,
//             impulse_joint_set,
//             multibody_joint_set,
//             ccd_solver,
//         }
//     }
//     pub fn step(&mut self){
//         self.physics_pipeline.step(
//             &self.gravity,
//             &self.integration_parameters,
//             &mut self.island_manager,
//             &mut self.broad_phase,
//             &mut self.narrow_phase,
//             &mut self.rigid_body_set,
//             &mut self.collider_set,
//             &mut self.impulse_joint_set,
//             &mut self.multibody_joint_set,
//             &mut self.ccd_solver,
//             &(),
//             &())
//     }
// }

// fn simulate_new_player_state(player_state: &mut PlayerState, inputs: &Inputs, delta_time: f64){
//     if player_state.position.y > 0.0 {
//         player_state.velocity.y -= 9.8 * delta_time;
//         player_state.is_jumping = false;
//     } else {
//         player_state.velocity.y = 0.0;
//         let movement = inputs.camera_direction;
//         let sideways = movement.cross(&directions::UP) * inputs.left_stick.x;
//         let forward = movement * inputs.left_stick.y;

//         let mut movement = sideways + forward;
//         let movement_magnitude = inputs.left_stick.magnitude();
//         movement.y = 0.0;

//         movement = if movement.magnitude() > 0.01{movement.normalize() * movement_magnitude} else {Vector3::zeros()};
//         player_state.velocity += 50.0 * delta_time * movement;
//         if inputs.jump && !player_state.is_jumping{
//             player_state.is_jumping = true;
//             player_state.velocity.y = 10.0;
//             player_state.position.y = 1.0;
//         }
//     }
//     let horizontal_friction = -50.0;
//     player_state.velocity.x *= E.powf(delta_time * horizontal_friction);
//     player_state.velocity.y *= E.powf(delta_time * -0.1);
//     player_state.velocity.z *= E.powf(delta_time * horizontal_friction);
//     player_state.position += player_state.velocity * delta_time;
//     // player_state.velocity -= player_state.velocity * (10.0 * delta_time).min(1.0);
// }

enum SimulationInput<'a>{
    SingleInput(&'a Inputs),
    Buffer(&'a CircularBuffer<Inputs, INPUT_BUFFER_LEN>)
}
impl<'a> SimulationInput<'a>{
    fn get_input(&self, seconds_ago: f64) -> Inputs{
        match self{
            SimulationInput::SingleInput(inputs) => **inputs,
            SimulationInput::Buffer(buffer) => {
                buffer[(seconds_ago * PHYSICS_TICKRATE ) as usize]
            },
        }
    }
}

fn simulate_new_player_state(player_state: &mut PlayerState, inputs: SimulationInput, time_to_simulate: f64, extra_sim_time: &mut f64){
    let mut time_to_simulate_left = time_to_simulate;

    while *extra_sim_time > 1.0/PHYSICS_TICKRATE{
        time_to_simulate_left += 1.0/PHYSICS_TICKRATE;
        *extra_sim_time -= 1.0/PHYSICS_TICKRATE;
    }

    while time_to_simulate_left > 1.0 / PHYSICS_TICKRATE{
        time_to_simulate_left -= 1.0 / PHYSICS_TICKRATE;
        let inputs = inputs.get_input(time_to_simulate_left);
        let delta_time = 1.0/PHYSICS_TICKRATE;
        if player_state.position.y > 0.0 {
            player_state.velocity.y -= 9.8 * delta_time;
            player_state.is_jumping = false;
        } else {
            player_state.velocity.y = 0.0;
            let movement = inputs.camera_direction;
            let sideways = movement.cross(&directions::UP) * inputs.left_stick.x;
            let forward = movement * inputs.left_stick.y;

            let mut movement = sideways + forward;
            let movement_magnitude = inputs.left_stick.magnitude();
            movement.y = 0.0;

            movement = if movement.magnitude() > 0.01{movement.normalize() * movement_magnitude} else {Vector3::zeros()};
            player_state.velocity += 10.0 * delta_time * movement;
            if inputs.jump && !player_state.is_jumping{
                player_state.is_jumping = true;
                player_state.velocity.y = 10.0;
                player_state.position.y = 1.0;
            }
        }
        let horizontal_friction = -5.0;
        player_state.velocity.x *= E.powf(delta_time * horizontal_friction);
        player_state.velocity.y *= E.powf(delta_time * -0.1);
        player_state.velocity.z *= E.powf(delta_time * horizontal_friction);
        player_state.position += player_state.velocity * delta_time;
    }
    // dbg!(player_state);
    *extra_sim_time += time_to_simulate_left;
}


pub struct ServerGame {
    pub players: PlayerMap<PlayerObject>,
    pub current_tick: NetworkTick,
    extra_sim_time: f64,
}
impl ServerGame {
    pub fn new() -> Self {
        Self {
            players: PlayerMap::new(),
            current_tick: NetworkTick::new(0),
            extra_sim_time: 0.0,
        }
    }
    pub fn process(&mut self, delta_time: f64) {
        for player in &mut self.players {
            // dbg!(&player.inputs);
            simulate_new_player_state(&mut player.player_state, SimulationInput::SingleInput(&player.inputs), delta_time, &mut self.extra_sim_time);
            // println!("{:}", player.player_state.position);
            // dbg!(&player.player_state);
        }
        self.current_tick.val += 1;
    }
}
