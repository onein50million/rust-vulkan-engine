/*
Some random ideas:

player has states, ie crouching, aiming, maybe a bitfield-like object?

compile time gltf parsing


Game object types:
Server object: calculated purely on server, client just copies data from server
Client predicted: Uses client side prediction to reduce latency
Client only (Animations, particles)
 */

pub mod directions {
    use nalgebra::Vector3;
    use std::f64::consts::FRAC_1_SQRT_2;

    pub const UP: Vector3<f64> = Vector3::new(0.0, 1.0, 0.0);
    pub const DOWN: Vector3<f64> = Vector3::new(0.0, -1.0, 0.0);
    pub const LEFT: Vector3<f64> = Vector3::new(-1.0, 0.0, 0.0);
    pub const RIGHT: Vector3<f64> = Vector3::new(1.0, 0.0, 0.0);
    pub const FORWARDS: Vector3<f64> = Vector3::new(0.0, 0.0, 1.0);
    pub const BACKWARDS: Vector3<f64> = Vector3::new(0.0, 0.0, -1.0);
}

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
pub struct NetworkBuffer<T, const L: usize>{
    buffer: Box<[T]>,
    start: usize,
}
impl<T,const L: usize> NetworkBuffer<T, L>{
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
impl<T,const L: usize> Index<GameTick> for NetworkBuffer<T, L>{
    type Output = T;

    fn index(&self, index: GameTick) -> &Self::Output {
        if index.val > L{
            eprintln!("Trying to access out of bounds in network buffer, clamping");
        }
        let index = index.val.min(L);
        &self.buffer[(self.start as i64 - index as i64).rem_euclid(L as i64 ) as usize]
    }
}
impl<T,const L: usize> IndexMut<GameTick> for NetworkBuffer<T, L>{
    fn index_mut(&mut self, index: GameTick) -> &mut Self::Output {
        if index.val > L{
            eprintln!("Trying to access out of bounds in network buffer, clamping");
        }
        let index = index.val.min(L);
        &mut self.buffer[(self.start as i64 - index as i64).rem_euclid(L as i64 ) as usize]
    }
}

pub struct ClientPlayer{
    pub game_object: GameObject,
    pub last_server_player_state: PlayerState,
    pub last_client_input_from_server: f64,
    pub input_buffer: Option<NetworkBuffer<Inputs, NETWORK_INPUT_BUFFER>>, //Only the local player has an input buffer

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

        }
    }
    pub fn update_gameobject(&mut self, client_start: &Instant){
        let mut new_state = self.last_server_player_state;
        let mut time_to_simulate = client_start.elapsed().as_secs_f64() - self.last_client_input_from_server;
        let mut ticks_behind = GameTick::new((time_to_simulate / (1.0 / FRAMERATE_TARGET)) as usize);
        while time_to_simulate > 1.0/FRAMERATE_TARGET{
            let input_at_time = match &self.input_buffer{
                Some(input_buffer) => input_buffer[ticks_behind],
                None => Inputs::new(),
            };

            simulate_new_player_state(&mut new_state, &input_at_time, 1.0/FRAMERATE_TARGET);
            ticks_behind.val -= 1;
            time_to_simulate -= 1.0/FRAMERATE_TARGET;
        }
        self.game_object.position = new_state.position;
        if let Some(input_buffer) = &self.input_buffer{
            self.game_object.rotation = UnitQuaternion::face_towards(&(input_buffer[GameTick::new(0)].camera_direction.component_mul(&Vector3::new(1.0,0.0,1.0))), &directions::UP)
        }
    }

} 
const NETWORK_INPUT_BUFFER: usize = (FRAMERATE_TARGET * (FRAMERATE_TARGET/NETWORK_TICKRATE) * 4.0) as usize;
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
    GameTick
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

fn simulate_new_player_state(player_state: &mut PlayerState, inputs: &Inputs, delta_time: f64){
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
        player_state.velocity += 4.0 * delta_time * movement;
        if inputs.jump && !player_state.is_jumping{
            player_state.is_jumping = true;
            player_state.velocity.y = 10.0;
            player_state.position.y = 1.0;
        }
    }
    player_state.position += player_state.velocity * delta_time;
    player_state.velocity -= player_state.velocity * (10.0 * delta_time).min(1.0);
}

pub struct ServerGame {
    pub players: PlayerMap<PlayerObject>,
    pub current_tick: GameTick,
}
impl ServerGame {
    pub fn new() -> Self {
        Self {
            players: PlayerMap::new(),
            current_tick: GameTick::new(0),
        }
    }
    pub fn process(&mut self, delta_time: f64) {
        for player in &mut self.players {
            simulate_new_player_state(&mut player.player_state, &player.inputs, delta_time);
            // dbg!(&player.player_state);
        }
        self.current_tick.val += 1;
    }
}
