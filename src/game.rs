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

    pub const UP: Vector3<f64> = Vector3::new(0.0, -1.0, 0.0);
    pub const DOWN: Vector3<f64> = Vector3::new(0.0, 1.0, 0.0);
    pub const LEFT: Vector3<f64> = Vector3::new(-1.0, 0.0, 0.0);
    pub const RIGHT: Vector3<f64> = Vector3::new(1.0, 0.0, 0.0);
    pub const FORWARDS: Vector3<f64> = Vector3::new(0.0, 0.0, 1.0);
    pub const BACKWARDS: Vector3<f64> = Vector3::new(0.0, 0.0, -1.0);

    pub const ISOMETRIC_DOWN: Vector3<f64> = Vector3::new(-FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2);
    pub const ISOMETRIC_UP: Vector3<f64> = Vector3::new(FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2);
    pub const ISOMETRIC_RIGHT: Vector3<f64> = Vector3::new(-FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2);
    pub const ISOMETRIC_LEFT: Vector3<f64> = Vector3::new(FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2);
}

pub(crate) mod server {
    use nalgebra::{UnitQuaternion, Vector3};
    use std::time::Instant;
    pub struct GameObject {
        pub position: Vector3<f64>,
        pub rotation: UnitQuaternion<f64>,
    }
    impl GameObject {
        pub fn new() -> Self {
            return Self {
                position: Vector3::new(0.0, 0.0, 0.0),
                rotation: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.0),
            };
        }
    }
    struct Game {
        game_start: Instant,
        last_frame_instant: Instant,
        planet: GameObject,
    }

    impl Game {
        pub fn new() -> Self {
            let game = Self {
                game_start: Instant::now(),
                last_frame_instant: Instant::now(),
                planet: GameObject::new(),
            };
            game
        }

        pub fn process(&mut self) {
            let _delta_time = self.last_frame_instant.elapsed().as_secs_f64();
            self.last_frame_instant = std::time::Instant::now();
        }
    }
}
pub mod client {

    use std::time::Instant;

    use crate::{support::Inputs, renderer::vulkan_data::VulkanData};
    use nalgebra::{Matrix4, Perspective3, Translation3, UnitQuaternion, Vector2, Vector3};

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

    pub struct GameObject {
        pub position: Vector3<f64>,
        pub rotation: UnitQuaternion<f64>,
        pub render_object_index: usize,
        pub animation_handler: Option<AnimationHandler>,
    }
    impl GameObject {
        pub fn get_transform(&self) -> Matrix4<f64> {
            return Matrix4::from(Translation3::from(self.position))
                * self.rotation.to_homogeneous();
        }
    }
    pub struct Player {
        movement: Vector3<f64>,
        position: Vector3<f64>,
        velocity: Vector3<f64>,
        rotation: UnitQuaternion<f64>,
        move_acceleration: f64,
        touching_ground: bool,
        pitch: f64,
        yaw: f64,
    }
    impl Default for Player {
        fn default() -> Self {
            Self {
                movement: Vector3::new(0.0, 0.0, 0.0),
                position: Vector3::new(2.0, -10.0, 1.0),
                velocity: Vector3::new(0.0, 0.0, 0.0),
                move_acceleration: Self::BASE_ACCEL,
                touching_ground: true,
                rotation: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.0),
                pitch: 0.0,
                yaw: 0.0
            }
        }
    }
    
    impl Player {
        const HEAD_OFFSET: Vector3<f64> = Vector3::new(0.0,-1.7,0.0);
        const FALL_ACCELERATION: f64 = 9.8;
        const JUMP_SPEED: f64 = 5.0;
        const BASE_ACCEL: f64 = 50.0;
    
        pub fn process(&mut self, delta_time: f64) {
    
            self.rotation =
                UnitQuaternion::from_axis_angle(&(-Vector3::y_axis()), self.yaw)*
                UnitQuaternion::from_axis_angle(&(Vector3::x_axis()), self.pitch);
    
            if self.touching_ground {
                self.velocity.x += self.movement.x * delta_time;
                self.velocity.z += self.movement.z * delta_time;
            }
    
            self.position += self.velocity * delta_time;
    
            if -self.position.y > 0.0 {
                self.velocity.y += Player::FALL_ACCELERATION * delta_time;
            } else {
                self.touching_ground = true;
                self.position.y = -0.0;
                self.velocity.x -= self.velocity.x * (10.0 * delta_time).clamp(-1.0, 1.0);
                self.velocity.z -= self.velocity.z * (10.0 * delta_time).clamp(-1.0, 1.0);
                self.velocity.y = 0.0;
            }
        }
        pub fn process_inputs(
            &mut self,
            inputs: &mut Inputs,
            _delta_time: f64,
            mouse_buffer: &Vector2<f64>,
        ) {
            self.pitch = (self.pitch + mouse_buffer.y/1000.0) % std::f64::consts::TAU;
            self.yaw = (self.yaw + mouse_buffer.x/1000.0) % std::f64::consts::TAU;
    
    
            self.movement = self.rotation
                * Vector3::new(
                    inputs.right - inputs.left,
                    0.0,
                    inputs.down - inputs.up,
                );
            self.movement.y = 0.0;
            if self.movement.magnitude() > 0.0 {
                self.movement = self.movement.normalize();
            }
            self.movement *= self.move_acceleration;
    
            if inputs.left_click && self.touching_ground {
                inputs.left_click = false;
                self.touching_ground = false;
                self.velocity.y = -Player::JUMP_SPEED;
            }
        }
        pub fn get_view_matrix(&self) -> Matrix4<f64>{
            (Translation3::from(self.position + Self::HEAD_OFFSET).to_homogeneous() * self.rotation.to_homogeneous()).try_inverse().unwrap()
        }
    }

    pub struct Game {
        pub inputs: Inputs,
        pub mouse_position: Vector2<f64>,
        pub mouse_buffer: Vector2<f64>,
        pub last_mouse_position: Vector2<f64>,
        pub start_time: Instant,
        pub player: Player,
    }
    impl Game {
        pub fn new() -> Self {
            Self { 
                inputs: Inputs::new(),
                mouse_position: Vector2::zeros(),
                last_mouse_position: Vector2::zeros(),
                start_time: Instant::now(),
                player: Player::default(),
                mouse_buffer: Vector2::zeros(),
            }
        }
        pub fn process(
            &mut self,
            delta_time: f64,
        ) {
            // dbg!(self.mouse_buffer);
            self.player.process_inputs(&mut self.inputs, delta_time, &self.mouse_buffer);
            self.mouse_buffer = Vector2::zeros();
            self.player.process(delta_time);
        }
    }
}
