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
    pub struct Game {
        pub inputs: Inputs,
        pub mouse_position: Vector2<f64>,
        pub last_mouse_position: Vector2<f64>,
        pub start_time: Instant
    }
    impl Game {
        pub fn new() -> Self {
            Self { 
                inputs: Inputs::new(),
                mouse_position: Vector2::zeros(),
                last_mouse_position: Vector2::zeros(),
                start_time: Instant::now()
            }
        }
        pub fn process(
            &mut self,
            delta_time: f64,
        ) {

        }
    }
}
