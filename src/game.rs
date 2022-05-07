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

    pub const ISOMETRIC_DOWN: Vector3<f64> =
        Vector3::new(-FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2);
    pub const ISOMETRIC_UP: Vector3<f64> = Vector3::new(FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2);
    pub const ISOMETRIC_RIGHT: Vector3<f64> =
        Vector3::new(-FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2);
    pub const ISOMETRIC_LEFT: Vector3<f64> =
        Vector3::new(FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2);
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
    use std::f64::consts::PI;
    use std::time::Instant;

    use crate::renderer::VulkanData;
    use crate::support::Inputs;
    use crate::world::World;
    use float_ord::FloatOrd;
    use nalgebra::{Matrix4, Translation3, UnitQuaternion, Vector2, Vector3, Vector4, Perspective3, Point3};

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

    pub struct Camera {
        latitude: f64,
        longitude: f64,
    }
    impl Camera {
        fn new() -> Self {
            Self {
                latitude: 0.0,
                longitude: 0.0,
            }
        }
        fn get_rotation(&self) -> UnitQuaternion<f64> {
            return UnitQuaternion::face_towards(
                &(self.get_position()),
                &Vector3::new(0.0, -1.0, 0.0),
            );
        }
        fn get_position(&self) -> Vector3<f64> {
            return UnitQuaternion::from_euler_angles(0.0, self.longitude, 0.0)
                * UnitQuaternion::from_euler_angles(0.0, 0.0, self.latitude)
                * Vector3::new(10_000_000.0, 0.0, 0.0);
        }
        pub fn get_view_matrix(&self) -> Matrix4<f64> {
            (Matrix4::from(Translation3::from(self.get_position()))
                * self.get_rotation().to_homogeneous())
            .try_inverse()
            .unwrap()
        }
    }

    pub struct GameObject {
        pub position: Vector3<f64>,
        pub rotation: UnitQuaternion<f64>,
        pub render_object_index: usize,
        pub animation_handler: Option<AnimationHandler>,
    }

    pub struct Game {
        pub world: World,
        pub inputs: Inputs,
        pub mouse_position: Vector2<f64>,
        pub last_mouse_position: Vector2<f64>,
        pub planet: GameObject,
        pub start_time: Instant,
        pub camera: Camera,
        pub selected_province: Option<usize>,
    }
    impl Game {
        pub fn new(planet_render_index: usize, world: World) -> Self {
            Self {
                inputs: Inputs::new(),
                mouse_position: Vector2::zeros(),
                planet: GameObject {
                    position: Vector3::zeros(),
                    rotation: UnitQuaternion::identity(),
                    render_object_index: planet_render_index,
                    animation_handler: None,
                },
                start_time: Instant::now(),
                camera: Camera::new(),
                last_mouse_position: Vector2::zeros(),
                world,
                selected_province: None,
            }
        }
        pub fn process(&mut self, delta_time: f64, projection: &Perspective3<f64>) {
            self.world.process(delta_time);

            let delta_mouse = self.last_mouse_position - self.mouse_position;
            self.last_mouse_position = self.mouse_position;
            if self.inputs.panning {
                self.camera.latitude =
                    (self.camera.latitude + 1.0 * delta_mouse.y).clamp(-PI / 2.01, PI / 2.01);
                self.camera.longitude += delta_mouse.x * 1.0;
            }

            if self.inputs.left_click{
                self.selected_province = {
                    //TODO: Figure out why I have to make these negative. Probably something to do with the inconsistent coordinate system
                    let direction = self.camera.get_view_matrix().try_inverse().unwrap().transform_vector(&projection.unproject_point(&Point3::new(-self.mouse_position.x, -self.mouse_position.y, 1.0)).coords);
                    
                    match World::intersect_planet(self.camera.get_position(), -direction.xyz()){
                        Some(point) => {
                            Some(self.world.provinces.iter().enumerate().min_by_key(|(_, province)|{FloatOrd((point - province.position).magnitude())}).expect("Failed to find closest provice to click").0)
                        }
                        None => None,
                    }
                };
                self.inputs.left_click = false;
            }
        }
    }
}
