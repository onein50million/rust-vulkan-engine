use crate::renderer::*;
use nalgebra::{Matrix4, Translation3, UnitQuaternion, Vector2, Vector3};
use std::time::Instant;
use winit::window::Window;

trait GlobalPosition {
    fn get_global_position(&self, game: &Game) -> Vector3<f64>;
}
trait GlobalRotation {
    fn get_global_rotation(&self, game: &Game) -> UnitQuaternion<f64>;
}

pub(crate) struct GameObject {
    position: Vector3<f64>,
    rotation: UnitQuaternion<f64>,
    render_object_index: usize,
}
impl GameObject {
    fn new(render_object_index: usize) -> Self{
        return Self{
            position: Vector3::new(0.0,0.0,0.0,),
            rotation: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.0),
            render_object_index
        }
    }
    fn process(&mut self, delta_time: f64) {
        //Do stuff
    }
}

pub(crate) struct Inputs {
    pub(crate) forward: f64,
    pub(crate) backward: f64,
    pub(crate) left: f64,
    pub(crate) right: f64,
    pub(crate) camera_y: f64,
    pub(crate) camera_x: f64,
    pub(crate) up: f64,
    pub(crate) down: f64,
    pub(crate) sprint: f64,
}
impl Inputs {
    pub(crate) fn new() -> Self {
        return Inputs {
            forward: 0.0,
            backward: 0.0,
            left: 0.0,
            right: 0.0,
            camera_y: 0.0,
            camera_x: 0.0,
            up: 0.0,
            down: 0.0,
            sprint: 0.0,
        };
    }
}



pub(crate) struct Player {
    pub(crate) model: Option<usize>,
    movement: Vector3<f64>,
    position: Vector3<f64>,
    velocity: Vector3<f64>,
    rotation: UnitQuaternion<f64>,
    stamina: f64,
    move_acceleration: f64,
    touching_ground: bool,
    sprinting: bool,
    ship_index: Option<usize>,
    pitch: f64,
    yaw: f64,
}

impl GlobalPosition for Player {
    fn get_global_position(&self, game: &Game) -> Vector3<f64> {
        return match self.ship_index{
            None => {self.position}
            Some(index) => {
                self.position + game.ships[index].position
            }
        }
    }
}

impl GlobalRotation for Player {
    fn get_global_rotation(&self, game: &Game) -> UnitQuaternion<f64> {
        return match self.ship_index{
            None => {UnitQuaternion::identity()}
            Some(index) => {
                game.ships[index].rotation
            }
        }* UnitQuaternion::from_axis_angle(&(-Vector3::y_axis()),self.yaw)
    }
}

impl Default for Player {
    fn default() -> Self {
        Self {
            model: None,
            movement: Vector3::new(0.0, 0.0, 0.0),
            position: Vector3::new(2.0, -10.0, 1.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            move_acceleration: Self::BASE_ACCEL,
            stamina: 1.0,
            touching_ground: true,
            sprinting: false,
            rotation: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.0),
            ship_index: None,
            pitch: 0.0,
            yaw: 0.0
        }
    }
}

impl Player {
    const HEAD_OFFSET: Vector3<f64> = Vector3::new(0.0,-1.7,0.0);
    const FALL_ACCELERATION: f64 = 9.8;
    const JUMP_SPEED: f64 = 5.0;
    const STAMINA_REGEN: f64 = 0.1;
    const BASE_ACCEL: f64 = 50.0;
    const SPRINT_MULTIPLIER: f64 = 2.0;

    pub(crate) fn process(&mut self, delta_time: f64) {

        self.rotation =
            UnitQuaternion::from_axis_angle(&(-Vector3::y_axis()), self.yaw)*
            UnitQuaternion::from_axis_angle(&(Vector3::x_axis()), self.pitch);

        self.stamina += Player::STAMINA_REGEN * delta_time;
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
        if self.stamina < 0.01 {
            self.sprinting = false;
        }
        if self.sprinting {
            self.move_acceleration = Self::SPRINT_MULTIPLIER * Self::BASE_ACCEL;
        } else {
            self.move_acceleration = Self::BASE_ACCEL;
        }
        if self.sprinting && self.movement.magnitude() > 0.0 {
            self.stamina -= Self::STAMINA_REGEN * 2.0 * delta_time;
        } else {
            self.stamina += Self::STAMINA_REGEN * delta_time;
        }
        self.stamina = self.stamina.min(1.0);
    }
    pub(crate) fn process_inputs(
        &mut self,
        inputs: &Inputs,
        delta_time: f64,
        mouse_buffer: &Vector2<f64>,
    ) {
        self.pitch = (self.pitch + mouse_buffer.y/1000.0) % std::f64::consts::TAU;
        self.yaw = (self.yaw + mouse_buffer.x/1000.0) % std::f64::consts::TAU;


        self.movement = self.rotation
            * Vector3::new(
                inputs.right - inputs.left,
                0.0,
                inputs.backward - inputs.forward,
            );
        self.movement.y = 0.0;
        if self.movement.magnitude() > 0.0 {
            self.movement = self.movement.normalize();
        }
        self.movement *= self.move_acceleration;

        if inputs.up > 0.5 && self.touching_ground {
            self.touching_ground = false;
            self.velocity.y = -Player::JUMP_SPEED;
        }
        if self.stamina > 0.25 && inputs.sprint > 0.5 {
            self.sprinting = true;
        }
        if inputs.sprint < 0.5 {
            self.sprinting = false;
        }
    }

    pub(crate) fn get_view_matrix_no_translation(&self) -> Matrix4<f32> {
        let matrix = Matrix4::from(self.rotation).cast();
        return matrix.try_inverse().unwrap();
    }

    pub(crate) fn get_view_matrix(&self) -> Matrix4<f32> {
        let matrix =
            Translation3::from(self.position + Self::HEAD_OFFSET).to_homogeneous() * Matrix4::from(self.rotation);
        return matrix.try_inverse().unwrap().cast();
    }


}

struct Ship{
    position: Vector3<f64>,
    rotation: UnitQuaternion<f64>,
}

pub(crate) struct Game {
    objects: Vec<GameObject>,
    pub(crate) player: Player,
    pub(crate) mouse_buffer: Vector2<f64>,
    pub(crate) inputs: Inputs,
    pub(crate) focused: bool,
    last_frame_instant: Instant,
    pub(crate) vulkan_data: VulkanData,
    ships: Vec<Ship>,
}

impl Game {
    pub(crate) fn new(window: &Window) -> Self {
        let mut objects = vec![];
        let mut player = Player {
            ..Default::default()
        };

        let mut vulkan_data = VulkanData::new();

        vulkan_data.init_vulkan(&window);


        objects.push(
            GameObject::new(
                vulkan_data.load_obj_model(
                    "models/ship/ship.obj".parse().unwrap(),
                    "models/ship/texture.png".parse().unwrap())));
        objects.push(
            GameObject::new(
                vulkan_data.load_obj_model(
                    "models/planet/planet.obj".parse().unwrap(),
                    "models/planet/texture.png".parse().unwrap())));


        vulkan_data.update_vertex_buffer();

        player.model = Some(vulkan_data.player_object_index);

        let game = Game {
            objects: vec![],
            player,
            mouse_buffer: Vector2::new(0.0, 0.0),
            inputs: Inputs::new(),
            focused: false,
            last_frame_instant: Instant::now(),
            vulkan_data,
            ships: vec![]
        };
        return game;
    }
    pub(crate) fn process(&mut self) {
        let delta_time = self.last_frame_instant.elapsed().as_secs_f64();
        self.last_frame_instant = std::time::Instant::now();

        self.player
            .process_inputs(&self.inputs, delta_time,&self.mouse_buffer);
        self.mouse_buffer = Vector2::zeros();
        self.player.process(delta_time);

        for i in 0..self.objects.len() {
            self.objects[i].process(delta_time);
        }
        self.vulkan_data.objects[self.player.model.unwrap()].model =
            (Matrix4::from(Translation3::from(self.player.get_global_position(self)))
                * Matrix4::from(self.player.get_global_rotation(self))).cast();
        self.vulkan_data.uniform_buffer_object.mouse_position = Vector2::new(
            (self.mouse_buffer.x
                / self
                    .vulkan_data
                    .surface_capabilities
                    .unwrap()
                    .current_extent
                    .width as f64) as f32,
            (self.mouse_buffer.y
                / self
                    .vulkan_data
                    .surface_capabilities
                    .unwrap()
                    .current_extent
                    .height as f64) as f32,
        );
        self.vulkan_data.process(&self.player);
        self.vulkan_data.draw_frame();
    }
}
