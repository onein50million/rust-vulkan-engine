use crate::renderer::*;
use nalgebra::{Matrix4, Rotation3, Translation3, UnitQuaternion, Vector2, Vector3, zero};
use std::time::Instant;
use winit::window::Window;

// trait GlobalPosition {
//     fn get_global_position(&self, game: &Game) -> Vector3<f64>;
// }
// trait GlobalRotation {
//     fn get_global_rotation(&self, game: &Game) -> UnitQuaternion<f64>;
// }
//
// trait UpdateRenderer {
//     fn update_renderer(&self, renderer: VulkanData);
// }
//
// trait HasModelIndex {
//     fn get_model_index(&self) -> usize;
// }

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
    fn process(&mut self, _delta_time: f64) {
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
        _delta_time: f64,
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
}

struct AngularVelocity{
    axis: Vector3<f64>,
    magnitude: f64,
}
impl AngularVelocity{
    fn new(x: f64, y: f64, z: f64, magnitude: f64) -> Self{
        Self{
            axis: Vector3::new(x,y,z),
            magnitude
        }
    }
}

struct Ship{
    velocity: Vector3<f64>,
    angular_velocity: AngularVelocity,
    game_object: GameObject
}

impl Ship{
    fn new(render_object_index: usize) -> Self{
        Self{
            velocity:zero(),
            angular_velocity: AngularVelocity::new(0.0,0.0,0.0,0.0),
            game_object: GameObject::new(render_object_index)
        }
    }
    fn process(&mut self, delta_time: f64){
        self.game_object.rotation *= UnitQuaternion::from_euler_angles(1.0*delta_time,0.3*delta_time,0.4*delta_time)
    }
    fn ship_space_transform(&self) -> Matrix4<f64>{
        return Matrix4::from(self.game_object.rotation).append_translation(&self.game_object.position);
    }
    fn ship_space_transform_no_translation(&self) -> Matrix4<f64>{
        return Matrix4::from(self.game_object.rotation);
    }
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
    viewmodel_index: usize,
}

impl Game {
    pub(crate) fn new(window: &Window) -> Self {
        let mut objects = vec![];
        let mut player = Player {
            ..Default::default()
        };

        let mut vulkan_data = VulkanData::new();

        vulkan_data.init_vulkan(&window);


        let mut ships = vec![];
        player.ship_index = Some(ships.len());
        ships.push(
            Ship::new(
                vulkan_data.load_obj_model(
                    "models/ship/ship.obj".parse().unwrap(),
                    "models/ship/texture.png".parse().unwrap())));

        let planet_index = objects.len();
        objects.push(
            GameObject::new(
                vulkan_data.load_obj_model(
                    "models/planet/planet.obj".parse().unwrap(),
                    "models/planet/texture.png".parse().unwrap())));

        objects[planet_index].position = Vector3::new(-10.0,0.0,0.0);

        let viewmodel_index = objects.len();
        objects.push(
            GameObject::new(
                vulkan_data.load_obj_model(
                    "models/hands/hands.obj".parse().unwrap(),
                    "models/hands/texture.png".parse().unwrap())));

        vulkan_data.objects[objects[viewmodel_index].render_object_index].is_viewmodel = true;
        objects[viewmodel_index].position = Vector3::new(0.0,0.0,0.5);


        vulkan_data.update_vertex_buffer();

        player.model = None;

        let game = Game {
            objects,
            player,
            mouse_buffer: Vector2::new(0.0, 0.0),
            inputs: Inputs::new(),
            focused: false,
            last_frame_instant: Instant::now(),
            vulkan_data,
            ships,
            viewmodel_index,
        };
        return game;
    }
    pub(crate) fn process(&mut self) {
        let delta_time = self.last_frame_instant.elapsed().as_secs_f64();
        self.last_frame_instant = std::time::Instant::now();

        for ship_index in 0..self.ships.len(){
            self.ships[ship_index].process(delta_time);
        }
        self.player
            .process_inputs(&self.inputs, delta_time,&self.mouse_buffer);
        self.mouse_buffer = Vector2::zeros();
        self.player.process(delta_time);

        // self.objects[self.viewmodel_index].rotation *= UnitQuaternion::from_euler_angles(0.0,1.0*delta_time,0.0);

        for i in 0..self.objects.len() {
            self.objects[i].process(delta_time);
        }

        self.update_renderer();
        self.vulkan_data.transfer_data_to_gpu();
        self.vulkan_data.draw_frame();
    }

    fn update_renderer(&mut self){
        let projection = self.vulkan_data.get_projection_matrix();

        let ship_transform = match self.player.ship_index{
            None => {Matrix4::identity()}
            Some(ship_index) => {
                self.ships[ship_index].ship_space_transform()
            }
        };

        let ship_transform_no_translation = match self.player.ship_index{
            None => {Matrix4::identity()}
            Some(ship_index) => {
                self.ships[ship_index].ship_space_transform_no_translation()
            }
        };
        let view_matrix = (ship_transform * Translation3::from(self.player.position + Player::HEAD_OFFSET).to_homogeneous()
            * Matrix4::from(self.player.rotation)).try_inverse().unwrap().cast();
        let view_matrix_no_translation = (ship_transform_no_translation * Matrix4::from(self.player.rotation)).try_inverse().unwrap().cast();




        for ship_index in 0..self.ships.len(){
            self.vulkan_data.objects[self.ships[ship_index].game_object.render_object_index].model =
                (Matrix4::from(Translation3::from(self.ships[ship_index].game_object.position))
                    * Matrix4::from(self.ships[ship_index].game_object.rotation)).cast();

            self.vulkan_data.objects[self.ships[ship_index].game_object.render_object_index].view = view_matrix;
            self.vulkan_data.objects[self.ships[ship_index].game_object.render_object_index].proj = projection;
        }

        self.vulkan_data.cubemap.as_mut().unwrap().process(view_matrix_no_translation, projection);



        match self.player.model{
            None => {}
            Some(model_index) => {
                self.vulkan_data.objects[self.player.model.unwrap()].model = (ship_transform*
                    Matrix4::from(Translation3::from(self.player.position))
                    * Rotation3::from_axis_angle(&(-Vector3::y_axis()), self.player.yaw).to_homogeneous()).cast();

                self.vulkan_data.objects[self.player.model.unwrap()].view = view_matrix;
                self.vulkan_data.objects[self.player.model.unwrap()].proj = projection;

            }
        }


        for i in 0..self.objects.len() {
            self.vulkan_data.objects[self.objects[i].render_object_index].model =
                (Matrix4::from(Translation3::from(self.objects[i].position))
                    * Matrix4::from(Rotation3::from(self.objects[i].rotation))).cast();

            self.vulkan_data.objects[self.objects[i].render_object_index].view = if i == self.viewmodel_index{
                Matrix4::identity()
            }else{
                view_matrix
            };
            self.vulkan_data.objects[self.objects[i].render_object_index].proj = projection;
        }

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
    }
}
