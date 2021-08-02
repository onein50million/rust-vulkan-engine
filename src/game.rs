use cgmath::{Vector3, Quaternion, Matrix4, Transform, One, Vector2, Matrix3, Deg, InnerSpace, Point3, Zero, Euler, Rotation3, Rad, Angle};
use crate::renderer::*;
use std::time::Instant;
use winit::window::Window;
use std::f32::consts::PI;
use cgmath::num_traits::Pow;

trait Position{
    fn get_position(&self) -> Vector3<f32>;
    fn set_position(&mut self, new_position: Vector3<f32>);
}
trait Rotation{
    fn get_rotation(&self) -> Quaternion<f32>;
    fn set_rotation(&mut self, new_rotation: Quaternion<f32>);
}

pub(crate) struct GameObject{
    position:Vector3<f32>,
    rotation:Quaternion<f32>,
    render_object: RenderObject,
}
impl GameObject{
    fn process(&mut self, delta_time: f64){
        //Do stuff
    }
}

pub(crate)struct Inputs{
    pub(crate)forward: f32,
    pub(crate)backward: f32,
    pub(crate)left: f32,
    pub(crate)right: f32,
    pub(crate)up: f32,
    pub(crate)down: f32,
    pub(crate)sprint: f32,
}
impl Inputs{
    pub(crate) fn new() -> Self{
        return Inputs{
            forward: 0.0,
            backward: 0.0,
            left: 0.0,
            right: 0.0,
            up: 0.0,
            down: 0.0,
            sprint: 0.0
        }
    }
}


pub(crate) struct Camera {
    position:Vector3<f32>,
    rotation:Quaternion<f32>,
    target: Vector3<f32>,
    angle: Deg<f32>,
}
impl Camera{
    fn process(&mut self, delta_time:  f64){

        let radius = 5.0;
        self.angle += Deg(10.0*delta_time as f32);
        self.position = self.target + Vector3::new(self.angle.cos() * radius, -radius, self.angle.sin() * radius);


        let difference = self.target - self.position;
        let horizontal_distance = (difference.z.pow(2.0) + difference.x.pow(2.0)).sqrt();
        let pitch = Rad((difference.y).atan2(horizontal_distance) + 0.0*PI );
        let yaw = Rad(difference.x.atan2(difference.z) + 1.0*PI);

        let yaw_rotation = Quaternion::from_angle_y(yaw - Rad(0.0*PI));
        self.rotation = Quaternion::from_axis_angle(yaw_rotation * Vector3::new(1.0,0.0,0.0),pitch) * Quaternion::from_axis_angle(Vector3::new(0.0,1.0,0.0), yaw);
    }
    pub(crate) fn get_view_matrix_no_translation(&self) -> Matrix4<f32>{
        let matrix = Matrix4::from(self.rotation);
        return matrix.inverse_transform().unwrap();
    }

    pub(crate) fn get_view_matrix(&self) -> Matrix4<f32>{
        let matrix = Matrix4::from_translation(self.position) * Matrix4::from(self.rotation);
        return matrix.inverse_transform().unwrap();
    }



}

pub(crate)struct Player {
    pub(crate) model: Option<usize>,
    movement: Vector3<f32>,
    position: Vector3<f32>,
    velocity: Vector3<f32>,
    rotation: Quaternion<f32>,
    stamina: f32,
    move_accel: f32,
    touching_ground: bool,
    sprinting: bool,
}

impl Position for Player {
    fn get_position(&self) -> Vector3<f32> {
        return self.position;
    }
    fn set_position(&mut self, new_position: Vector3<f32>) {
        self.position = new_position;
    }
}


impl Default for Player{
    fn default() -> Self {
        Player{
            model: None,
            movement: Vector3::new(0.0, 0.0, 0.0),
            position: Vector3::new(2.0, -10.0, 1.0),
            velocity: Vector3::new(0.0,0.0,0.0),
            move_accel: Self::BASE_ACCEL,stamina: 1.0,
            touching_ground: true, sprinting: false,
            rotation: Quaternion::from_angle_y(Rad(0.0)),
        }

    }
}

impl Player{

    const HEIGHT: f32 = 1.7;
    const FALL_ACCELERATION: f32 = 9.8;
    const JUMP_SPEED: f32 = 5.0;
    const STAMINA_REGEN: f32 = 0.1;
    const BASE_ACCEL: f32 = 50.0;
    const SPRINT_MULTIPLIER: f32 = 2.0;

    pub(crate) fn process(&mut self, delta_time: f32){
        self.stamina += Player::STAMINA_REGEN * delta_time;
        if self.touching_ground{
            self.velocity.x += self.movement.x*delta_time;
            self.velocity.z += self.movement.z*delta_time;
        }

        self.position += self.velocity * delta_time;

        if -self.position.y > 0.0{
            self.velocity.y += Player::FALL_ACCELERATION*delta_time;
        }else{
            self.touching_ground = true;
            self.position.y = -0.0;
            self.velocity.x -= self.velocity.x*(10.0*delta_time).clamp(-1.0,1.0);
            self.velocity.z -= self.velocity.z*(10.0*delta_time).clamp(-1.0, 1.0);
            self.velocity.y = 0.0;

        }
        if self.stamina < 0.01{
            self.sprinting = false;
        }
        if self.sprinting{
            self.move_accel = Self::SPRINT_MULTIPLIER * Self::BASE_ACCEL;
        }else{
            self.move_accel = Self::BASE_ACCEL;
        }
        if self.sprinting && self.movement.magnitude() > 0.0{
            self.stamina -= Self::STAMINA_REGEN * 2.0 * delta_time;
        } else{
            self.stamina += Self::STAMINA_REGEN * delta_time;
        }
        self.stamina = self.stamina.min(1.0);

    }
    pub(crate) fn process_inputs(&mut self, inputs: &Inputs, delta_time:f64, relative_direction: Quaternion<f32>){
        self.movement = relative_direction * Vector3::new(
            inputs.right - inputs.left,
            0.0,
            inputs.backward - inputs.forward,
        );
        self.movement.y = 0.0;
        if self.movement.magnitude() > 0.0{
            self.movement = self.movement.normalize();
            self.rotation = Quaternion::from_angle_y(Rad(self.movement.x.atan2(self.movement.z) + PI/1.0));
        }
        self.movement *= self.move_accel;

        if inputs.up > 0.5 && self.touching_ground {
            self.touching_ground = false;
            self.velocity.y = -Player::JUMP_SPEED;
        }
        if self.stamina > 0.25 && inputs.sprint > 0.5{
            self.sprinting = true;
        }
        if inputs.sprint < 0.5{
            self.sprinting = false;
        }
    }
}



pub(crate) struct Game {
    pub(crate) camera: Camera,
    objects: Vec<GameObject>,
    pub(crate) player: Player,
    pub(crate) mouse_buffer: Vector2<f64>,
    pub(crate) inputs: Inputs,
    pub(crate) focused: bool,
    last_frame_instant: Instant,
    pub(crate) vulkan_data: VulkanData,
}

impl Game{
    pub(crate) fn new(window: &Window) -> Self{
        let mut player = Player{
            ..Default::default()
        };
        let camera = Camera{
            position: Vector3::new(0.0,-5.0,0.0),
            rotation: Quaternion::one(),
            target: player.position,
            angle: Deg(0.0)
        };

        let mut vulkan_data = VulkanData::new();
        vulkan_data.init_vulkan(&window);
        player.model = Some(vulkan_data.player_object_index);

        let game = Game{
            camera,
            objects: vec![],
            player,
            mouse_buffer: Vector2::new(0.0,0.0),
            inputs: Inputs::new(),
            focused: false,
            last_frame_instant: Instant::now(),
            vulkan_data
        };
        return game;
    }
    pub(crate) fn process(&mut self){
        let delta_time = self.last_frame_instant.elapsed().as_secs_f64();
        self.last_frame_instant = std::time::Instant::now();

        self.player.process_inputs(&self.inputs, delta_time, self.camera.rotation);
        self.player.process(delta_time as f32);

        self.camera.target = self.player.position - Vector3::new(0.0,Player::HEIGHT, 0.0);
        self.camera.process( delta_time);


        for i in 0..self.objects.len(){
            self.objects[i].process(delta_time);
        }
        self.vulkan_data.process(&self.camera);
        self.vulkan_data.uniform_buffer_object.model[self.player.model.unwrap()] = Matrix4::from_translation(self.player.position) * Matrix4::from(self.player.rotation);
        self.vulkan_data.draw_frame();
    }
}