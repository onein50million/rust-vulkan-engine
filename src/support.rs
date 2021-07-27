use cgmath::{Vector3, Vector2, Matrix4, Transform, Deg, Matrix3, InnerSpace, Vector4};
use ash::vk;
use crate::NUM_RANDOM;

pub(crate) const NUM_MODELS: usize = 100;

#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) struct Vertex {
    pub(crate) position: Vector3<f32>,
    pub(crate) color: Vector3<f32>,
    pub(crate) texture_coordinate: Vector2<f32>,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub(crate)struct UniformBufferObject {
    pub(crate)model: [Matrix4<f32>; NUM_MODELS],
    pub(crate)view: [Matrix4<f32>; NUM_MODELS],
    pub(crate)proj: [Matrix4<f32>; NUM_MODELS],
    pub(crate)random: [Vector4<f32>; NUM_RANDOM], //std140 packing so it needs to be 16 bytes wide

}

#[repr(C)]
pub(crate)struct PushConstants {
    pub(crate)uniform_index: u32,
    pub(crate)texture_index: u32,
}

pub(crate)struct Player {
    movement: Vector3<f64>,
    position: Vector3<f64>,
    velocity: Vector3<f64>,
    yaw: f64,
    pitch: f64,
    stamina: f64,
    move_accel: f64,
    touching_ground: bool,
    sprinting: bool,
}

impl Default for Player{
    fn default() -> Self {
        Player{
            movement: Vector3::new(0.0,0.0,0.0),
            position: Vector3::new(2.0, -10.0, 1.0),
            velocity: Vector3::new(0.0,0.0,0.0),
            yaw: 0.0, pitch: 0.0,
            move_accel: Self::BASE_ACCEL,stamina: 1.0,
            touching_ground: true, sprinting: false,

        }

    }
}

impl Player{

    const HEIGHT: f64 = 1.7;
    const FALL_ACCELERATION: f64 = 9.8;
    const JUMP_SPEED: f64 = 5.0;
    const STAMINA_REGEN: f64 = 0.1;
    const BASE_ACCEL: f64 = 50.0;
    const SPRINT_MULTIPLIER: f64 = 2.0;

    pub(crate) fn get_view_matrix_no_translation(&self) -> Matrix4<f32>{
        let matrix = Matrix4::<f32>::from_angle_y(Deg(-self.yaw as f32))
            * Matrix4::<f32>::from_angle_x(Deg(self.pitch as f32));

        return matrix.inverse_transform().unwrap();
    }

    pub(crate) fn get_view_matrix(&self) -> Matrix4<f32>{
        let matrix = Matrix4::<f32>::from_translation(Vector3::new(self.position.x as f32, self.position.y as f32, self.position.z as f32))
            * Matrix4::<f32>::from_angle_y(Deg(-self.yaw as f32))
            * Matrix4::<f32>::from_angle_x(Deg(self.pitch as f32));

        return matrix.inverse_transform().unwrap();
    }
    pub(crate) fn process(&mut self, delta_time: f64){
        self.stamina += Player::STAMINA_REGEN * delta_time;
        if self.touching_ground{
            self.velocity.x += self.movement.x*delta_time;
            self.velocity.z += self.movement.z*delta_time;
        }

        self.position += self.velocity * delta_time;

        if -self.position.y > Player::HEIGHT{
            self.velocity.y += Player::FALL_ACCELERATION*delta_time;
        }else{
            self.touching_ground = true;
            self.position.y = -Player::HEIGHT;
            self.velocity.x -= 10.0*self.velocity.x*delta_time;
            self.velocity.z -= 10.0*self.velocity.z*delta_time;
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
    pub(crate) fn process_inputs(&mut self, inputs: &Inputs, delta_time:f64, mouse_buffer:Vector2<f64>){
        self.yaw += mouse_buffer.x*0.1;
        self.pitch = (self.pitch +  mouse_buffer.y*0.1).clamp(-80.0,80.0);


        self.movement = (Matrix3::from_angle_y(Deg(-self.yaw))) * Vector3::new(
            inputs.right - inputs.left,
            0.0,
            inputs.backward - inputs.forward,
        ) * self.move_accel;

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


impl Vertex {
    //noinspection RsSelfConvention
    pub(crate) fn get_binding_description() -> vk::VertexInputBindingDescription {
        return vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build();
    }
    pub(crate) fn get_attribute_description() -> Vec<vk::VertexInputAttributeDescription> {
        let attribute_descriptions = vec![
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(12), //might be off, could be fun to see what happens when it's off
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(24),

        ];

        return attribute_descriptions
            .into_iter()
            .map(|attribute_description| attribute_description.build())
            .collect();
    }
}

pub(crate)struct Inputs{
    pub(crate)forward: f64,
    pub(crate)backward: f64,
    pub(crate)left: f64,
    pub(crate)right: f64,
    pub(crate)up: f64,
    pub(crate)down: f64,
    pub(crate)sprint: f64,
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