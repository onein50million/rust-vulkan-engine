use erupt::vk;
use nalgebra::{Matrix4, Translation3, Vector2, Vector3, Vector4};
use serde::{Deserialize, Serialize};

pub const NETWORK_TICK_RATE: f64 = 10.0;
pub const FRAMERATE_TARGET: f64 = 280.0;
pub const NUM_RANDOM: usize = 100;
pub const FRAME_SAMPLES: usize = 100;
pub const NUM_MODELS: usize = 1000;
pub const NUM_LIGHTS: usize = 1;

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct Inputs {
    pub up: f64,
    pub down: f64,
    pub left: f64,
    pub right: f64,
    pub map_mode: u8,
    pub zoom: f64,
    pub exposure: f64,
    pub angle: f64,
    pub panning: bool,
    pub left_click: bool,
}
impl Inputs {
    pub fn new() -> Self {
        return Inputs {
            left: 0.0,
            right: 0.0,
            up: 0.0,
            down: 0.0,
            map_mode: 0,
            zoom: 1.0,
            exposure: 1.0,
            angle: 0.0,
            panning: false,
            left_click: false,
        };
    }
}

pub(crate) mod flags {
    pub(crate) const IS_CUBEMAP: u32 = 0b1;
    pub(crate) const IS_HIGHLIGHTED: u32 = 0b10;
    pub(crate) const IS_VIEWMODEL: u32 = 0b100;
    pub(crate) const IS_GLOBE: u32 = 0b1000;
    pub(crate) const IS_VIEW_PROJ_MATRIX_IGNORED: u32 = 0b10000;
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Vertex {
    pub position: Vector3<f32>,
    pub normal: Vector3<f32>,
    pub tangent: Vector4<f32>,
    pub texture_coordinate: Vector2<f32>,
    pub texture_type: u32, //Texture index for multiple textures
    pub bone_indices: Vector4<u32>,
    pub bone_weights: Vector4<f32>,
    pub elevation: f32,
}

impl Vertex {
    pub(crate) const fn new(
        position: Vector3<f32>,
        normal: Vector3<f32>,
        texture_coordinate: Vector2<f32>,
    ) -> Self {
        return Self {
            position,
            normal,
            tangent: Vector4::new(0.0, 0.0, 0.0, 0.0),
            texture_coordinate,
            texture_type: 0,
            bone_indices: Vector4::new(0, 0, 0, 0),
            bone_weights: Vector4::new(0.0, 0.0, 0.0, 0.0),
            elevation: 0.0,
        };
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) struct Light {
    pub(crate) position: Vector4<f32>,
    pub(crate) color: Vector4<f32>,
}
impl Light {
    pub(crate) fn new() -> Self {
        let _rng = fastrand::Rng::new();
        let _distance = 5.0;
        let power = 1.0;
        let position = Vector4::new(0.0, 0.0, 0.0, 0.0);
        let color = Vector4::new(1.0, 0.9, 0.9, 0.0) * power;

        println!("Position: {:}", position);
        Self { position, color }
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Bone {
    pub matrix: Matrix4<f32>,
}
impl Bone {
    pub(crate) fn new() -> Self {
        return Self {
            matrix: Matrix4::from(Translation3::new(0.0, 0.0, 0.0)),
        };
    }
}

pub(crate) const NUM_BONES_PER_BONESET: usize = 256;
pub(crate) const NUM_BONE_SETS: usize = 256;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct BoneSet {
    //One frame
    pub(crate) input_tangent: f32,
    pub(crate) output_tangent: f32,
    pub(crate) _padding1: f32,
    pub(crate) _padding2: f32,
    pub bones: [Bone; NUM_BONES_PER_BONESET],
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct ShaderStorageBufferObject {
    pub bone_sets: [BoneSet; NUM_BONE_SETS],
}
impl ShaderStorageBufferObject {
    pub(crate) fn new_boxed() -> Box<Self> {
        //need to do unsafe things because we can't create a array on the heap (afaik)
        let layout = std::alloc::Layout::new::<[BoneSet; NUM_BONE_SETS]>();
        unsafe {
            let pointer = std::alloc::alloc(layout) as *mut BoneSet;
            for i in 0..(NUM_BONE_SETS) {
                pointer.offset(i as isize).write(BoneSet {
                    input_tangent: 1.0,
                    output_tangent: 1.0,
                    _padding1: 0.0,
                    _padding2: 0.0,
                    bones: [Bone::new(); NUM_BONES_PER_BONESET],
                });
            }
            Box::new(
                std::ptr::NonNull::new_unchecked(pointer as *mut Self)
                    .as_ref()
                    .to_owned(),
            )
        }
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct UniformBufferObject {
    pub view: Matrix4<f32>,
    pub proj: Matrix4<f32>,
    pub(crate) random: [Vector4<f32>; NUM_RANDOM], //std140 packing so it needs to be 16 bytes wide
    pub(crate) lights: [Light; NUM_LIGHTS],        //std140 packing so it needs to be 16 bytes wide
    pub(crate) player_index: u32,
    pub(crate) num_lights: u32,
    pub(crate) _map_mode: u32,
    pub exposure: f32,
    pub mouse_position: Vector2<f32>,
    pub(crate) screen_size: Vector2<f32>,
    pub time: f32,
    pub player_position: Vector3<f32>,
}
#[derive(Debug)]
#[repr(C)]
pub(crate) struct PushConstants {
    pub(crate) model: Matrix4<f32>,
    pub(crate) texture_index: u32,
    pub(crate) bitfield: u32,
    pub(crate) animation_frames: u32,
}

impl Vertex {
    pub(crate) fn get_binding_description() -> vk::VertexInputBindingDescriptionBuilder<'static> {
        return vk::VertexInputBindingDescriptionBuilder::new()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);
    }
    pub(crate) fn get_attribute_description(
    ) -> Vec<vk::VertexInputAttributeDescriptionBuilder<'static>> {
        let attribute_descriptions = vec![
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(12), //might be off, could be fun to see what happens when it's off
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(24),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(3)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(40),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(4)
                .format(vk::Format::R32_UINT)
                .offset(48),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(5)
                .format(vk::Format::R32G32B32A32_UINT)
                .offset(52),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(6)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(68),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(7)
                .format(vk::Format::R32_SFLOAT)
                .offset(84),
        ];

        return attribute_descriptions
            .into_iter()
            .map(|attribute_description| attribute_description)
            .collect();
    }
}
pub(crate) fn map_range_linear(
    value: f32,
    from_min: f32,
    from_max: f32,
    to_min: f32,
    to_max: f32,
) -> f32 {
    let result = f32::clamp(
        to_min + ((value - from_min) / (from_max - from_min)) * (to_max - to_min),
        f32::min(to_min, to_max),
        f32::max(to_max, to_min),
    );
    return result;
}
