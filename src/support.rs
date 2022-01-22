use erupt::vk;
use nalgebra::{Matrix4, Scale3, Translation3, UnitQuaternion, Vector2, Vector3, Vector4};
use std::collections::HashMap;
use std::convert::TryInto;
use std::hash::Hash;
use std::ops::Index;

pub(crate) const FRAMERATE_TARGET: f64 = 280.0;
pub(crate) const NUM_RANDOM: usize = 100;
pub(crate) const FRAME_SAMPLES: usize = 100;
pub(crate) const NUM_MODELS: usize = 1000;
pub(crate) const NUM_LIGHTS: usize = 1;

pub(crate) mod flags {
    pub(crate) const IS_CUBEMAP: u32 = 0b1;
    pub(crate) const IS_HIGHLIGHTED: u32 = 0b10;
    pub(crate) const IS_VIEWMODEL: u32 = 0b100;
    pub(crate) const IS_GLOBE: u32 = 0b1000;
}

#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) struct Vertex {
    pub(crate) position: Vector3<f32>,
    pub(crate) normal: Vector3<f32>,
    pub(crate) tangent: Vector4<f32>,
    pub(crate) texture_coordinate: Vector2<f32>,
    pub(crate) texture_type: u32, //Texture index for multiple textures I think
    pub(crate) bone_indices: Vector4<u32>,
    pub(crate) bone_weights: Vector4<f32>,
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
        let rng = fastrand::Rng::new();
        let distance = 5.0;
        let power = 0.0;
        let position = Vector4::new(0.0, 0.0, 0.0, 0.0);
        let color = Vector4::new(1.0, 0.9, 0.9, 0.0) * power;

        println!("Position: {:}", position);
        Self { position, color }
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) struct Bone {
    pub(crate) matrix: Matrix4<f32>,
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
pub(crate) struct BoneSet { //One frame
    pub(crate) input_tangent: f32,
    pub(crate) output_tangent: f32,
    pub(crate) _padding1: f32,
    pub(crate) _padding2: f32,
    pub(crate) bones: [Bone; NUM_BONES_PER_BONESET],
}

#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) struct ShaderStorageBufferObject {
    pub(crate) bone_sets: [BoneSet; NUM_BONE_SETS],
}
impl ShaderStorageBufferObject{
    pub(crate) fn new_boxed() -> Box<Self>{

        //need to do unsafe things because we can't create a array on the heap (afaik)
        let layout =  std::alloc::Layout::new::<[BoneSet;NUM_BONE_SETS]>();
        unsafe{
            let pointer = std::alloc::alloc(layout) as *mut BoneSet;
            for i in 0..(NUM_BONE_SETS){
                pointer.offset(i as isize).write(BoneSet{
                    input_tangent: 0.0,
                    output_tangent: 0.0,
                    _padding1: 0.0,
                    _padding2: 0.0,
                    bones: [Bone::new();NUM_BONES_PER_BONESET]
                });
            }
            Box::new(std::ptr::NonNull::new_unchecked(pointer as *mut Self).as_ref().to_owned())
        }
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) struct UniformBufferObject {
    pub(crate) view: Matrix4<f32>,
    pub(crate) proj: Matrix4<f32>,
    pub(crate) random: [Vector4<f32>; NUM_RANDOM], //std140 packing so it needs to be 16 bytes wide
    pub(crate) lights: [Light; NUM_LIGHTS],        //std140 packing so it needs to be 16 bytes wide
    pub(crate) player_index: u32,
    pub(crate) num_lights: u32,
    pub(crate) _map_mode: u32,
    pub(crate) exposure: f32,
    pub(crate) mouse_position: Vector2<f32>,
    pub(crate) screen_size: Vector2<f32>,
    pub(crate) time: f32,
    pub(crate) player_position: Vector3<f32>,
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
        ];

        return attribute_descriptions
            .into_iter()
            .map(|attribute_description| attribute_description)
            .collect();
    }
}
