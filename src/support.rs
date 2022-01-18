use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Index;
use erupt::vk;
use nalgebra::{Matrix4, Vector2, Vector3, Vector4};

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
pub(crate) struct UniformBufferObject {
    pub(crate) random: [Vector4<f32>; NUM_RANDOM], //std140 packing so it needs to be 16 bytes wide
    pub(crate) lights: [Light; NUM_LIGHTS],        //std140 packing so it needs to be 16 bytes wide
    pub(crate) player_index: u32,
    pub(crate) num_lights: u32,
    pub(crate) _map_mode: u32,
    pub(crate) _selected_province: u32,
    pub(crate) mouse_position: Vector2<f32>,
    pub(crate) screen_size: Vector2<f32>,
    pub(crate) time: f32,
    pub(crate) player_position: Vector3<f32>
}
#[derive(Debug)]
#[repr(C)]
pub(crate) struct PushConstants {
    pub(crate) model: Matrix4<f32>,
    pub(crate) view: Matrix4<f32>,
    pub(crate) proj: Matrix4<f32>,
    pub(crate) texture_index: u32,
    pub(crate) constant: f32,
    pub(crate) bitfield: u32,
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
        ];

        return attribute_descriptions
            .into_iter()
            .map(|attribute_description| attribute_description)
            .collect();
    }
}