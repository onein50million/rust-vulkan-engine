use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Index;
use erupt::vk;
use nalgebra::{Matrix4, Vector2, Vector3, Vector4};
use strum::{EnumCount, IntoEnumIterator};

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
    pub(crate) elevation: f32,
    pub(crate) aridity: f32,
    pub(crate) population: f32,
    pub(crate) warmest_temperature: f32,
    pub(crate) coldest_temperature: f32,
    pub(crate) province_id: u32,
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
            elevation: 0.0,
            aridity: 0.0,
            population: 0.0,
            warmest_temperature: 0.0,
            coldest_temperature: 0.0,
            province_id: 0,
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
        let power = 10.0;
        let position = Vector4::new(0.0, 0.0, 0.0, 0.0);
        let color = Vector4::new(1.0, 0.9, 0.9, 0.0) * power;

        println!("Position: {:}", position);
        Self { position, color }
    }
}

pub(crate) mod map_modes {
    pub(crate) const SATELITE: u8 = 0;
    pub(crate) const ELEVATION: u8 = 1;
    pub(crate) const ARIDITY: u8 = 2;
    pub(crate) const POPULATION: u8 = 3;
    pub(crate) const TEMPERATURE: u8 = 4;
}

#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) struct UniformBufferObject {
    pub(crate) random: [Vector4<f32>; NUM_RANDOM], //std140 packing so it needs to be 16 bytes wide
    pub(crate) lights: [Light; NUM_LIGHTS],        //std140 packing so it needs to be 16 bytes wide
    pub(crate) player_index: u32,
    pub(crate) num_lights: u32,
    pub(crate) map_mode: u32,
    pub(crate) selected_province: u32,
    pub(crate) mouse_position: Vector2<f32>,
    pub(crate) screen_size: Vector2<f32>,
    pub(crate) time: f32,
    pub(crate) b: f32,
    pub(crate) c: f32,
    pub(crate) d: f32,
    pub(crate) planet_model_matrix: Matrix4<f32>,
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
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(5)
                .format(vk::Format::R32_SFLOAT)
                .offset(52),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(6)
                .format(vk::Format::R32_SFLOAT)
                .offset(56),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(7)
                .format(vk::Format::R32_SFLOAT)
                .offset(60),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(8)
                .format(vk::Format::R32_SFLOAT)
                .offset(64),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(9)
                .format(vk::Format::R32_SFLOAT)
                .offset(68),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(10)
                .format(vk::Format::R32_UINT)
                .offset(72),
        ];

        return attribute_descriptions
            .into_iter()
            .map(|attribute_description| attribute_description)
            .collect();
    }
}

#[derive(Debug)]
pub(crate) struct RatioMap<Key>{
    ratios: HashMap<Key, f64>,
}

impl<Key: Hash + Eq> RatioMap<Key> {
     pub(crate) fn new() -> Self{
         Self{
             ratios: HashMap::new()
         }
     }

    pub(crate) fn get(&self, index: &Key) -> f64{
        *self.ratios.get(index).unwrap()
    }
    // pub(crate) fn set(&self, index: Key, new_ratio: f64) -> f64{
    //     self.ratios[index]
    // }
}