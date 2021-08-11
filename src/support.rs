use cgmath::{Vector3, Vector2, Matrix4, Transform, Deg, Matrix3, InnerSpace, Vector4, Quaternion, Point3};
use ash::vk;
use crate::octree::Node;

pub(crate) const FRAMERATE_TARGET: f64 = 280.0;
pub(crate) const NUM_RANDOM: usize = 100;
pub(crate) const FRAME_SAMPLES: usize = 100;
pub(crate) const NUM_MODELS: usize = 100;

#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) struct Vertex {
    pub(crate) position: Vector3<f32>,
    pub(crate) color: Vector3<f32>,
    pub(crate) texture_coordinate: Vector2<f32>,
}

// #[derive(Copy, Clone)]
// #[repr(C)]
// pub(crate) struct NodeBuffer{
//     pub(crate) node_type: Vector4<u32>,
//     pub(crate) node_indices: [Vector4<u32>;8],
// }


#[derive(Copy, Clone)]
#[repr(C)]
pub(crate)struct UniformBufferObject {
    pub(crate)random: [Vector4<f32>; NUM_RANDOM], //std140 packing so it needs to be 16 bytes wide
    pub(crate) player_index: u32,
    pub(crate) value2: i32,
    pub(crate) value3: i32,
    pub(crate) value4: i32,
    pub(crate) mouse_position: Vector2<f32>,
}

#[repr(C)]
pub(crate)struct PushConstants {
    pub(crate)model: Matrix4<f32>,
    pub(crate)view: Matrix4<f32>,
    pub(crate)proj: Matrix4<f32>,
    pub(crate)texture_index: u32,
    pub(crate)constant: f32,
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

