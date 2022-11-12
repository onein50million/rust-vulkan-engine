use std::{mem::size_of, ffi::c_void};

use erupt::{vk, DeviceLoader};
use nalgebra::Matrix4;

use crate::{support::{Vertex, PushConstants}, cube::{QUAD_INDICES, FULLSCREEN_QUAD_VERTICES}};

use super::{animation::AnimationObject, combination_types::{TextureSet, CombinedSampledImage}, vulkan_data::VulkanData};

pub struct RenderObject {
    pub(crate) vertex_start: u32,
    pub(crate) vertex_count: u32,
    pub(crate) index_start: u32,
    pub(crate) index_count: u32,
    texture_index: usize,
    texture_type: vk::ImageViewType,
    is_highlighted: bool,
    pub(crate) is_globe: bool,
    pub(crate) is_view_proj_matrix_ignored: bool,
    pub(crate) is_viewmodel: bool,
    pub model: Matrix4<f32>,
    pub(crate) animations: Vec<AnimationObject>,
    previous_frame: usize,
    next_frame: usize,
    animation_progress: f64,
}

impl RenderObject {
    pub(crate) fn new(
        vulkan_data: &mut VulkanData,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
        animations: Vec<AnimationObject>,
        texture: TextureSet,
        is_cubemap: bool,
    ) -> Self {
        let vertex_start = vulkan_data.vertices.len() as u32;
        let index_start = vulkan_data.indices.len() as u32;

        let vertex_count = vertices.len() as u32;
        let index_count = indices.len() as u32;

        vulkan_data.vertices.extend(vertices);
        for index in indices {
            vulkan_data.indices.push(index + vertex_start);
        }
        let texture_index;
        match is_cubemap {
            true => {
                texture_index = vulkan_data.cubemaps.len();
                vulkan_data.cubemaps.push(texture.albedo.unwrap());
            }
            false => {
                texture_index = vulkan_data.textures.len();
                vulkan_data.textures.push(texture);
            }
        }

        let (previous_frame, next_frame) = if animations.len() > 0 {
            (animations[0].frame_start, animations[0].frame_start + 1)
        } else {
            (0, 0)
        };

        let mut object = RenderObject {
            vertex_start,
            vertex_count,
            index_start,
            index_count,
            texture_index,
            texture_type: if is_cubemap {
                vk::ImageViewType::CUBE
            } else {
                vk::ImageViewType::_2D
            },
            is_highlighted: false,
            is_viewmodel: false,
            is_globe: false,
            model: Matrix4::identity(),
            animations,
            previous_frame,
            next_frame,
            animation_progress: 0.0,
            is_view_proj_matrix_ignored: false,
        };
        if object.animations.len() > 0 {
            object.set_animation(0, 0.0, 0, 1);
        }
        return object;
    }
    pub fn set_animation(
        &mut self,
        animation_index: usize,
        progress: f64,
        previous_frame: usize,
        next_frame: usize,
    ) {
        self.previous_frame = self.animations[animation_index].frame_start + previous_frame;
        self.next_frame = self.animations[animation_index].frame_start + next_frame;
        self.animation_progress = progress;
    }
    pub(crate) fn get_animation_length(&self, animation_index: usize) -> usize {
        self.animations[animation_index].frame_count
    }
}

impl Drawable for RenderObject {
    fn draw(
        &self,
        device: &DeviceLoader,
        command_buffer: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
    ) {
        unsafe {
            let texture_index = self.texture_index as u32;
            let mut bitfield = 0;
            bitfield |= if self.texture_type == vk::ImageViewType::CUBE {
                crate::support::flags::IS_CUBEMAP
            } else {
                0
            };
            bitfield |= if self.is_highlighted {
                crate::support::flags::IS_HIGHLIGHTED
            } else {
                0
            };
            bitfield |= if self.is_viewmodel {
                crate::support::flags::IS_VIEWMODEL
            } else {
                0
            };
            bitfield |= if self.is_globe {
                crate::support::flags::IS_GLOBE
            } else {
                0
            };
            bitfield |= if self.is_view_proj_matrix_ignored {
                crate::support::flags::IS_VIEW_PROJ_MATRIX_IGNORED
            } else {
                0
            };
            let previous_frame = (self.previous_frame as u8).to_be_bytes();
            let next_frame = (self.next_frame as u8).to_be_bytes();

            let animation_progress =
                ((self.animation_progress * u16::MAX as f64) as u16).to_be_bytes();

            let animation_bytes = [
                &previous_frame[..],
                &next_frame[..],
                &animation_progress[..],
            ]
            .concat();

            let push_constant = PushConstants {
                model: self.model,
                texture_index,
                bitfield,
                animation_frames: u32::from_be_bytes(animation_bytes.try_into().unwrap()),
            };
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                size_of::<PushConstants>() as u32,
                &push_constant as *const _ as *const c_void,
            );
        }

        if self.index_count > 0 {
            unsafe {
                device.cmd_draw_indexed(
                    command_buffer,
                    self.index_count,
                    1,
                    self.index_start,
                    0,
                    0,
                );
            }
        } else if self.vertex_count > 0 {
            unsafe { device.cmd_draw(command_buffer, self.vertex_count, 1, self.vertex_start, 0) };
        }
    }
}

pub(crate) trait Drawable {
    fn draw(
        &self,
        device: &DeviceLoader,
        command_buffer: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
    );
}
pub struct Cubemap {
    pub render_object: RenderObject,
}

impl Cubemap {
    pub(crate) fn new(vulkan_data: &mut VulkanData, texture_path: std::path::PathBuf) -> Self {
        let indices = QUAD_INDICES.to_vec();
        let vertices = FULLSCREEN_QUAD_VERTICES.to_vec();

        let texture = TextureSet {
            albedo: CombinedSampledImage::new(
                vulkan_data,
                texture_path,
                vk::ImageViewType::CUBE,
                vk::Format::R32G32B32A32_SFLOAT,
                false,
            ),
            normal: None,
            roughness_metalness_ao: None,
        };

        let mut render_object =
            RenderObject::new(vulkan_data, vertices, indices, vec![], texture, true);
        render_object.is_view_proj_matrix_ignored = true;
        let cubemap = Cubemap { render_object };

        return cubemap;
    }
}

impl Cubemap {
    pub(crate) fn process(&mut self) {
        self.render_object.model = Matrix4::identity();
    }
}
impl Drawable for Cubemap {
    fn draw(
        &self,
        device: &DeviceLoader,
        command_buffer: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
    ) {
        self.render_object
            .draw(device, command_buffer, pipeline_layout);
    }
}
