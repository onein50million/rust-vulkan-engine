use std::{sync::Arc, ffi::CString};

use egui::Rgba;
use erupt::vk;

use crate::support::{UniformBufferObject, PushConstants};

use super::{combination_types::CombinedSampledImage, vulkan_data::VulkanData};


const UI_BUFFER_LENGTH: usize = 8192 * 32;


#[repr(C)]
pub struct UIVertex{
    pub pos: egui::Pos2,
    pub uv: egui::Pos2,
    pub color: [u8;4],
}

pub struct UiData {
    pub(crate) vertex_buffer: Option<vk::Buffer>,
    pub(crate) vertex_allocation: Option<vk_mem_erupt::Allocation>,
    pub(crate) vertex_allocation_info: Option<vk_mem_erupt::AllocationInfo>,
    pub(crate) vertex_pointer: Option<*mut UIVertex>,
    pub(crate) index_buffer: Option<vk::Buffer>,
    pub(crate) index_allocation: Option<vk_mem_erupt::Allocation>,
    pub(crate) index_allocation_info: Option<vk_mem_erupt::AllocationInfo>,
    pub(crate) index_pointer: Option<*mut u32>,
    pub(crate) num_indices: u32,
    pub(crate) pipeline: Option<vk::Pipeline>,
    pub(crate) pipeline_layout: Option<vk::PipelineLayout>,
    pub(crate) descriptor_set: Option<vk::DescriptorSet>,
    pub(crate) descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    pub(crate) image: Option<CombinedSampledImage>,
}
impl UiData {
    pub fn update_buffers(&mut self, vertices: &[egui::epaint::Vertex], indices: &[u32]) {
        assert!(vertices.len() < UI_BUFFER_LENGTH);
        assert!(indices.len() < UI_BUFFER_LENGTH);


        let vertices: Box<[_]> = vertices.iter().map(|v|{
            let c = Rgba::from(v.color); //linear color
            
            UIVertex{
                pos: v.pos,
                uv: v.uv,
                color: [
                    (c.r()*255.0) as u8,
                    (c.g()*255.0) as u8,
                    (c.b()*255.0) as u8,
                    (c.a()*255.0) as u8,
                ],
            }
        }).collect();

        // println!("{:?}", vertices[0]);
        unsafe {
            self.vertex_pointer
                .unwrap()
                .copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
            self.index_pointer
                .unwrap()
                .copy_from_nonoverlapping(indices.as_ptr(), indices.len());
        }
        self.num_indices = indices.len() as u32;
    }
}


impl VulkanData{
    pub(crate) fn create_ui_descriptor_set_layout(&mut self) {
        let bindings = [
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        ];
        let layout_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_descriptor_set_layout(&layout_info, None)
        }
        .unwrap();
        self.ui_data.descriptor_set_layout = Some(descriptor_set_layout);
    }

    pub(crate) fn update_ui_descriptors(&self) {
        let descriptor_set = self.ui_data.descriptor_set.unwrap();
        let buffer_infos = vec![vk::DescriptorBufferInfoBuilder::new()
            .buffer(self.uniform_buffers[0])
            .offset(0)
            .range((std::mem::size_of::<UniformBufferObject>()) as vk::DeviceSize)];
        let combined_image = self.ui_data.image.as_ref().unwrap_or(
            self.fallback_texture
                .as_ref()
                .unwrap()
                .albedo
                .as_ref()
                .unwrap(),
        );
        let font_image_infos = vec![vk::DescriptorImageInfoBuilder::new()
            .image_view(combined_image.image_view)
            .sampler(combined_image.sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

        let descriptor_writes = [
            vk::WriteDescriptorSetBuilder::new()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_infos),
            vk::WriteDescriptorSetBuilder::new()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&font_image_infos),
        ];

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .update_descriptor_sets(&descriptor_writes, &[]);
        }
    }

    pub fn update_ui_texture(&mut self, texture: Arc<egui::epaint::Texture>) {
        match &self.ui_data.image {
            None => {}
            Some(image) => {
                unsafe {
                    self.device.as_ref().unwrap().device_wait_idle().unwrap();
                }

                self.allocator
                    .as_ref()
                    .unwrap()
                    .destroy_image(image.image, &image.allocation);
                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .destroy_image_view(Some(image.image_view), None)
                };
                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .destroy_sampler(Some(image.sampler), None)
                };
            }
        }
        let image_format = vk::Format::R8_UINT;

        let image_info = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_2D)
            .extent(vk::Extent3D {
                width: texture.width as u32,
                height: texture.height as u32,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(image_format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(
                vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .samples(vk::SampleCountFlagBits::_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_info = vk_mem_erupt::AllocationCreateInfo {
            usage: vk_mem_erupt::MemoryUsage::GpuOnly,
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ..Default::default()
        };

        let (image, allocation, _) = self
            .allocator
            .as_ref()
            .unwrap()
            .create_image(&image_info, &allocation_info)
            .unwrap();

        let view_info = vk::ImageViewCreateInfoBuilder::new()
            .image(image)
            .view_type(vk::ImageViewType::_2D)
            .format(image_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let image_view = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_image_view(&view_info, None)
        }
        .unwrap();

        let sampler_info = vk::SamplerCreateInfoBuilder::new()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(1.0)
            .border_color(vk::BorderColor::INT_OPAQUE_WHITE)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE);

        let sampler = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_sampler(&sampler_info, None)
                .unwrap()
        };

        let buffer_info = vk::BufferCreateInfoBuilder::new()
            .size(texture.pixels.len() as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC);

        let buffer_allocation_create_info = vk_mem_erupt::AllocationCreateInfo {
            usage: vk_mem_erupt::MemoryUsage::CpuOnly,
            flags: vk_mem_erupt::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };
        let (staging_buffer, staging_buffer_allocation, staging_buffer_allocation_info) = self
            .allocator
            .as_ref()
            .unwrap()
            .create_buffer(&buffer_info, &buffer_allocation_create_info)
            .unwrap();
        unsafe {
            staging_buffer_allocation_info
                .get_mapped_data()
                .copy_from_nonoverlapping(texture.pixels.as_ptr(), texture.pixels.len());
        };
        self.transition_image_layout(
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        );
        let command_buffer = self.begin_single_time_commands();

        let regions = vec![vk::BufferImageCopyBuilder::new()
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: texture.width as u32,
                height: texture.height as u32,
                depth: 1,
            })];
        unsafe {
            self.device.as_ref().unwrap().cmd_copy_buffer_to_image(
                command_buffer,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            )
        };

        self.end_single_time_commands(command_buffer);

        self.transition_image_layout(
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        );

        unsafe {
            self.device.as_ref().unwrap().device_wait_idle().unwrap();
        }
        self.allocator
            .as_ref()
            .unwrap()
            .destroy_buffer(staging_buffer, &staging_buffer_allocation);

        self.ui_data.image = Some(CombinedSampledImage {
            image,
            image_view,
            sampler,
            allocation,
            width: texture.width as u32,
            height: texture.height as u32,
        });

        self.update_ui_descriptors();
    }

    pub(crate) fn create_ui_data(&mut self) {
        //Small, CPU accessible, persistently mapped buffer for the UI to write to

        let buffer_info = vk::BufferCreateInfoBuilder::new()
            .size(
                (UI_BUFFER_LENGTH * std::mem::size_of::<egui::epaint::Vertex>()) as vk::DeviceSize,
            )
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER);

        let allocation_info = vk_mem_erupt::AllocationCreateInfo {
            usage: vk_mem_erupt::MemoryUsage::CpuToGpu,
            flags: vk_mem_erupt::AllocationCreateFlags::MAPPED,
            required_flags: Default::default(),
            preferred_flags: Default::default(),
            memory_type_bits: 0,
            pool: None,
            user_data: None,
        };

        let (vertex_buffer, vertex_allocation, vertex_allocation_info) = self
            .allocator
            .as_ref()
            .unwrap()
            .create_buffer(&buffer_info, &allocation_info)
            .unwrap();

        let vertex_pointer = self
            .allocator
            .as_ref()
            .unwrap()
            .map_memory(&vertex_allocation)
            .unwrap() as *mut UIVertex;

        let buffer_info = vk::BufferCreateInfoBuilder::new()
            .size((UI_BUFFER_LENGTH * std::mem::size_of::<u32>()) as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::INDEX_BUFFER);

        let allocation_info = vk_mem_erupt::AllocationCreateInfo {
            usage: vk_mem_erupt::MemoryUsage::CpuToGpu,
            flags: vk_mem_erupt::AllocationCreateFlags::MAPPED,
            required_flags: Default::default(),
            preferred_flags: Default::default(),
            memory_type_bits: 0,
            pool: None,
            user_data: None,
        };

        let (index_buffer, index_allocation, index_allocation_info) = self
            .allocator
            .as_ref()
            .unwrap()
            .create_buffer(&buffer_info, &allocation_info)
            .unwrap();
        let index_pointer = self
            .allocator
            .as_ref()
            .unwrap()
            .map_memory(&index_allocation)
            .unwrap() as *mut u32;

        //get descriptor layout
        let descriptor_set_layout = self.ui_data.descriptor_set_layout.unwrap();
        //create descriptor set

        let layouts = [descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfoBuilder::new()
            .descriptor_pool(self.descriptor_pool.unwrap())
            .set_layouts(&layouts);
        let descriptor_sets = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .allocate_descriptor_sets(&allocate_info)
        }
        .unwrap();
        let descriptor_set = descriptor_sets[0];

        //create pipeline layout

        let push_constant_range = vk::PushConstantRangeBuilder::new()
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
        let push_constant_ranges = [push_constant_range];

        let descriptor_set_layouts = [descriptor_set_layout];
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let pipeline_layout = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_pipeline_layout(&pipeline_layout_create_info, None)
        }
        .unwrap();

        self.ui_data.vertex_buffer = Some(vertex_buffer);
        self.ui_data.vertex_allocation = Some(vertex_allocation);
        self.ui_data.vertex_allocation_info = Some(vertex_allocation_info);
        self.ui_data.vertex_pointer = Some(vertex_pointer);
        self.ui_data.index_buffer = Some(index_buffer);
        self.ui_data.index_allocation = Some(index_allocation);
        self.ui_data.index_allocation_info = Some(index_allocation_info);
        self.ui_data.index_pointer = Some(index_pointer);
        self.ui_data.num_indices = 0;
        self.ui_data.pipeline_layout = Some(pipeline_layout);
        self.ui_data.descriptor_set = Some(descriptor_set);
    }

    pub(crate) fn create_ui_pipeline(&mut self) {
        //create pipeline
        let binding_descriptions = vec![vk::VertexInputBindingDescriptionBuilder::new()
            .binding(0)
            .stride(std::mem::size_of::<egui::epaint::Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let attribute_descriptions = vec![
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(8),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(2)
                .format(vk::Format::R8G8B8A8_UNORM)
                .offset(16),
        ];

        let vertex_input_create_info = vk::PipelineVertexInputStateCreateInfoBuilder::new()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);
        let viewport = vk::ViewportBuilder::new()
            .x(0.0f32)
            .y(0.0f32)
            .width(self.surface_capabilities.unwrap().current_extent.width as f32)
            .height(self.surface_capabilities.unwrap().current_extent.height as f32)
            .min_depth(0.0f32)
            .max_depth(1.0f32);

        let scissor = vk::Rect2DBuilder::new()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(self.surface_capabilities.unwrap().current_extent);

        let viewports = [viewport];
        let scissors = [scissor];
        let viewport_state_create_info = vk::PipelineViewportStateCreateInfoBuilder::new()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer = vk::PipelineRasterizationStateCreateInfoBuilder::new()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0f32);

        let multisampling = vk::PipelineMultisampleStateCreateInfoBuilder::new()
            .sample_shading_enable(false)
            .rasterization_samples(self.msaa_samples);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentStateBuilder::new()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_blend_attachments = [color_blend_attachment];

        let color_blending = vk::PipelineColorBlendStateCreateInfoBuilder::new()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::OR)
            .attachments(&color_blend_attachments);

        let vert_entry_string = CString::new("main").unwrap();
        let frag_entry_string = CString::new("main").unwrap();

        let vert_shader_file = std::fs::read("shaders/ui_vert.spv").unwrap();
        let frag_shader_file = std::fs::read("shaders/ui_frag.spv").unwrap();
        let vert_shader_code = erupt::utils::decode_spv(&vert_shader_file).unwrap();
        let frag_shader_code = erupt::utils::decode_spv(&frag_shader_file).unwrap();
        let vert_shader_module =
            Self::create_shader_module(&self.device.as_ref().unwrap(), vert_shader_code);
        let frag_shader_module =
            Self::create_shader_module(&self.device.as_ref().unwrap(), frag_shader_code);

        let vert_shader_stage_create_info = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::VERTEX)
            .module(vert_shader_module)
            .name(vert_entry_string.as_c_str());
        let frag_shader_stage_create_info = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::FRAGMENT)
            .module(frag_shader_module)
            .name(frag_entry_string.as_c_str());
        let shader_stages = vec![vert_shader_stage_create_info, frag_shader_stage_create_info];

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfoBuilder::new()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::ALWAYS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false);

        let pipeline_infos = [vk::GraphicsPipelineCreateInfoBuilder::new()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_create_info)
            .input_assembly_state(&input_assembly_create_info)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .layout(self.ui_data.pipeline_layout.unwrap())
            .render_pass(self.render_pass.unwrap())
            .subpass(1)
            .depth_stencil_state(&depth_stencil)];

        let pipeline = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_graphics_pipelines(None, &pipeline_infos, None)
        }
        .unwrap()[0];

        self.ui_data.pipeline = Some(pipeline);
    }

}