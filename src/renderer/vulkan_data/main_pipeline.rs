use std::{ffi::CString, path::{Path, PathBuf}};

use erupt::{vk, utils::decode_spv, vk1_0::Extent3DBuilder};
use nalgebra::{Vector4, Vector2};

use crate::{renderer::{DanielError, drawables::Cubemap, combination_types::{CombinedSampledImage, CombinedDescriptor, DescriptorInfoData, CombinedImage}}, support::{PushConstants, Vertex, PostProcessPushConstants, NUM_RANDOM, UniformBufferObject,}};

use super::{utilities::get_random_vector, VulkanData};

impl VulkanData {
    pub(crate) fn create_depth_resources(&mut self) {
        let depth_format = self.find_depth_format();

        let (depth_image, depth_image_memory) = self.create_image_with_memory(
            self.surface_capabilities.unwrap().current_extent.width,
            self.surface_capabilities.unwrap().current_extent.height,
            1,
            self.msaa_samples,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::INPUT_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        let sampler_info = vk::SamplerCreateInfoBuilder::new()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
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

        self.depth_image = Some(depth_image);
        self.depth_sampler = Some(sampler);
        self.depth_image_memory = Some(depth_image_memory);
        self.depth_image_view = Some(self.create_image_view(
            depth_image,
            vk::Format::D32_SFLOAT,
            vk::ImageAspectFlags::DEPTH,
            1,
        ));
    }




    pub(crate) fn create_swapchain_image_views(&mut self) {
        self.swapchain_images = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_swapchain_images_khr(self.swapchain.unwrap(), None)
        }
        .unwrap();

        dbg!(self.swapchain_images.len());

        self.swapchain_image_views = 
            self.swapchain_images
                .iter()
                .map(|image| {
                    return self.create_image_view(
                        *image,
                        self.surface_format.unwrap().format,
                        vk::ImageAspectFlags::COLOR,
                        1,
                    );
                })
                .collect::<Vec<_>>();
    }

    pub(crate) fn create_render_pass(&mut self) {
        let push_constant_range = vk::PushConstantRangeBuilder::new()
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
        let push_constant_ranges = [push_constant_range];

        let descriptor_set_layouts = [
            self.descriptor_set_layout.unwrap(),
            self.ui_data.descriptor_set_layout.unwrap(),
        ];
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(&push_constant_ranges);

        self.pipeline_layout = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_pipeline_layout(&pipeline_layout_create_info, None)
            }
            .unwrap(),
        );

        let albedo_attachment = vk::AttachmentDescriptionBuilder::new()
            .format(vk::Format::R8G8B8A8_SRGB)
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let normals_attachment = vk::AttachmentDescriptionBuilder::new()
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let rough_metal_ao_attachment = vk::AttachmentDescriptionBuilder::new()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let postprocess_color_attachment = vk::AttachmentDescriptionBuilder::new()
            .format(self.surface_format.unwrap().format)
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_resolve_attachment = vk::AttachmentDescriptionBuilder::new()
            .format(self.surface_format.unwrap().format)
            .samples(vk::SampleCountFlagBits::_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_resolve_attachment_reference = vk::AttachmentReferenceBuilder::new()
            .attachment(4)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attachment = vk::AttachmentDescriptionBuilder::new()
            .format(self.find_depth_format())
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL);
        let depth_attachment_reference = vk::AttachmentReferenceBuilder::new()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachment_references = vec![
            vk::AttachmentReferenceBuilder::new()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            vk::AttachmentReferenceBuilder::new()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            vk::AttachmentReferenceBuilder::new()
            .attachment(3)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),

        ];

        let mut draw_subpass = vk::SubpassDescriptionBuilder::new()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_references)
            .depth_stencil_attachment(&depth_attachment_reference);
        
        let resolve_attachment_references = vec![color_resolve_attachment_reference];

        if self.msaa_samples != vk::SampleCountFlagBits::_1 {
            draw_subpass = draw_subpass.resolve_attachments(&resolve_attachment_references)
        }
        let dependencies = [
            vk::SubpassDependencyBuilder::new()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE
                )
                .src_access_mask(vk::AccessFlags::MEMORY_READ)
                .dst_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                )
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::COLOR_ATTACHMENT_READ,
                ),
            vk::SubpassDependencyBuilder::new()
                .src_subpass(0)
                .dst_subpass(1)
                .src_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                )
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE | vk::PipelineStageFlags::FRAGMENT_SHADER,
                )
                .dst_access_mask(
                    vk::AccessFlags::MEMORY_READ,
                ),
            vk::SubpassDependencyBuilder::new()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                )
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                )
                .dst_access_mask(
                    vk::AccessFlags::MEMORY_READ,
                ),
            
            
            ];
            
        let attachments = if self.msaa_samples != vk::SampleCountFlagBits::_1 {
            vec![albedo_attachment, depth_attachment, postprocess_color_attachment, color_resolve_attachment]
        } else {
            vec![albedo_attachment, depth_attachment, normals_attachment, rough_metal_ao_attachment, postprocess_color_attachment]
        };

        // let input_attachments = &[color_attachment_reference, depth_attachment_reference];
        let postprocess_input_attachments = &[
            vk::AttachmentReferenceBuilder::new().attachment(0).layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::AttachmentReferenceBuilder::new().attachment(1).layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::AttachmentReferenceBuilder::new().attachment(2).layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::AttachmentReferenceBuilder::new().attachment(3).layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ];
        let postprocess_color_attachments = &[
            vk::AttachmentReferenceBuilder::new().attachment(4).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        ];
        let postprocess_subpass = vk::SubpassDescriptionBuilder::new()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .input_attachments(postprocess_input_attachments)
        .color_attachments(postprocess_color_attachments);
        let subpasses = [draw_subpass, postprocess_subpass];

        let render_pass_info = vk::RenderPassCreateInfoBuilder::new()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        self.render_pass = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_render_pass(&render_pass_info, None)
            }
            .unwrap(),
        );
    }


    pub(crate) fn create_graphics_pipelines(&mut self) {
        let binding_descriptions = vec![Vertex::get_binding_description()];
        let attribute_descriptions = Vertex::get_attribute_description();

        let vertex_input_create_info = vk::PipelineVertexInputStateCreateInfoBuilder::new()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let dynamic_states = [vk::DynamicState::DEPTH_TEST_ENABLE_EXT];

        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfoBuilder::new().dynamic_states(&dynamic_states);

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
            .cull_mode(vk::CullModeFlags::BACK)
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

        let blend_attachment_deferred = &[color_blend_attachment,color_blend_attachment,color_blend_attachment];
        let color_blending_deferred_pass = vk::PipelineColorBlendStateCreateInfoBuilder::new()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::OR)
            .attachments(blend_attachment_deferred);

        let blend_attachment_postprocess = &[color_blend_attachment];
        let color_blending_postprocess_pass = vk::PipelineColorBlendStateCreateInfoBuilder::new()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::OR)
            .attachments(blend_attachment_postprocess);

        let shader_main_entry = CString::new("main").unwrap();

        let vert_shader_stage_create_info = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::VERTEX)
            .module(self.vert_shader_module.unwrap())
            .name(shader_main_entry.as_c_str());
        let frag_shader_stage_create_info = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::FRAGMENT)
            .module(self.frag_shader_module.unwrap())
            .name(shader_main_entry.as_c_str());
        let shader_stages = vec![vert_shader_stage_create_info, frag_shader_stage_create_info];

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfoBuilder::new()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false);


        let postprocess_subpass_shader_stages = &[
            vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::VERTEX)
            .module(VulkanData::create_shader_module(self.device.as_ref().unwrap(), decode_spv(include_bytes!("../../../shaders/postprocess_subpass/vert.spv")).unwrap()))
            .name(shader_main_entry.as_c_str()),
            vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::FRAGMENT)
            .module(VulkanData::create_shader_module(self.device.as_ref().unwrap(), decode_spv(include_bytes!("../../../shaders/postprocess_subpass/frag.spv")).unwrap()))
            .name(shader_main_entry.as_c_str()),
        ];

        let postprocess_subpass_vertex_input_create_info = vk::PipelineVertexInputStateCreateInfoBuilder::new(); //bufferless fullscreen quad https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/

        let push_constant_range = vk::PushConstantRangeBuilder::new()
            .offset(0)
            .size(std::mem::size_of::<PostProcessPushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let push_constant_ranges = [push_constant_range];

        let descriptor_set_layouts = [
            self.postprocess_descriptor_set_layout.unwrap()
        ];
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(&push_constant_ranges);

        self.postprocess_subpass_pipeline_layout = 
            Some(unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_pipeline_layout(&pipeline_layout_create_info, None)
            }
            .unwrap());

        let postprocess_subpass_depth_stencil = vk::PipelineDepthStencilStateCreateInfoBuilder::new()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false);
        let postprocess_subpass_pipeline_info = vk::GraphicsPipelineCreateInfoBuilder::new()
            .stages(postprocess_subpass_shader_stages)
            .vertex_input_state(&postprocess_subpass_vertex_input_create_info)
            .input_assembly_state(&input_assembly_create_info)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending_postprocess_pass)
            .layout(self.postprocess_subpass_pipeline_layout.unwrap())
            .render_pass(self.render_pass.unwrap())
            .subpass(1)
            .depth_stencil_state(&postprocess_subpass_depth_stencil);

        let pipeline_infos = [
            vk::GraphicsPipelineCreateInfoBuilder::new()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_create_info)
            .input_assembly_state(&input_assembly_create_info)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending_deferred_pass)
            .layout(self.pipeline_layout.unwrap())
            .render_pass(self.render_pass.unwrap())
            .subpass(0)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_state_info),
            postprocess_subpass_pipeline_info
            ];


        

        self.graphics_pipelines = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_graphics_pipelines(None, &pipeline_infos, None)
        }
        .unwrap();
    }

    pub(crate) fn create_framebuffers(&mut self) {
        self.framebuffers = self.swapchain_image_views.iter().map(|&swapchan_image_view|{
            let attachments = if self.msaa_samples != vk::SampleCountFlagBits::_1 {
                vec![
                    self.albedo_image_view.unwrap(),
                    self.depth_image_view.unwrap(),
                    self.color_resolve_image_view.unwrap(),
                ]
            } else {
                vec![
                    self.albedo_image_view.unwrap(),
                    self.depth_image_view.unwrap(),
                    self.normals_image.as_ref().unwrap().image_view,
                    self.rough_metal_ao_image.as_ref().unwrap().image_view,
                    swapchan_image_view
                ]
            };
            let framebuffer_create_info = vk::FramebufferCreateInfoBuilder::new()
                .render_pass(self.render_pass.unwrap())
                .attachments(&attachments)
                .width(self.surface_capabilities.unwrap().current_extent.width)
                .height(self.surface_capabilities.unwrap().current_extent.height)
                .layers(1);
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_framebuffer(&framebuffer_create_info, None)
            }
            .unwrap()
        }).collect();
        
        
        
        ({
});
    }

    pub(crate) fn create_command_buffers(&mut self) {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
            .command_pool(self.command_pool.unwrap())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        self.command_buffer = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .allocate_command_buffers(&command_buffer_allocate_info)
            }
            .unwrap()[0],
        );
    }

    pub(crate) fn create_swapchain(&mut self) {
        let surface_formats = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_surface_formats_khr(
                    self.physical_device.unwrap(),
                    self.surface.unwrap(),
                    None,
                )
        }
        .unwrap();

        let default_surface_format = surface_formats[0];
        println!("Surface formats: {:?}", surface_formats);

        let surface_format = surface_formats.into_iter().find(|format| {
            let format_matches = format.format == vk::Format::R8G8B8A8_SRGB
                || format.format == vk::Format::B8G8R8_SRGB
                || format.format == vk::Format::R8G8B8_SRGB
                || format.format == vk::Format::B8G8R8A8_SRGB
                || format.format == vk::Format::A8B8G8R8_SRGB_PACK32;
            let color_space_matches = format.color_space
                == vk::ColorSpaceKHR::COLORSPACE_SRGB_NONLINEAR_KHR
                || format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR;
            return format_matches && color_space_matches;
        });

        self.surface_format = Some(match surface_format {
            None => {
                println!("ERROR: Unable to find surface format, using default");
                default_surface_format
            }
            Some(found_format) => found_format,
        });
        println!(
            "Selected Surface format: {:?}",
            self.surface_format.unwrap()
        );

        let swapchain_create_info = vk::SwapchainCreateInfoKHRBuilder::new()
            .surface(self.surface.unwrap())
            .min_image_count(self.surface_capabilities.unwrap().min_image_count)
            .image_color_space(self.surface_format.unwrap().color_space)
            .image_format(self.surface_format.unwrap().format)
            .image_extent(self.surface_capabilities.unwrap().current_extent)
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
            )
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(self.surface_capabilities.unwrap().current_transform)
            .composite_alpha(vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
            .present_mode(vk::PresentModeKHR::IMMEDIATE_KHR)
            .clipped(true);

        self.swapchain = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_swapchain_khr(&swapchain_create_info, None)
            }
            .unwrap(),
        );
        self.swapchain_created = true;
    }
    pub(crate) fn recreate_swapchain(&mut self) -> Result<(), DanielError> {
        self.get_surface_capabilities();

        if self.surface_capabilities.unwrap().current_extent.height == 0
            || self.surface_capabilities.unwrap().current_extent.width == 0
        {
            unsafe {
                self.device.as_ref().unwrap().device_wait_idle().unwrap();
            }
            return Err(DanielError::Minimized);
        }

        self.create_color_resources();
        self.create_normals_image();
        self.create_rough_metal_ao_image();
        self.create_depth_resources();
        self.create_swapchain();
        self.create_swapchain_image_views();
        self.create_render_pass();
        self.create_graphics_pipelines();

        self.create_framebuffers();
        self.create_buffers();
        self.transfer_data_to_storage_buffer(&self.storage_buffer_object);
        self.create_ui_pipeline();
        self.create_command_buffers();
        self.update_descriptor_sets();

        self.swapchain_created = true;
        Ok(())
    }

    pub(crate) fn create_color_resources(&mut self) {
        let color_format = vk::Format::R8G8B8A8_SRGB;
        let (color_image, color_image_memory) = self.create_image_with_memory(
            self.surface_capabilities.unwrap().current_extent.width,
            self.surface_capabilities.unwrap().current_extent.height,
            1,
            self.msaa_samples,
            color_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::INPUT_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        self.color_resolve_image = Some(color_image);
        self.color_resolve_image_memory = Some(color_image_memory);
        self.color_resolve_image_view =
            Some(self.create_image_view(color_image, color_format, vk::ImageAspectFlags::COLOR, 1));
    

        let draw_pass_image_info = vk::ImageCreateInfoBuilder::new()
            .extent(
                *Extent3DBuilder::new()
                .width(self.surface_capabilities.unwrap().current_extent.width)
                .height(self.surface_capabilities.unwrap().current_extent.height)
                .depth(1))
            .image_type(vk::ImageType::_2D)
            .array_layers(1)
            .flags(vk::ImageCreateFlags::empty())
            .format(color_format)
            .mip_levels(1)
            .samples(self.msaa_samples)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT)
            .tiling(vk::ImageTiling::OPTIMAL);
        let allocation_info = vk_mem_erupt::AllocationCreateInfo::default();
        let (draw_pass_color_image, draw_pass_color_image_allocation, _) = self.allocator.as_ref().unwrap().create_image(&draw_pass_image_info, &allocation_info).unwrap();
        self.albedo_image = Some(draw_pass_color_image);
        self.albedo_image_allocation = Some(draw_pass_color_image_allocation);


        let draw_pass_image_view_info = vk::ImageViewCreateInfoBuilder::new()
            .image(self.albedo_image.unwrap())
            .view_type(vk::ImageViewType::_2D)
            .format(color_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: self.mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            })
            .flags(vk::ImageViewCreateFlags::empty());
        self.albedo_image_view = Some(unsafe{
            self.device.as_ref().unwrap().create_image_view(&draw_pass_image_view_info, None)
        }.unwrap());
    }

    pub(crate) fn create_array_image_resources(&mut self) {
        self.images_3d
            .push(self.load_image_sequence(&Path::new("models/planet/drawn-globe/mountain_drawn")));
        self.images_3d
            .push(self.load_image_sequence(&Path::new("models/planet/drawn-globe/grass")));
        self.images_3d
            .push(self.load_image_sequence(&Path::new("models/planet/drawn-globe/trees")));
        self.images_3d
            .push(self.load_image_sequence(&Path::new("models/planet/drawn-globe/wavygrass")));
    }

    pub(crate) fn create_normals_image(&mut self){
        let format = vk::Format::R32G32B32A32_SFLOAT;
        let width = self.surface_capabilities.unwrap().current_extent.width;
        let height = self.surface_capabilities.unwrap().current_extent.height;
        let image_create_info = vk::ImageCreateInfoBuilder::new()
            .extent(
                *Extent3DBuilder::new()
                .width(width)
                .height(height)
                .depth(1))
            .image_type(vk::ImageType::_2D)
            .array_layers(1)
            .flags(vk::ImageCreateFlags::empty())
            .format(format)
            .mip_levels(1)
            .samples(self.msaa_samples)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT)
            .tiling(vk::ImageTiling::OPTIMAL);
        let allocation_info = vk_mem_erupt::AllocationCreateInfo::default();
        let (image, allocation, _) = self.allocator.as_ref().unwrap().create_image(&image_create_info, &allocation_info).unwrap();


        let image_view_create_info = vk::ImageViewCreateInfoBuilder::new()
            .image(image)
            .view_type(vk::ImageViewType::_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: self.mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            })
            .flags(vk::ImageViewCreateFlags::empty());
        let image_view = unsafe{
            self.device.as_ref().unwrap().create_image_view(&image_view_create_info, None)
        }.unwrap();

        self.normals_image = Some(
            CombinedImage{
                image,
                image_view,
                allocation,
                width,
                height,
            }
        )
    }
    pub(crate) fn create_rough_metal_ao_image(&mut self){
        let format = vk::Format::R8G8B8A8_UNORM;
        let width = self.surface_capabilities.unwrap().current_extent.width;
        let height = self.surface_capabilities.unwrap().current_extent.height;
        let image_create_info = vk::ImageCreateInfoBuilder::new()
            .extent(
                *Extent3DBuilder::new()
                .width(width)
                .height(height)
                .depth(1))
            .image_type(vk::ImageType::_2D)
            .array_layers(1)
            .flags(vk::ImageCreateFlags::empty())
            .format(format)
            .mip_levels(1)
            .samples(self.msaa_samples)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT)
            .tiling(vk::ImageTiling::OPTIMAL);
        let allocation_info = vk_mem_erupt::AllocationCreateInfo::default();
        let (image, allocation, _) = self.allocator.as_ref().unwrap().create_image(&image_create_info, &allocation_info).unwrap();


        let image_view_create_info = vk::ImageViewCreateInfoBuilder::new()
            .image(image)
            .view_type(vk::ImageViewType::_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: self.mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            })
            .flags(vk::ImageViewCreateFlags::empty());
        let image_view = unsafe{
            self.device.as_ref().unwrap().create_image_view(&image_view_create_info, None)
        }.unwrap();

        self.rough_metal_ao_image = Some(
            CombinedImage{
                image,
                image_view,
                allocation,
                width,
                height,
            }
        )
    }

    pub(crate) fn create_cubemap_resources(&mut self) {
        let cubemap_folders = [
            PathBuf::from("cubemap_space"),
            PathBuf::from("cubemap_fire"),
        ];
        for cubemap_folder in cubemap_folders {
            self.cubemap = Some(Cubemap::new(
                self,
                cubemap_folder.join("StandardCubeMap.hdr"),
            ));

            let base_cubemap = CombinedSampledImage::new(
                self,
                cubemap_folder.join("StandardCubeMap.hdr"),
                vk::ImageViewType::CUBE,
                vk::Format::R32G32B32A32_SFLOAT,
                false,
            )
            .unwrap();

            let target_cubemap = self.create_blank_cubemap(
                16,
                16,
                1,
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageLayout::GENERAL,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            );
            let combined_descriptors = [
                CombinedDescriptor {
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                    descriptor_info: DescriptorInfoData::Image {
                        image_view: base_cubemap.image_view,
                        sampler: Some(base_cubemap.sampler),
                        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    },
                },
                CombinedDescriptor {
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    descriptor_info: DescriptorInfoData::Image {
                        image_view: target_cubemap.image_view,
                        sampler: None,
                        layout: vk::ImageLayout::GENERAL,
                    },
                },
            ];
            println!("Running irradiance shader");
            self.run_arbitrary_compute_shader(
                self.load_shader("shaders/irradiance.spv".parse().unwrap()),
                1u32,
                &combined_descriptors,
                (
                    target_cubemap.width / 8 + u32::from(target_cubemap.width % 8 == 0),
                    target_cubemap.height / 8 + u32::from(target_cubemap.height % 8 == 0),
                    6,
                ),
            );

            let target_barrier = vk::ImageMemoryBarrierBuilder::new()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(target_cubemap.image)
                .subresource_range(
                    *vk::ImageSubresourceRangeBuilder::new()
                        .level_count(1)
                        .layer_count(6)
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .base_array_layer(0),
                );

            let command_buffer = self.begin_single_time_commands();
            unsafe {
                self.device.as_ref().unwrap().cmd_pipeline_barrier(
                    command_buffer,
                    Some(vk::PipelineStageFlags::ALL_COMMANDS),
                    Some(vk::PipelineStageFlags::ALL_COMMANDS),
                    None,
                    &[],
                    &[],
                    &[target_barrier],
                )
            };

            self.end_single_time_commands(command_buffer);

            self.irradiance_maps.push(target_cubemap);

            let roughness_mipmaps = 10;

            let target_cubemap = self.create_blank_cubemap(
                1024,
                1024,
                roughness_mipmaps,
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageLayout::GENERAL,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            );

            for i in 0..roughness_mipmaps {
                println!("Running environment shader for mip level {:}", i);

                let image_view_create_info = vk::ImageViewCreateInfoBuilder::new()
                    .image(target_cubemap.image)
                    .view_type(vk::ImageViewType::CUBE)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: i,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 6,
                    });

                let current_mip_image_view = unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .create_image_view(&image_view_create_info, None)
                }
                .unwrap();

                let combined_descriptors = [
                    CombinedDescriptor {
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                        descriptor_info: DescriptorInfoData::Image {
                            image_view: base_cubemap.image_view,
                            sampler: Some(base_cubemap.sampler),
                            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                    },
                    CombinedDescriptor {
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        descriptor_count: 1,
                        descriptor_info: DescriptorInfoData::Image {
                            image_view: current_mip_image_view,
                            sampler: None,
                            layout: vk::ImageLayout::GENERAL,
                        },
                    },
                ];

                self.run_arbitrary_compute_shader(
                    self.load_shader("shaders/environment.spv".parse().unwrap()),
                    i as f32 / (roughness_mipmaps - 1) as f32,
                    &combined_descriptors,
                    (
                        target_cubemap.width / 8 + u32::from(target_cubemap.width % 8 == 0),
                        target_cubemap.height / 8 + u32::from(target_cubemap.height % 8 == 0),
                        6,
                    ),
                );
            }

            let target_barrier = vk::ImageMemoryBarrierBuilder::new()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(target_cubemap.image)
                .subresource_range(
                    *vk::ImageSubresourceRangeBuilder::new()
                        .level_count(roughness_mipmaps)
                        .layer_count(6)
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .base_array_layer(0),
                );

            let command_buffer = self.begin_single_time_commands();
            unsafe {
                self.device.as_ref().unwrap().cmd_pipeline_barrier(
                    command_buffer,
                    Some(vk::PipelineStageFlags::ALL_COMMANDS),
                    Some(vk::PipelineStageFlags::ALL_COMMANDS),
                    None,
                    &[],
                    &[],
                    &[target_barrier],
                )
            };

            self.end_single_time_commands(command_buffer);

            self.environment_maps.push(target_cubemap);
        }

        self.brdf_lut = CombinedSampledImage::new(
            self,
            PathBuf::from("brdf_lut.png"),
            vk::ImageViewType::_2D,
            vk::Format::R8G8B8A8_UNORM,
            false,
        );
    }

    pub(crate) fn create_command_pool(&mut self) {
        let command_pool_create_info = vk::CommandPoolCreateInfoBuilder::new()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(self.main_queue_index.unwrap());
        self.command_pool = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_command_pool(&command_pool_create_info, None)
            }
            .unwrap(),
        );
    }

    
    pub(crate) fn cleanup_swapchain(&mut self) {
        unsafe {
            self.device.as_ref().unwrap().device_wait_idle().unwrap();
            for &framebuffer in &self.framebuffers{
                self.device
                .as_ref()
                .unwrap()
                .destroy_framebuffer(Some(framebuffer), None);

            }
            self.device
                .as_ref()
                .unwrap()
                .destroy_image_view(Some(self.depth_image_view.unwrap()), None);
            self.device
                .as_ref()
                .unwrap()
                .destroy_image(Some(self.depth_image.unwrap()), None);
            self.device
                .as_ref()
                .unwrap()
                .free_memory(Some(self.depth_image_memory.unwrap()), None);

            self.device.as_ref().unwrap().free_command_buffers(
                self.command_pool.unwrap(),
                &[self.command_buffer.unwrap()],
            );

            for pipeline in &self.graphics_pipelines {
                self.device
                    .as_ref()
                    .unwrap()
                    .destroy_pipeline(Some(*pipeline), None)
            }
            self.device
                .as_ref()
                .unwrap()
                .destroy_pipeline(self.ui_data.pipeline, None);
        }

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_pipeline_layout(Some(self.pipeline_layout.unwrap()), None)
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_render_pass(Some(self.render_pass.unwrap()), None)
        };
        self.swapchain_image_views
            .iter()
            .for_each(|image_view| unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .destroy_image_view(Some(*image_view), None)
            });
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_swapchain_khr(Some(self.swapchain.unwrap()), None)
        };
        self.swapchain_created = false;
    }

    pub(crate) fn destroy_vertex_buffer(&mut self) {
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(Some(self.vertex_buffer.unwrap()), None);
            self.device
                .as_ref()
                .unwrap()
                .free_memory(Some(self.vertex_buffer_memory.unwrap()), None);
        }
        self.vertex_buffer = None;
        self.vertex_buffer_memory = None;
    }
    pub(crate) fn destroy_index_buffer(&mut self) {
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(Some(self.index_buffer.unwrap()), None);
            self.device
                .as_ref()
                .unwrap()
                .free_memory(Some(self.index_buffer_memory.unwrap()), None);
        }
        self.index_buffer = None;
        self.index_buffer_memory = None;
    }

    pub fn get_projection(&self, zoom: f64) -> nalgebra::Perspective3<f64> {
        let surface_width = self.surface_capabilities.unwrap().current_extent.width as f64;
        let surface_height = self.surface_capabilities.unwrap().current_extent.height as f64;
        let aspect_ratio = surface_width / surface_height;

        return nalgebra::Perspective3::new(
            aspect_ratio,
            90.0f64.to_radians() * zoom,
            0.1,
            1000.0,
        );
    }
    pub fn transfer_data_to_gpu(&mut self) {
        let random: [f32; NUM_RANDOM] = get_random_vector(&mut self.rng, NUM_RANDOM)
            .try_into()
            .unwrap();
        for i in 0..NUM_RANDOM {
            self.uniform_buffer_object.random[i] =
                Vector4::new(random[i], random[i], random[i], random[i]);
        }
        self.uniform_buffer_object.screen_size = Vector2::new(
            self.surface_capabilities.unwrap().current_extent.width as f32,
            self.surface_capabilities.unwrap().current_extent.height as f32,
        );

        for i in 0..self.uniform_buffer_pointers.len() {
            unsafe {
                self.uniform_buffer_pointers[i].copy_from_nonoverlapping(
                    &self.uniform_buffer_object as *const UniformBufferObject as *const u8,
                    std::mem::size_of::<UniformBufferObject>(),
                );
            };
        }
    }

}
