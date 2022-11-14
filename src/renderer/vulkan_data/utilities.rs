use std::{mem::size_of, ffi::CString};

use erupt::{vk, DeviceLoader, vk1_0::{DescriptorImageInfoBuilder, DescriptorBufferInfoBuilder}};
use rand::Rng;

use crate::{renderer::{combination_types::{CombinedSampledImage, CombinedDescriptor, DescriptorInfoData}}};

use super::VulkanData;

pub fn get_random_vector(rng: &mut rand::rngs::ThreadRng, length: usize) -> Vec<f32> {
    let mut vector = Vec::new();
    for _ in 0..length {
        vector.push(rng.gen::<f32>());
    }
    return vector;
}

impl VulkanData{
    pub(crate) fn get_surface_support(&mut self) {
        unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_surface_support_khr(
                    self.physical_device.unwrap(),
                    self.main_queue_index.unwrap(),
                    self.surface.unwrap(),
                )
        }
        .unwrap();
    }

    pub(crate) fn get_surface_capabilities(&mut self) {
        self.surface_capabilities = Some(
            unsafe {
                self.instance
                    .as_ref()
                    .unwrap()
                    .get_physical_device_surface_capabilities_khr(
                        self.physical_device.unwrap(),
                        self.surface.unwrap(),
                    )
            }
            .unwrap(),
        );
    }

    pub(crate) fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> u32 {
        let memory_properties = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_memory_properties(self.physical_device.unwrap())
        };

        let mut i: u32 = 0;
        while i < memory_properties.memory_type_count {
            if type_filter & (1 << i) != 0
                && memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return i;
            }

            i += 1;
        }
        panic!("Failed to find suitable memory type");
    }

    pub(crate) fn load_shader(&self, path: std::path::PathBuf) -> vk::ShaderModule {
        let shader_code = erupt::utils::decode_spv(&std::fs::read(path).unwrap()).unwrap();
        return VulkanData::create_shader_module(self.device.as_ref().unwrap(), shader_code);
    }
    #[deprecated(note="Use `load_shader` instead")]
    pub(crate) fn load_shaders(&mut self) {
        let vert_shader_file = std::fs::read("shaders/vert.spv").unwrap();
        let frag_shader_file = std::fs::read("shaders/frag.spv").unwrap();
        let vert_shader_code = erupt::utils::decode_spv(&vert_shader_file).unwrap();
        let frag_shader_code = erupt::utils::decode_spv(&frag_shader_file).unwrap();
        self.vert_shader_module = Some(VulkanData::create_shader_module(
            &self.device.as_ref().unwrap(),
            vert_shader_code,
        ));
        self.frag_shader_module = Some(VulkanData::create_shader_module(
            &self.device.as_ref().unwrap(),
            frag_shader_code,
        ));
    }
    pub(crate) fn create_blank_cubemap(
        &self,
        width: u32,
        height: u32,
        mip_levels: u32,
        format: vk::Format,
        final_layout: vk::ImageLayout,
        usage: vk::ImageUsageFlags,
    ) -> CombinedSampledImage {
        let image_info = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_2D)
            .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(6)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .samples(vk::SampleCountFlagBits::_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_info = vk_mem_erupt::AllocationCreateInfo {
            usage: vk_mem_erupt::MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let (image, allocation, _) = self
            .allocator
            .as_ref()
            .unwrap()
            .create_image(&image_info, &allocation_info)
            .unwrap();

        let image_view_create_info = vk::ImageViewCreateInfoBuilder::new()
            .image(image)
            .view_type(vk::ImageViewType::CUBE)
            .format(image_info.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 6,
            });
        let image_view = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_image_view(&image_view_create_info, None)
        }
        .unwrap();

        let sampler_create_info = vk::SamplerCreateInfoBuilder::new()
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
                .create_sampler(&sampler_create_info, None)
        }
        .unwrap();

        let subresource_range = vk::ImageSubresourceRangeBuilder::new()
            .level_count(mip_levels)
            .layer_count(6)
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .base_array_layer(0);
        let barrier = vk::ImageMemoryBarrierBuilder::new()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(final_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(*subresource_range);

        let command_buffer = self.begin_single_time_commands();
        unsafe {
            self.device.as_ref().unwrap().cmd_pipeline_barrier(
                command_buffer,
                Some(vk::PipelineStageFlags::ALL_COMMANDS),
                Some(vk::PipelineStageFlags::ALL_COMMANDS),
                None,
                &[],
                &[],
                &[barrier],
            )
        };

        self.end_single_time_commands(command_buffer);

        return CombinedSampledImage {
            image,
            image_view,
            sampler,
            allocation,
            width,
            height,
        };
    }
    fn find_supported_format(
        &self,
        candidates: Vec<vk::Format>,
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> vk::Format {
        return candidates
            .into_iter()
            .find(|format| {
                let properties = unsafe {
                    self.instance
                        .as_ref()
                        .unwrap()
                        .get_physical_device_format_properties(
                            self.physical_device.unwrap(),
                            *format,
                        )
                };
                match tiling {
                    vk::ImageTiling::LINEAR => {
                        return properties.linear_tiling_features.contains(features);
                    }
                    vk::ImageTiling::OPTIMAL => {
                        return properties.optimal_tiling_features.contains(features);
                    }
                    _ => panic!("No supported format or something idk I'm tired"),
                }
            })
            .unwrap();
    }

    pub(crate) fn find_depth_format(&self) -> vk::Format {
        return self.find_supported_format(
            vec![
                vk::Format::D32_SFLOAT,
                vk::Format::D24_UNORM_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        );
    }

    pub fn create_shader_module(device: &DeviceLoader, spv_code: Vec<u32>) -> vk::ShaderModule {
        let shader_module_create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&spv_code);
        unsafe { device.create_shader_module(&shader_module_create_info, None) }.unwrap()
    }

    pub fn run_arbitrary_compute_shader<PushConstantType>(
        &self,
        shader_module: vk::ShaderModule,
        push_constants: PushConstantType,
        combined_descriptors: &[CombinedDescriptor],
        group_count: (u32, u32, u32),
    ) {
        let device = self.device.as_ref().unwrap();

        let descriptor_set_layout_bindings: Vec<_> = combined_descriptors
            .iter()
            .enumerate()
            .map(|(index, combined_descriptor)| {
                vk::DescriptorSetLayoutBindingBuilder::new()
                    .binding(index as u32)
                    .descriptor_count(combined_descriptor.descriptor_count)
                    .descriptor_type(combined_descriptor.descriptor_type)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
            .bindings(&descriptor_set_layout_bindings);

        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_info, None) }
                .unwrap();

        let pool_sizes: Vec<_> = combined_descriptors
            .iter()
            .enumerate()
            .map(|(_index, combined_descriptor)| {
                vk::DescriptorPoolSizeBuilder::new()
                    .descriptor_count(combined_descriptor.descriptor_count)
                    ._type(combined_descriptor.descriptor_type)
            })
            .collect();
        let desciptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfoBuilder::new()
                    .pool_sizes(&pool_sizes)
                    .max_sets(pool_sizes.len() as u32), //TODO: This might be off when there are many descriptors in each set
                None,
            )
        }
        .unwrap();

        let descriptor_set_layouts = [descriptor_set_layout];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfoBuilder::new()
            .descriptor_pool(desciptor_pool)
            .set_layouts(&descriptor_set_layouts);

        let descriptor_set =
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }.unwrap()[0];

        enum DescriptorInfoBuilders<'a> {
            Image(Vec<DescriptorImageInfoBuilder<'a>>),
            Buffer(Vec<DescriptorBufferInfoBuilder<'a>>),
        }

        let info_builders: Vec<_> = combined_descriptors
            .iter()
            .enumerate()
            .map(
                |(_index, combined_descriptor)| match combined_descriptor.descriptor_info {
                    DescriptorInfoData::Image {
                        image_view,
                        sampler,
                        layout,
                    } => {
                        let mut image_info = vk::DescriptorImageInfoBuilder::new()
                            .image_view(image_view)
                            .image_layout(layout);
                        match sampler {
                            None => {}
                            Some(sampler) => image_info = image_info.sampler(sampler),
                        }
                        DescriptorInfoBuilders::Image(vec![image_info])
                    }
                    DescriptorInfoData::Buffer { buffer, range } => {
                        DescriptorInfoBuilders::Buffer(vec![vk::DescriptorBufferInfoBuilder::new()
                            .buffer(buffer)
                            .range(range)])
                    }
                },
            )
            .collect();

        let descriptor_writes: Vec<_> = info_builders
            .iter()
            .enumerate()
            .map(|(index, info_builder)| match info_builder {
                DescriptorInfoBuilders::Image(builder) => vk::WriteDescriptorSetBuilder::new()
                    .dst_set(descriptor_set)
                    .dst_binding(index as u32)
                    .dst_array_element(0)
                    .descriptor_type(descriptor_set_layout_bindings[index].descriptor_type)
                    .image_info(&builder),
                DescriptorInfoBuilders::Buffer(builder) => vk::WriteDescriptorSetBuilder::new()
                    .dst_set(descriptor_set)
                    .dst_binding(index as u32)
                    .dst_array_element(0)
                    .descriptor_type(descriptor_set_layout_bindings[index].descriptor_type)
                    .buffer_info(&builder),
            })
            .collect();

        unsafe {
            device.update_descriptor_sets(&descriptor_writes, &[]);
        }
        let push_constant_range = vk::PushConstantRangeBuilder::new()
            .offset(0)
            .size(size_of::<PushConstantType>() as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let push_constant_ranges = [push_constant_range];
        let descriptor_set_layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .push_constant_ranges(&push_constant_ranges)
            .set_layouts(&descriptor_set_layouts);

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }.unwrap();

        let main_string = CString::new("main").unwrap();
        let pipeline_stage_info = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::COMPUTE)
            .module(shader_module)
            .name(&main_string);

        let pipeline_infos = [vk::ComputePipelineCreateInfoBuilder::new()
            .stage(*pipeline_stage_info)
            .layout(pipeline_layout)];

        let pipeline =
            unsafe { device.create_compute_pipelines(None, &pipeline_infos, None) }.unwrap()[0];

        let command_buffer = self.begin_single_time_commands();
        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline)
        };

        unsafe {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            )
        };

        unsafe {
            self.device.as_ref().unwrap().cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                size_of::<PushConstantType>() as u32,
                &push_constants as *const _ as *const _,
            );
        };

        unsafe { device.cmd_dispatch(command_buffer, group_count.0, group_count.1, group_count.2) };
        self.end_single_time_commands(command_buffer);
    }

    pub(crate) fn generate_mipmaps(
        &self,
        image: vk::Image,
        texture_width: u32,
        texture_height: u32,
        mip_levels: u32,
        layer_count: u32,
    ) {
        let command_buffer = self.begin_single_time_commands();
        let mut mip_width = texture_width as i32;
        let mut mip_height = texture_height as i32;

        let mut i = 1;
        while i < mip_levels && mip_height > 2 && mip_width > 2 {
            let barriers = [vk::ImageMemoryBarrierBuilder::new()
                .image(image)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: i - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count,
                })
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::all())
                .dst_access_mask(vk::AccessFlags::all())];
            unsafe {
                self.device.as_ref().unwrap().cmd_pipeline_barrier(
                    command_buffer,
                    Some(vk::PipelineStageFlags::ALL_COMMANDS),
                    Some(vk::PipelineStageFlags::ALL_COMMANDS),
                    Some(vk::DependencyFlags::empty()),
                    &[],
                    &[],
                    &barriers,
                )
            };

            let blit = vk::ImageBlitBuilder::new()
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ])
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i - 1,
                    base_array_layer: 0,
                    layer_count,
                })
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width / 2,
                        y: mip_height / 2,
                        z: 1,
                    },
                ])
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count,
                });
            unsafe {
                self.device.as_ref().unwrap().cmd_blit_image(
                    command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                )
            };
            let barriers = [vk::ImageMemoryBarrierBuilder::new()
                .image(image)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: i - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count,
                })
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::all())
                .dst_access_mask(vk::AccessFlags::all())];

            unsafe {
                self.device.as_ref().unwrap().cmd_pipeline_barrier(
                    command_buffer,
                    Some(vk::PipelineStageFlags::ALL_COMMANDS),
                    Some(vk::PipelineStageFlags::ALL_COMMANDS),
                    Some(vk::DependencyFlags::empty()),
                    &[],
                    &[],
                    &barriers,
                )
            };
            mip_width /= 2;
            mip_height /= 2;
            i += 1;
        }

        let barriers = [vk::ImageMemoryBarrierBuilder::new()
            .image(image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count,
            })
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::all())
            .dst_access_mask(vk::AccessFlags::all())];

        unsafe {
            self.device.as_ref().unwrap().cmd_pipeline_barrier(
                command_buffer,
                Some(vk::PipelineStageFlags::ALL_COMMANDS),
                Some(vk::PipelineStageFlags::ALL_COMMANDS),
                Some(vk::DependencyFlags::empty()),
                &[],
                &[],
                &barriers,
            )
        };

        self.end_single_time_commands(command_buffer);
    }

    pub fn begin_single_time_commands(&self) -> vk::CommandBuffer {
        let allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool.unwrap())
            .command_buffer_count(1);
        let command_buffer = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .allocate_command_buffers(&allocate_info)
        }
        .unwrap()[0];
        let begin_info = vk::CommandBufferBeginInfoBuilder::new()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .begin_command_buffer(command_buffer, &begin_info)
        }
        .unwrap();
        return command_buffer;
    }

    pub fn end_single_time_commands(&self, command_buffer: vk::CommandBuffer) {
        let command_buffers = [command_buffer];
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .end_command_buffer(command_buffer)
        }
        .unwrap();
        let submit_info = vk::SubmitInfoBuilder::new().command_buffers(&command_buffers);
        unsafe {
            self.device.as_ref().unwrap().queue_submit(
                self.main_queue.unwrap(),
                &[submit_info],
                None,
            )
        }
        .unwrap();
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .queue_wait_idle(self.main_queue.unwrap())
        }
        .unwrap();

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .free_command_buffers(self.command_pool.unwrap(), &command_buffers)
        };
    }
    pub fn lazy_transition_image_layout(
        &self,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        subresource_range: vk::ImageSubresourceRange,
    ) {
        let command_buffer = self.begin_single_time_commands();

        let barrier = vk::ImageMemoryBarrierBuilder::new()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ)
            .dst_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ)
            .image(image)
            .subresource_range(subresource_range);

        unsafe {
            self.device.as_ref().unwrap().cmd_pipeline_barrier(
                command_buffer,
                Some(vk::PipelineStageFlags::ALL_COMMANDS),
                Some(vk::PipelineStageFlags::ALL_COMMANDS),
                None,
                &[],
                &[],
                &[barrier],
            )
        };

        self.end_single_time_commands(command_buffer);
    }
    #[deprecated(note="Too specialized")]
    pub(crate) fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspect_flags: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> vk::ImageView {
        let view_info = vk::ImageViewCreateInfoBuilder::new()
            .image(image)
            .view_type(vk::ImageViewType::_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            })
            .flags(vk::ImageViewCreateFlags::empty());

        return unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_image_view(&view_info, None)
        }
        .unwrap();
    }
    #[deprecated(note="Too specialized")]
    pub(crate) fn copy_buffer_to_image(&self, buffer: vk::Buffer, image: vk::Image, width: u32, height: u32) {
        let command_buffer = self.begin_single_time_commands();

        let region = vk::BufferImageCopyBuilder::new()
            .buffer_offset(0)
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
                width,
                height,
                depth: 1,
            });

        unsafe {
            self.device.as_ref().unwrap().cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            )
        };

        self.end_single_time_commands(command_buffer);
    }
    #[deprecated(note="Too specialized")]
    pub(crate) fn create_image_with_memory(
        &self,
        width: u32,
        height: u32,
        mip_levels: u32,
        num_samples: vk::SampleCountFlagBits,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> (vk::Image, vk::DeviceMemory) {
        let image_info = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .samples(num_samples)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_image(&image_info, None)
        }
        .unwrap();

        let memory_requirements = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_image_memory_requirements(image)
        };

        let allocate_info = vk::MemoryAllocateInfoBuilder::new()
            .allocation_size(memory_requirements.size)
            .memory_type_index(
                self.find_memory_type(memory_requirements.memory_type_bits, properties),
            );

        let image_memory = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .allocate_memory(&allocate_info, None)
        }
        .unwrap();

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .bind_image_memory(image, image_memory, 0)
        }
        .unwrap();
        return (image, image_memory);
    }


}