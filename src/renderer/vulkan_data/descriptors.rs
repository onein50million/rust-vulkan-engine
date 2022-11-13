use std::mem::size_of;

use crate::{support::{NUM_MODELS, UniformBufferObject, ShaderStorageBufferObject, PostProcessUniformBufferObject}};
use erupt::vk;

use super::VulkanData;

impl VulkanData{
    pub(crate) fn create_descriptor_pool(&mut self) {
        const POOL_SIZE: u32 = 64; //eh probably enough for now

        let pool_sizes = [
            vk::DescriptorPoolSizeBuilder::new()
                .descriptor_count(1)
                ._type(vk::DescriptorType::UNIFORM_BUFFER),
            vk::DescriptorPoolSizeBuilder::new()
                .descriptor_count(self.swapchain_image_views.len() as u32)
                ._type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            vk::DescriptorPoolSizeBuilder::new()
                .descriptor_count(1)
                ._type(vk::DescriptorType::STORAGE_BUFFER),
        ];

        let pool_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .pool_sizes(&pool_sizes)
            .max_sets(POOL_SIZE);

        self.descriptor_pool = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_descriptor_pool(&pool_info, None)
            }
            .unwrap(),
        );
    }

    pub(crate) fn create_descriptor_set_layout(&mut self) {
        {
            let ubo_layout_binding = vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
            let sampler_layout_binding = vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(1)
                .descriptor_count(NUM_MODELS as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

            let layout_bindings = [
                ubo_layout_binding,
                sampler_layout_binding,
                vk::DescriptorSetLayoutBindingBuilder::new()
                    .binding(2)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBindingBuilder::new()
                    .binding(3)
                    .descriptor_count(NUM_MODELS as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBindingBuilder::new()
                    .binding(4)
                    .descriptor_count(NUM_MODELS as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBindingBuilder::new()
                    .binding(5)
                    .descriptor_count(NUM_MODELS as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBindingBuilder::new()
                    .binding(6)
                    .descriptor_count(NUM_MODELS as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBindingBuilder::new()
                    .binding(7)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBindingBuilder::new()
                    .binding(8)
                    .descriptor_count(NUM_MODELS as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBindingBuilder::new()
                    .binding(9)
                    .descriptor_count(NUM_MODELS as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
                // vk::DescriptorSetLayoutBindingBuilder::new()
                //     .binding(10)
                //     .descriptor_count(NUM_PLANET_TEXTURES as u32)
                //     .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                //     .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBindingBuilder::new()
                    .binding(11)
                    .descriptor_count(NUM_MODELS as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            ];

            let layout_info =
                vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&layout_bindings);

            self.descriptor_set_layout = Some(
                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .create_descriptor_set_layout(&layout_info, None)
                }
                .unwrap(),
            )
    }
    {
        let layout_bindings = &[
            vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(3)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(4)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(5)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        ];

        let layout_info =
            vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(layout_bindings);

        self.postprocess_descriptor_set_layout = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_descriptor_set_layout(&layout_info, None)
            }
            .unwrap(),
        )
    }

    }

    pub(crate) fn create_descriptor_sets(&mut self) {
        let layouts = [self.descriptor_set_layout.unwrap()];
        let allocate_info = vk::DescriptorSetAllocateInfoBuilder::new()
            .descriptor_pool(self.descriptor_pool.unwrap())
            .set_layouts(&layouts);

        self.descriptor_sets = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .allocate_descriptor_sets(&allocate_info)
            }
            .unwrap(),
        );
        let postprocess_layouts = [self.postprocess_descriptor_set_layout.unwrap()];
        let postprocess_allocate_info = vk::DescriptorSetAllocateInfoBuilder::new()
            .descriptor_pool(self.descriptor_pool.unwrap())
            .set_layouts(&postprocess_layouts);

        self.postprocess_descriptor_sets = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .allocate_descriptor_sets(&postprocess_allocate_info)
            }
            .unwrap(),
        );
        println!(
            "Num descriptor sets: {:?}",
            self.descriptor_sets.as_ref().unwrap().len()
        );
    }

    pub fn update_descriptor_sets(&mut self) {
        println!(
            "Descriptor set count when updating descriptor sets: {:}",
            self.descriptor_sets.as_ref().unwrap().len()
        );
        self.descriptor_sets
            .as_ref()
            .unwrap()
            .into_iter()
            .for_each(|descriptor_set| {
                let buffer_infos = vec![vk::DescriptorBufferInfoBuilder::new()
                    .buffer(self.uniform_buffers[0])
                    .offset(0)
                    .range((std::mem::size_of::<UniformBufferObject>()) as vk::DeviceSize)];
                let mut albedo_infos = vec![];
                let mut normal_infos = vec![];
                let mut roughness_infos = vec![];

                for texture in &self.textures {
                    let albedo = texture.albedo.as_ref().unwrap_or(
                        self.fallback_texture
                            .as_ref()
                            .unwrap()
                            .albedo
                            .as_ref()
                            .unwrap(),
                    );
                    albedo_infos.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_view(albedo.image_view)
                            .sampler(albedo.sampler)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    );

                    let normal = texture.normal.as_ref().unwrap_or(
                        self.fallback_texture
                            .as_ref()
                            .unwrap()
                            .normal
                            .as_ref()
                            .unwrap(),
                    );

                    normal_infos.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_view(normal.image_view)
                            .sampler(normal.sampler)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    );
                    let roughness = texture.roughness_metalness_ao.as_ref().unwrap_or(
                        self.fallback_texture
                            .as_ref()
                            .unwrap()
                            .roughness_metalness_ao
                            .as_ref()
                            .unwrap(),
                    );

                    roughness_infos.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_view(roughness.image_view)
                            .sampler(roughness.sampler)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    );
                }

                let textures_left = NUM_MODELS - self.textures.len();
                for _ in 0..textures_left {
                    albedo_infos.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(self.fallback_texture.as_ref().unwrap().albedo.as_ref().unwrap().image_view)
                            .sampler(self.fallback_texture.as_ref().unwrap().albedo.as_ref().unwrap().sampler),
                    );
                    let normal = self
                        .fallback_texture
                        .as_ref()
                        .unwrap()
                        .normal
                        .as_ref()
                        .unwrap();
                    normal_infos.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(normal.image_view)
                            .sampler(normal.sampler),
                    );

                    let roughness = self
                        .fallback_texture
                        .as_ref()
                        .unwrap()
                        .roughness_metalness_ao
                        .as_ref()
                        .unwrap();

                    roughness_infos.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_view(roughness.image_view)
                            .sampler(roughness.sampler)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    );
                }
                let mut cubemap_infos = vec![];
                for cubemap in &self.cubemaps {
                    cubemap_infos.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_view(cubemap.image_view)
                            .sampler(cubemap.sampler)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    );
                }
                let cubemaps_left = NUM_MODELS - cubemap_infos.len();
                for _ in 0..cubemaps_left {
                    cubemap_infos.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(self.cubemaps[0].image_view)
                            .sampler(self.cubemaps[0].sampler),
                    );
                }
                let mut irradiance_infos = vec![];
                for cubemap in &self.irradiance_maps {
                    irradiance_infos.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_view(cubemap.image_view)
                            .sampler(cubemap.sampler)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    );
                }
                let cubemaps_left = NUM_MODELS - irradiance_infos.len();
                for _ in 0..cubemaps_left {
                    irradiance_infos.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(self.irradiance_maps[0].image_view)
                            .sampler(self.irradiance_maps[0].sampler),
                    );
                }

                let brdf_infos = vec![vk::DescriptorImageInfoBuilder::new()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(self.brdf_lut.as_ref().unwrap().image_view)
                    .sampler(self.brdf_lut.as_ref().unwrap().sampler)];

                let mut environment_infos = vec![];
                for cubemap in &self.environment_maps {
                    environment_infos.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_view(cubemap.image_view)
                            .sampler(cubemap.sampler)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    );
                }
                let cubemaps_left = NUM_MODELS - environment_infos.len();
                for _ in 0..cubemaps_left {
                    environment_infos.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(self.environment_maps[0].image_view)
                            .sampler(self.environment_maps[0].sampler),
                    );
                }

                let mut cpu_image_infos = vec![];
                for cpu_image in &self.cpu_images {
                    cpu_image_infos.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_view(cpu_image.image.image_view)
                            .sampler(cpu_image.image.sampler)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    );
                }
                let cpu_images_left = NUM_MODELS - cpu_image_infos.len();
                for _ in 0..cpu_images_left {
                    cpu_image_infos.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(self.cpu_images[0].image.image_view)
                            .sampler(self.cpu_images[0].image.sampler),
                    );
                }

                let bone_ssbo_infos = [vk::DescriptorBufferInfoBuilder::new()
                    .buffer(self.storage_buffer.unwrap())
                    .range(size_of::<ShaderStorageBufferObject>() as vk::DeviceSize)
                    .offset(0)];

                let mut images_3d_info = vec![];
                for image_3d in &self.images_3d {
                    images_3d_info.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_view(image_3d.image_view)
                            .sampler(image_3d.sampler)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    );
                }
                let images_3d_left = NUM_MODELS - images_3d_info.len();
                for _ in 0..images_3d_left {
                    images_3d_info.push(
                        vk::DescriptorImageInfoBuilder::new()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(self.images_3d[0].image_view)
                            .sampler(self.images_3d[0].sampler),
                    );
                }

                // let mut planet_textures_info = vec![];

                // for planet_texture in &self.planet_textures {
                //     planet_textures_info.push(
                //         vk::DescriptorImageInfoBuilder::new()
                //             .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                //             .image_view(planet_texture.image_view)
                //             .sampler(planet_texture.sampler),
                //     );
                // }
                // for _ in 0..(NUM_PLANET_TEXTURES - self.planet_textures.len()) {
                //     planet_textures_info.push(
                //         vk::DescriptorImageInfoBuilder::new()
                //             .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                //             .image_view(self.planet_textures[0].image_view)
                //             .sampler(self.planet_textures[0].sampler),
                //     );
                // }

                let mut descriptor_writes = vec![
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(*descriptor_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&buffer_infos),
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(*descriptor_set)
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&albedo_infos),
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(*descriptor_set)
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&bone_ssbo_infos),
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(*descriptor_set)
                        .dst_binding(3)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&cubemap_infos),
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(*descriptor_set)
                        .dst_binding(4)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&normal_infos),
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(*descriptor_set)
                        .dst_binding(5)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&roughness_infos),
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(*descriptor_set)
                        .dst_binding(6)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&irradiance_infos),
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(*descriptor_set)
                        .dst_binding(7)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&brdf_infos),
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(*descriptor_set)
                        .dst_binding(8)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&environment_infos),
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(*descriptor_set)
                        .dst_binding(9)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&cpu_image_infos),
                ];

                // descriptor_writes.push(
                //     vk::WriteDescriptorSetBuilder::new()
                //         .dst_set(*descriptor_set)
                //         .dst_binding(10)
                //         .dst_array_element(0)
                //         .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                //         .image_info(&planet_textures_info),
                // );
                descriptor_writes.push(
                    vk::WriteDescriptorSetBuilder::new()
                        .dst_set(*descriptor_set)
                        .dst_binding(11)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&images_3d_info),
                );

                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .update_descriptor_sets(&descriptor_writes, &[]);

                }
            });

            for &descriptor_set in self.postprocess_descriptor_sets.as_ref().unwrap(){
                unsafe{
                    self.device
                    .as_ref()
                    .unwrap()
                    .update_descriptor_sets(&[
                        vk::WriteDescriptorSetBuilder::new()
                            .dst_set(descriptor_set)
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                            .image_info(&[
                                vk::DescriptorImageInfoBuilder::new()
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image_view(self.albedo_image_view.unwrap())
                            ]),
                        vk::WriteDescriptorSetBuilder::new()
                            .dst_set(descriptor_set)
                            .dst_binding(1)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                            .image_info(&[
                                vk::DescriptorImageInfoBuilder::new()
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image_view(self.depth_image_view.unwrap())
                            ]),
                        vk::WriteDescriptorSetBuilder::new()
                            .dst_set(descriptor_set)
                            .dst_binding(2)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                            .image_info(&[
                                vk::DescriptorImageInfoBuilder::new()
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image_view(self.normals_image.as_ref().unwrap().image_view)
                            ]),
                        vk::WriteDescriptorSetBuilder::new()
                            .dst_set(descriptor_set)
                            .dst_binding(3)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                            .image_info(&[
                                vk::DescriptorImageInfoBuilder::new()
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image_view(self.rough_metal_ao_image.as_ref().unwrap().image_view)
                            ]),
                        vk::WriteDescriptorSetBuilder::new()
                            .dst_set(descriptor_set)
                            .dst_binding(4)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(&[
                                vk::DescriptorImageInfoBuilder::new()
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image_view(self.voxel_sdf.as_ref().unwrap().image_view)
                                .sampler(self.voxel_sdf.as_ref().unwrap().sampler)
                            ]),
                            vk::WriteDescriptorSetBuilder::new()
                            .dst_set(descriptor_set)
                            .dst_binding(5)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&[
                                vk::DescriptorBufferInfoBuilder::new()
                                    .buffer(self.post_process_ubo.as_ref().unwrap().buffer)
                                    .offset(0)
                                    .range(size_of::<PostProcessUniformBufferObject>() as u64)
                            ]),

                    ], &[]);    
                }
            }

    }
}