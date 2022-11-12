use erupt::vk;

use super::{combination_types::CombinedSampledImage, vulkan_data::VulkanData};

pub(crate) struct CpuImage {
    pub(crate) image: CombinedSampledImage,
    pub(crate) allocation_info: vk_mem_erupt::AllocationInfo,
}
impl CpuImage {
    pub(crate) fn new(vulkan_data: &VulkanData, width: u32, height: u32) -> Self {
        let image_info = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8_UINT)
            .tiling(vk::ImageTiling::LINEAR)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlagBits::_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_create_info = vk_mem_erupt::AllocationCreateInfo {
            usage: vk_mem_erupt::MemoryUsage::GpuOnly,
            flags: vk_mem_erupt::AllocationCreateFlags::MAPPED,
            required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ..Default::default()
        };

        let (image, allocation, allocation_info) = vulkan_data
            .allocator
            .as_ref()
            .unwrap()
            .create_image(&image_info, &allocation_create_info)
            .unwrap();

        let view_info = vk::ImageViewCreateInfoBuilder::new()
            .image(image)
            .view_type(vk::ImageViewType::_2D)
            .format(vk::Format::R8_UINT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let image_view = unsafe {
            vulkan_data
                .device
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
            vulkan_data
                .device
                .as_ref()
                .unwrap()
                .create_sampler(&sampler_info, None)
                .unwrap()
        };

        let command_buffer = vulkan_data.begin_single_time_commands();

        let barrier = vk::ImageMemoryBarrierBuilder::new()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(view_info.subresource_range)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        unsafe {
            vulkan_data.device.as_ref().unwrap().cmd_pipeline_barrier(
                command_buffer,
                Some(vk::PipelineStageFlags::TRANSFER),
                Some(vk::PipelineStageFlags::FRAGMENT_SHADER),
                None,
                &[],
                &[],
                &[barrier],
            )
        };

        vulkan_data.end_single_time_commands(command_buffer);

        return Self {
            image: CombinedSampledImage {
                image,
                image_view,
                sampler,
                allocation,
                width,
                height,
            },
            allocation_info,
        };
    }

    pub(crate) fn get_data(&self) -> Vec<u8> {
        unsafe {
            std::slice::from_raw_parts(
                self.allocation_info.get_mapped_data(),
                (self.image.width * self.image.height) as usize,
            )
            .try_into()
            .unwrap()
        }
    }

    pub(crate) fn write_data(&mut self, data: Vec<u8>) {
        unsafe {
            self.allocation_info
                .get_mapped_data()
                .copy_from_nonoverlapping(
                    data.as_ptr(),
                    data.len()
                        .max((self.image.width * self.image.height) as usize),
                )
        }
    }
}
