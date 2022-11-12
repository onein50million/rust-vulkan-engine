use std::{io::BufReader, fs::File};

use erupt::vk;
use image::{ImageFormat, GenericImageView};

use super::vulkan_data::VulkanData;

const MIP_LEVELS: u32 = 6;


pub enum DescriptorInfoData {
    Image {
        image_view: vk::ImageView,
        sampler: Option<vk::Sampler>,
        layout: vk::ImageLayout,
    },
    Buffer {
        buffer: vk::Buffer,
        range: vk::DeviceSize,
    },
}

pub struct CombinedDescriptor {
    //combines info needed for DescriptorSetLayout and WriteDescriptorSet in one time compute shader
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
    pub descriptor_info: DescriptorInfoData,
}

pub struct CombinedImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: vk_mem_erupt::Allocation,
    pub width: u32,
    pub height: u32,
}

pub struct CombinedSampledImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub allocation: vk_mem_erupt::Allocation,
    pub width: u32,
    pub height: u32,
}
impl CombinedSampledImage {
    pub fn new(
        vulkan_data: &VulkanData,
        path: std::path::PathBuf,
        view_type: vk::ImageViewType,
        image_format: vk::Format,
        generate_new_image: bool,
    ) -> Option<Self> {
        let mut width;
        let height;
        let layer_count;
        let mut bytes;
        let pixel_size;

        if !generate_new_image {
            println!("Loading image: {:?}", path);

            if !path.is_file() {
                println!("File not found: {:?}", path);
                return None;
            }
            let reader = image::io::Reader::open(path.clone()).unwrap();
            let format = reader.format().unwrap();
            let dynamic_image = reader.decode().unwrap();
            let color = dynamic_image.color();
            width = dynamic_image.width();
            height = dynamic_image.height();

            bytes = if format == ImageFormat::Hdr {
                let image =
                    image::codecs::hdr::HdrDecoder::new(BufReader::new(File::open(path).unwrap()))
                        .unwrap();

                let pixels = image.read_image_hdr().unwrap();

                let mut out_bytes = vec![];
                for pixel in pixels {
                    let mut pixel_bytes = vec![];
                    for channel in pixel.0 {
                        pixel_bytes.extend_from_slice(&channel.to_le_bytes())
                    }
                    pixel_bytes.extend_from_slice(&1.0_f32.to_be_bytes());

                    out_bytes.extend_from_slice(&pixel_bytes)
                }
                out_bytes
            } else {
                dynamic_image.into_rgba8().into_raw()
            };

            pixel_size = bytes.len() / (width * height) as usize;

            println!("channel count: {:}", color.channel_count());

            width = if view_type == vk::ImageViewType::CUBE {
                height
            } else {
                width
            };

            println!("pixel_size: {:}", pixel_size);

            let mut faces = vec![vec![]; 6];
            if view_type == vk::ImageViewType::CUBE {
                for (face, line) in bytes.chunks(pixel_size * width as usize).enumerate() {
                    // println!("face: {:}, line: {:?}",face,line);
                    faces[face % 6].extend(line);
                }

                bytes = faces.concat();
            }
            layer_count = if view_type == vk::ImageViewType::CUBE {
                6
            } else {
                1
            };
        } else {
            width = 256;
            height = 256;
            layer_count = 1;
            pixel_size = 4;
            bytes = vec![69; (width * height * 4) as usize];
        }
        let image_info = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(MIP_LEVELS)
            .array_layers(layer_count)
            .format(image_format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(
                vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .samples(vk::SampleCountFlagBits::_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .flags(if view_type == vk::ImageViewType::CUBE {
                vk::ImageCreateFlags::CUBE_COMPATIBLE
            } else {
                vk::ImageCreateFlags::empty()
            });

        let allocation_info = vk_mem_erupt::AllocationCreateInfo {
            usage: vk_mem_erupt::MemoryUsage::GpuOnly,
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ..Default::default()
        };

        let (image, allocation, _) = vulkan_data
            .allocator
            .as_ref()
            .unwrap()
            .create_image(&image_info, &allocation_info)
            .unwrap();

        let view_info = vk::ImageViewCreateInfoBuilder::new()
            .image(image)
            .view_type(view_type)
            .format(image_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: MIP_LEVELS,
                base_array_layer: 0,
                layer_count,
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
            vulkan_data
                .device
                .as_ref()
                .unwrap()
                .create_sampler(&sampler_info, None)
                .unwrap()
        };

        let buffer_info = vk::BufferCreateInfoBuilder::new()
            .size(bytes.len() as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC);

        let buffer_allocation_create_info = vk_mem_erupt::AllocationCreateInfo {
            usage: vk_mem_erupt::MemoryUsage::CpuOnly,
            flags: vk_mem_erupt::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };
        let (staging_buffer, staging_buffer_allocation, staging_buffer_allocation_info) =
            vulkan_data
                .allocator
                .as_ref()
                .unwrap()
                .create_buffer(&buffer_info, &buffer_allocation_create_info)
                .unwrap();
        unsafe {
            staging_buffer_allocation_info
                .get_mapped_data()
                .copy_from_nonoverlapping(bytes.as_ptr(), bytes.len())
        };
        vulkan_data.transition_image_layout(
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: MIP_LEVELS,
                base_array_layer: 0,
                layer_count,
            },
        );
        let command_buffer = vulkan_data.begin_single_time_commands();

        let mut regions = vec![];
        println!("Width: {:}, Height: {:}", width, height);
        let face_order = match layer_count {
            1 => vec![0],
            // 6 => vec![0,1,4,5,2,3],
            6 => vec![0, 1, 2, 3, 4, 5],
            // 6 => vec![2,3,4,5,0,1],
            // 6 => vec![0,0,0,0,0,0],
            _ => unimplemented!(),
        };
        for (face_index, face) in face_order.into_iter().enumerate() {
            regions.push(
                vk::BufferImageCopyBuilder::new()
                    .buffer_offset((face * width * width * pixel_size as u32) as u64)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: face_index as u32,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    }),
            );
        }

        unsafe {
            vulkan_data
                .device
                .as_ref()
                .unwrap()
                .cmd_copy_buffer_to_image(
                    command_buffer,
                    staging_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                )
        };

        vulkan_data.end_single_time_commands(command_buffer);

        unsafe {
            vulkan_data
                .device
                .as_ref()
                .unwrap()
                .device_wait_idle()
                .unwrap();
        }

        vulkan_data.generate_mipmaps(image, width, height, MIP_LEVELS, layer_count);

        vulkan_data
            .allocator
            .as_ref()
            .unwrap()
            .destroy_buffer(staging_buffer, &staging_buffer_allocation);

        let texture = CombinedSampledImage {
            image,
            image_view,
            sampler,
            allocation,
            width,
            height,
        };

        return Some(texture);
    }
}

pub struct TextureSet {
    pub albedo: Option<CombinedSampledImage>,
    pub normal: Option<CombinedSampledImage>,
    pub roughness_metalness_ao: Option<CombinedSampledImage>,
}

impl TextureSet {
    pub(crate) fn new_empty() -> Self {
        Self {
            albedo: None,
            normal: None,
            roughness_metalness_ao: None,
        }
    }
}