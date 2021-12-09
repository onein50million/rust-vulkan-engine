use crate::cube::*;
use crate::support::*;
use erupt::{vk, DeviceLoader, EntryLoader, ExtendableFromConst, InstanceLoader, SmallVec};
use image::{GenericImageView, ImageFormat};
use nalgebra::{Matrix3x2, Matrix4, Rotation3, Transform3, Vector2, Vector3, Vector4};
use std::convert::{TryFrom, TryInto};
use std::ffi::{c_void, CStr, CString};
use std::fmt::Debug;
use std::fs::File;
use std::io::BufReader;
use std::mem::size_of;
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use erupt::vk1_0::{Format, ImageViewType};
use fastrand::f32;
use gdal::Metadata;
use image::codecs::hdr::HdrDecoder;
use winit::window::Window;

const BASE_VOXEL_SIZE: f32 = 128.0;
const MIP_LEVELS: u32 = 6;

pub(crate) struct CpuImage{
    image: CombinedImage,
    allocation_info: vk_mem_erupt::AllocationInfo,
}
impl CpuImage{
    fn new(vulkan_data: &VulkanData, width: u32, height: u32) -> Self{
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
            required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE |
                vk::MemoryPropertyFlags::HOST_COHERENT |
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
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

        return Self{
            image: CombinedImage {
                image,
                image_view,
                sampler,
                allocation,
                width,
                height
            },
            allocation_info,
        }
    }

    pub(crate) fn get_data(&self) -> Vec<u8>{
        unsafe{
            std::slice::from_raw_parts(
                self.allocation_info.get_mapped_data(),
                (self.image.width * self.image.height) as usize)
                .try_into().unwrap()

        }
    }

    pub(crate) fn write_data(&mut self, data: Vec<u8>){
        unsafe{
            self.allocation_info.get_mapped_data().copy_from_nonoverlapping(data.as_ptr(), data.len().max((self.image.width * self.image.height) as usize))
        }
    }
}

struct CombinedImage {
    image: vk::Image,
    image_view: vk::ImageView,
    sampler: vk::Sampler,
    allocation: vk_mem_erupt::Allocation,
    width: u32,
    height: u32,
}
impl CombinedImage {
    fn new(
        vulkan_data: &VulkanData,
        path: std::path::PathBuf,
        view_type: vk::ImageViewType,
        image_format: vk::Format,
        generate_new_image: bool,
    ) -> Option<Self> {
        let mut width;
        let mut height;
        let layer_count;
        let mut bytes;
        let pixel_size;
        if !generate_new_image{
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
                let image = image::codecs::hdr::HdrDecoder::new(BufReader::new(File::open(path).unwrap())).unwrap();

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
        }
        else{
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
        for face in 0..layer_count {
            regions.push(
                vk::BufferImageCopyBuilder::new()
                    .buffer_offset((face * width * width * pixel_size as u32) as u64)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: face,
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

        let texture = CombinedImage {
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

struct TextureSet {
    albedo: Option<CombinedImage>,
    normal: Option<CombinedImage>,
    roughness_metalness_ao: Option<CombinedImage>
}

pub(crate) struct RenderObject {
    pub(crate) vertex_start: u32,
    pub(crate) vertex_count: u32,
    pub(crate) index_start: u32,
    pub(crate) index_count: u32,
    texture_index: usize,
    texture_type: vk::ImageViewType,
    is_highlighted: bool,
    pub(crate) is_globe: bool,
    pub(crate) is_viewmodel: bool,
    pub(crate) model: Matrix4<f32>,
    pub(crate) view: Matrix4<f32>,
    pub(crate) proj: Matrix4<f32>,
}

impl RenderObject {
    fn new(
        vulkan_data: &mut VulkanData,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
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
            false=> {
                texture_index = vulkan_data.textures.len();
                vulkan_data.textures.push(texture);
            }
        }

        let object = RenderObject {
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
            view: Matrix4::identity(),
            proj: Matrix4::identity(),
        };
        return object;
    }
}

impl Drawable for RenderObject {
    fn draw(
        &self,
        device: &DeviceLoader,
        command_buffer: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
        frag_constant: f32,
    ) {
        unsafe {
            let texture_index = self.texture_index as u32;
            let mut bitfield = 0;
            bitfield |= if self.texture_type == vk::ImageViewType::CUBE {
                super::support::flags::IS_CUBEMAP
            } else {
                0
            };
            bitfield |= if self.is_highlighted {
                super::support::flags::IS_HIGHLIGHTED
            } else {
                0
            };
            bitfield |= if self.is_viewmodel {
                super::support::flags::IS_VIEWMODEL
            } else {
                0
            };
            bitfield |= if self.is_globe {
                super::support::flags::IS_GLOBE
            } else {
                0
            };

            let push_constant = PushConstants {
                model: self.model,
                view: self.view,
                proj: self.proj,
                texture_index,
                constant: frag_constant,
                bitfield,
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
        } else {
            unsafe { device.cmd_draw(command_buffer, self.vertex_count, 1, self.vertex_start, 0) };
        }
    }
}

trait Drawable {
    fn draw(
        &self,
        device: &DeviceLoader,
        command_buffer: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
        frag_constant: f32,
    );
}
pub(crate) struct Cubemap {
    render_object: RenderObject,
}

impl Cubemap {
    fn new(vulkan_data: &mut VulkanData, texture_path: std::path::PathBuf) -> Self {
        let indices = CUBEMAP_INDICES.to_vec();
        let vertices = [
            POSITIVE_X_VERTICES,
            NEGATIVE_X_VERTICES,
            POSITIVE_Y_VERTICES,
            NEGATIVE_Y_VERTICES,
            POSITIVE_Z_VERTICES,
            NEGATIVE_Z_VERTICES,
        ]
        .concat();

        let texture = TextureSet {
            albedo: CombinedImage::new(
                vulkan_data,
                texture_path,
                vk::ImageViewType::CUBE,
                vk::Format::R8G8B8A8_SRGB,
                false,
            ),
            normal: None,
            roughness_metalness_ao: None,
        };

        let cubemap = Cubemap {
            render_object: RenderObject::new(vulkan_data, vertices, indices, texture, true),
        };

        return cubemap;
    }
}

impl Cubemap {
    pub(crate) fn process(&mut self, view_matrix_no_translation: Matrix4<f32>, proj: Matrix4<f32>) {
        self.render_object.view = view_matrix_no_translation;
        self.render_object.model = Matrix4::identity();
        self.render_object.proj = proj;
    }
}
impl Drawable for Cubemap {
    fn draw(
        &self,
        device: &DeviceLoader,
        command_buffer: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
        frag_constant: f32,
    ) {
        self.render_object
            .draw(device, command_buffer, pipeline_layout, frag_constant);
    }
}

pub(crate) struct VulkanData {
    rng: fastrand::Rng,
    instance: Option<InstanceLoader>,
    entry: Option<EntryLoader>,
    pub(crate) device: Option<DeviceLoader>,
    physical_device: Option<vk::PhysicalDevice>,
    allocator: Option<vk_mem_erupt::Allocator>,
    main_queue: Option<vk::Queue>,
    main_queue_index: Option<u32>,
    surface: Option<vk::SurfaceKHR>,
    surface_format: Option<vk::SurfaceFormatKHR>,
    pub(crate) surface_capabilities: Option<vk::SurfaceCapabilitiesKHR>,
    swapchain: Option<vk::SwapchainKHR>,
    swapchain_images: SmallVec<vk::Image>,
    swapchain_image_views: Option<Vec<vk::ImageView>>,
    compute_images: Vec<vk::Image>,
    compute_image_allocations: Vec<vk_mem_erupt::Allocation>,
    compute_image_views: Vec<vk::ImageView>,
    compute_samplers: Vec<vk::Sampler>,
    vert_shader_module: Option<vk::ShaderModule>,
    frag_shader_module: Option<vk::ShaderModule>,
    pipeline_layout: Option<vk::PipelineLayout>,
    compute_pipeline_layout: Option<vk::PipelineLayout>,
    render_pass: Option<vk::RenderPass>,
    post_process_render_pass: Option<vk::RenderPass>,
    graphics_pipelines: SmallVec<vk::Pipeline>,
    compute_pipelines: SmallVec<vk::Pipeline>,
    swapchain_framebuffers: Option<Vec<vk::Framebuffer>>,
    command_pool: Option<vk::CommandPool>,
    command_buffers: Option<SmallVec<vk::CommandBuffer>>,
    image_available_semaphore: Option<vk::Semaphore>,
    render_finished_semaphore: Option<vk::Semaphore>,
    in_flight_fence: Option<vk::Fence>,
    vertex_buffer: Option<vk::Buffer>,
    vertex_buffer_memory: Option<vk::DeviceMemory>,
    index_buffer: Option<vk::Buffer>,
    index_buffer_memory: Option<vk::DeviceMemory>,
    mip_levels: u32,
    depth_image: Option<vk::Image>,
    depth_sampler: Option<vk::Sampler>,
    depth_image_memory: Option<vk::DeviceMemory>,
    depth_image_view: Option<vk::ImageView>,
    color_image: Option<vk::Image>,
    color_image_memory: Option<vk::DeviceMemory>,
    color_image_view: Option<vk::ImageView>,
    pub(crate) cubemap: Option<Cubemap>,
    fullscreen_quads: Vec<RenderObject>,
    pub(crate) vertices: Vec<Vertex>,
    pub(crate) indices: Vec<u32>,
    uniform_buffer_pointers: Vec<*mut u8>,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffer_allocations: Vec<vk_mem_erupt::Allocation>,
    pub(crate) uniform_buffer_object: UniformBufferObject,
    storage_buffer_pointer: Option<*mut u8>,
    storage_buffer: Option<vk::Buffer>,
    storage_buffer_allocation: Option<vk_mem_erupt::Allocation>,
    msaa_samples: vk::SampleCountFlagBits,
    descriptor_pool: Option<vk::DescriptorPool>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_sets: Option<SmallVec<vk::DescriptorSet>>,
    compute_descriptor_pool: Option<vk::DescriptorPool>,
    compute_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    compute_descriptor_sets: Option<SmallVec<vk::DescriptorSet>>,
    last_frame_instant: std::time::Instant,
    pub(crate) objects: Vec<RenderObject>,
    textures: Vec<TextureSet>,
    cubemaps: Vec<CombinedImage>,
    irradiance_maps: Vec<CombinedImage>,
    environment_maps: Vec<CombinedImage>,
    brdf_lut: Option<CombinedImage>,
    fallback_texture: Option<TextureSet>,
    pub(crate) cpu_images: Vec<CpuImage>,
}

fn get_random_vector(rng: &fastrand::Rng, length: usize) -> Vec<f32> {
    let mut vector = Vec::new();
    for _ in 0..length {
        vector.push(rng.f32());
    }
    return vector;
}

impl VulkanData {
    pub(crate) fn new() -> Self {

        let mut lights = [Light::new(); NUM_LIGHTS];

        for i in 0..NUM_LIGHTS{
            lights[i] = Light::new();
        }

        let uniform_buffer_object = UniformBufferObject {
            random: [Vector4::new(0.0f32, 0.0f32, 0.0f32, 0.0f32); NUM_RANDOM],
            lights,
            player_index: 0,
            num_lights: 100,
            map_mode: 0,
            value4: 0,
            mouse_position: Vector2::new(0.0, 0.0),
            time: 0.0,
            b: 0.0,
            c: 0.0,
            d: 0.0
        };

        let indices = vec![];
        let vertices = vec![];

        return VulkanData {
            rng: fastrand::Rng::new(),
            instance: None,
            entry: None,
            device: None,
            physical_device: None,
            allocator: None,
            main_queue: None,
            main_queue_index: None,
            surface: None,
            surface_format: None,
            surface_capabilities: None,
            swapchain: None,
            swapchain_images: SmallVec::new(),
            swapchain_image_views: None,
            compute_images: vec![],
            compute_image_allocations: vec![],
            compute_image_views: vec![],
            compute_samplers: vec![],
            vert_shader_module: None,
            frag_shader_module: None,
            pipeline_layout: None,
            compute_pipeline_layout: None,
            render_pass: None,
            post_process_render_pass: None,
            graphics_pipelines: SmallVec::new(),
            compute_pipelines: SmallVec::new(),
            swapchain_framebuffers: None,
            command_pool: None,
            command_buffers: None,
            image_available_semaphore: None,
            render_finished_semaphore: None,
            in_flight_fence: None,
            vertex_buffer: None,
            vertex_buffer_memory: None,
            index_buffer: None,
            index_buffer_memory: None,
            mip_levels: 1,
            depth_image: None,
            depth_sampler: None,
            depth_image_memory: None,
            depth_image_view: None,
            color_image: None,
            color_image_memory: None,
            color_image_view: None,
            msaa_samples: vk::SampleCountFlagBits::_1,
            vertices,
            indices,
            descriptor_pool: None,
            descriptor_set_layout: None,
            descriptor_sets: None,
            compute_descriptor_pool: None,
            compute_descriptor_set_layout: None,
            compute_descriptor_sets: None,
            uniform_buffer_object,
            storage_buffer_pointer: None,
            storage_buffer: None,
            storage_buffer_allocation: None,
            last_frame_instant: std::time::Instant::now(),
            cubemap: None,
            uniform_buffer_pointers: vec![],
            uniform_buffers: vec![],
            uniform_buffer_allocations: vec![],
            objects: vec![],
            fullscreen_quads: vec![],
            textures: vec![],
            cubemaps: vec![],
            irradiance_maps: vec![],
            environment_maps: vec![],
            brdf_lut: None,
            fallback_texture: None,
            cpu_images: vec![],
        };
    }
    pub(crate) fn init_vulkan(&mut self, window: &Window) {

        let mut validation_layer_names =
            vec![];

        #[cfg(debug_assertions)]
        validation_layer_names.push(erupt::cstr!( "VK_LAYER_KHRONOS_validation"));

        self.entry = Some(erupt::EntryLoader::new().unwrap());

        unsafe {
            self.entry
                .as_ref()
                .unwrap()
                .enumerate_instance_extension_properties(None, None)
                .unwrap()
                .into_iter()
                .for_each(|extension_property| {
                    println!(
                        "Supported Extension: {:}",
                         CStr::from_ptr(extension_property.extension_name.as_ptr()).to_string_lossy()
                    );
                });
        }

        println!("Verts: {:}", self.vertices.len());

        let app_info = vk::ApplicationInfoBuilder::new();
        let mut surface_extensions =
            erupt::utils::surface::enumerate_required_extensions(window).unwrap();
        surface_extensions.push(erupt::extensions::ext_debug_utils::EXT_DEBUG_UTILS_EXTENSION_NAME);
        // surface_extensions.push(vk::ImgFilterCubicFn::name());

        let enables = [vk::ValidationFeatureEnableEXT::BEST_PRACTICES_EXT];
        let mut features =
            vk::ValidationFeaturesEXTBuilder::new().enabled_validation_features(&enables);

        let create_info = vk::InstanceCreateInfoBuilder::new()
            .enabled_extension_names(&surface_extensions)
            .application_info(&app_info)
            .enabled_layer_names(&validation_layer_names)
            .extend_from(&mut features);
        self.instance = Some(unsafe {
            InstanceLoader::new(self.entry.as_ref().unwrap(), &create_info, None).unwrap()
        });

        let physical_devices = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .enumerate_physical_devices(None)
        }
        .unwrap();
        self.physical_device = None;

        if physical_devices.len() == 1 {
            self.physical_device = Some(physical_devices[0]);
        } else {
            physical_devices.into_iter().find(|device| {
                //TODO: implement multiple gpu finding

                let properties = unsafe {
                    self.instance
                        .as_ref()
                        .unwrap()
                        .get_physical_device_properties(*device)
                };
                let device_name_cstring =
                    unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }.to_owned();
                let device_name = device_name_cstring.to_str().unwrap();

                println!(
                    "Device Name: {}\nDriver: {}",
                    device_name, &properties.driver_version
                );
                return true;
            });
        }

        self.msaa_samples = self.get_max_usable_sample_count();
        // self.msaa_samples = vk::SampleCountFlagBits::_1;

        println!("Samples: {:?}", self.msaa_samples);

        self.main_queue_index = Some(
            unsafe {
                self.instance
                    .as_ref()
                    .unwrap()
                    .get_physical_device_queue_family_properties(
                        self.physical_device.unwrap(),
                        None,
                    )
            }
            .iter()
            .position(|queue| queue.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .unwrap() as u32,
        );
        let queue_create_info = vk::DeviceQueueCreateInfoBuilder::new()
            .queue_family_index(self.main_queue_index.unwrap())
            .queue_priorities(&[1.0f32]);
        let queue_create_infos = &[queue_create_info];

        let device_features = vk::PhysicalDeviceFeaturesBuilder::new().sampler_anisotropy(true);

        let mut extended_dynamic_state_features =
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXTBuilder::new()
                .extended_dynamic_state(true);

        let device_extension_names_raw = vec![erupt::extensions::khr_swapchain::KHR_SWAPCHAIN_EXTENSION_NAME,
                                              erupt::extensions::ext_extended_dynamic_state::EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME];
        let device_create_info = vk::DeviceCreateInfoBuilder::new()
            .extend_from(&mut extended_dynamic_state_features)
            .queue_create_infos(queue_create_infos)
            .enabled_layer_names(&validation_layer_names)
            .enabled_extension_names(&device_extension_names_raw)
            .enabled_features(&device_features);
        self.device = Some(
            unsafe {
                DeviceLoader::new(
                    self.instance.as_ref().unwrap(),
                    self.physical_device.unwrap(),
                    &device_create_info,
                    None,
                )
            }
            .unwrap(),
        );

        self.main_queue = Some(unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_device_queue(self.main_queue_index.unwrap(), 0)
        });

        println!("Creating Allocator");
        self.create_allocator();

        println!("Creating Surface");
        self.create_surface(window);

        println!("Getting surface support and capabilities");
        self.get_surface_support();
        self.get_surface_capabilities();

        println!("Creating Swapchain");
        self.create_swapchain();
        self.create_swapchain_image_views();
        println!("Creating uniform buffers");
        self.create_uniform_buffers();

        self.create_descriptor_set_layout();
        self.create_render_pass();
        self.create_post_process_render_pass();

        self.create_command_pool();

        self.fallback_texture = Some(TextureSet {
            albedo: CombinedImage::new(
                self,
                "models/fallback/albedo.png".parse().unwrap(),
                vk::ImageViewType::_2D,
                vk::Format::R8G8B8A8_SRGB,
                false,
            ),
            normal: CombinedImage::new(
                self,
                "models/fallback/normal.png".parse().unwrap(),
                vk::ImageViewType::_2D,
                vk::Format::R8G8B8A8_UNORM,
                false,
            ),
            roughness_metalness_ao: CombinedImage::new(
                self,
                "models/fallback/rough_metal_ao.png".parse().unwrap(),
                vk::ImageViewType::_2D,
                vk::Format::R8G8B8A8_UNORM,
                false,
            ),
        });

        for _ in 0..NUM_MODELS{
            self.cpu_images.push(CpuImage::new(self,128,128))
        }

        self.create_compute_images();
        for _ in 0..1 {
            let mut quad = RenderObject::new(
                self,
                FULLSCREEN_QUAD_VERTICES.to_vec(),
                QUAD_INDICES.to_vec(),
                TextureSet {
                    albedo: Some(CombinedImage {
                        image: self.compute_images[0],
                        image_view: self.compute_image_views[0],
                        sampler: self.compute_samplers[0],
                        allocation: self.compute_image_allocations[0],
                        width: self.surface_capabilities.unwrap().current_extent.width,
                        height: self.surface_capabilities.unwrap().current_extent.height,
                    }),
                    normal: None,
                    roughness_metalness_ao: None,
                },
                false,
            );
            quad.is_viewmodel = true;
            self.fullscreen_quads.push(quad);
        }


        self.create_color_resources();
        self.create_depth_resources();
        self.create_cubemap_resources();

        println!("Loading Shaders");
        self.load_shaders();
        self.create_graphics_pipelines();

        self.create_framebuffers();

        self.create_vertex_buffer();
        self.create_index_buffer();

        println!("Creating Descriptor pools");
        self.create_descriptor_pool();
        println!("Creating compute pipeline");
        self.create_compute_pipelines();
        println!("Creating descriptor sets");
        self.create_descriptor_sets();
        println!("Creating command buffer");
        self.create_command_buffers();
        println!("Creating sync objects");
        self.create_sync_objects();

        println!("Finished init");
    }

    fn create_compute_images(&mut self) {
        for _ in 0..1 {
            let image_info = vk::ImageCreateInfoBuilder::new()
                .image_type(vk::ImageType::_2D)
                .extent(vk::Extent3D {
                    width: self.surface_capabilities.unwrap().current_extent.width,
                    height: self.surface_capabilities.unwrap().current_extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .format(vk::Format::R8G8B8A8_UNORM)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
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

            let image_view =
                self.create_image_view(image, image_info.format, vk::ImageAspectFlags::COLOR, 1);

            let sampler_info = vk::SamplerCreateInfoBuilder::new()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
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

            let command_buffer = self.begin_single_time_commands();

            let barrier = vk::ImageMemoryBarrierBuilder::new()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            unsafe {
                self.device.as_ref().unwrap().cmd_pipeline_barrier(
                    command_buffer,
                    Some(vk::PipelineStageFlags::ALL_COMMANDS),
                    Some(vk::PipelineStageFlags::ALL_COMMANDS),
                    Some(vk::DependencyFlags::empty()),
                    &[],
                    &[],
                    &[barrier],
                )
            }

            self.end_single_time_commands(command_buffer);

            self.compute_images.push(image);
            self.compute_image_allocations.push(allocation);
            self.compute_image_views.push(image_view);
            self.compute_samplers.push(sampler);
        }
    }

    fn destroy_compute_images(&mut self) {
        for i in 0..self.compute_images.len() {
            let image = self.compute_images.remove(i);
            let allocation = self.compute_image_allocations.remove(i);
            let image_view = self.compute_image_views.remove(i);
            let sampler = self.compute_samplers.remove(i);
            self.allocator
                .as_ref()
                .unwrap()
                .destroy_image(image, &allocation);
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .destroy_image_view(Some(image_view), None)
            };
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .destroy_sampler(Some(sampler), None)
            };
        }
    }

    fn create_allocator(&mut self) {
        //I don't know what I'm doing but I know this is bad

        /*
        what I think is happening:
          We assume that vk_mem_erupt doesn't do anything bad to device or instance,
          and I think this is guaranteed by Arc.
          We take the raw bytes that make up device and transmute it into a new device
          which the borrow checker doesn't know is related (hmmm this sounds bad).
          We put this device into an Arc that vk_mem_erupt requires for some reason
        */
        let device = self.device.as_ref().unwrap() as *const DeviceLoader;
        let instance = self.instance.as_ref().unwrap() as *const InstanceLoader;
        unsafe {
            let device = std::ptr::read(device);
            let device = std::mem::transmute(device);
            let device = Arc::new(device);

            let instance = std::ptr::read(instance);
            let instance = std::mem::transmute(instance);
            let instance = Arc::new(instance);

            let allocator_create_info = vk_mem_erupt::AllocatorCreateInfo {
                physical_device: self.physical_device.unwrap(),
                device,
                instance,
                flags: vk_mem_erupt::AllocatorCreateFlags::NONE,
                preferred_large_heap_block_size: 0,
                frame_in_use_count: 0,
                heap_size_limits: None,
            };
            self.allocator = Some(vk_mem_erupt::Allocator::new(&allocator_create_info).unwrap());
        }
    }

    fn create_surface(&mut self, window: &Window) {
        self.surface = Some(
            unsafe {
                erupt::utils::surface::create_surface(self.instance.as_ref().unwrap(), window, None)
            }
            .unwrap(),
        );
    }

    fn create_sync_objects(&mut self) {
        let semaphore_create_info = vk::SemaphoreCreateInfoBuilder::new();
        self.image_available_semaphore = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_semaphore(&semaphore_create_info, None)
            }
            .unwrap(),
        );
        self.render_finished_semaphore = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_semaphore(&semaphore_create_info, None)
            }
            .unwrap(),
        );
        let fence_create_info =
            vk::FenceCreateInfoBuilder::new().flags(vk::FenceCreateFlags::SIGNALED);
        self.in_flight_fence = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_fence(&fence_create_info, None)
            }
            .unwrap(),
        );
    }

    fn get_max_usable_sample_count(&self) -> vk::SampleCountFlagBits {
        let physical_device_properties = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_properties(self.physical_device.unwrap())
        };
        let counts = physical_device_properties
            .limits
            .framebuffer_color_sample_counts
            & physical_device_properties
                .limits
                .framebuffer_depth_sample_counts;

        if counts.contains(vk::SampleCountFlags::_64) {
            return vk::SampleCountFlagBits::_64;
        } else if counts.contains(vk::SampleCountFlags::_32) {
            return vk::SampleCountFlagBits::_32;
        } else if counts.contains(vk::SampleCountFlags::_16) {
            return vk::SampleCountFlagBits::_16;
        } else if counts.contains(vk::SampleCountFlags::_8) {
            return vk::SampleCountFlagBits::_8;
        } else if counts.contains(vk::SampleCountFlags::_4) {
            return vk::SampleCountFlagBits::_4;
        } else if counts.contains(vk::SampleCountFlags::_2) {
            return vk::SampleCountFlagBits::_2;
        } else if counts.contains(vk::SampleCountFlags::_1) {
            return vk::SampleCountFlagBits::_1;
        } else {
            panic!("No samples found???")
        }
    }
    pub(crate) fn load_folder(&mut self, folder: PathBuf) -> usize {
        let albedo_path = folder.join("albedo.png");
        let albedo = CombinedImage::new(
            self,
            albedo_path,
            vk::ImageViewType::_2D,
            vk::Format::R8G8B8A8_SRGB,
            false,
        );

        let normal_path = folder.join("normal.png");
        let normal = CombinedImage::new(
            self,
            normal_path,
            vk::ImageViewType::_2D,
            vk::Format::R8G8B8A8_UNORM,
            false
        );


        let roughness_path = folder.join("rough_metal_ao.png");
        let rough_metal_ao = CombinedImage::new(
            self,
            roughness_path,
            vk::ImageViewType::_2D,
            vk::Format::R8G8B8A8_UNORM,
            false
        );

        let texture = TextureSet { albedo, normal, roughness_metalness_ao: rough_metal_ao };
        let vertices_and_indices;
        if folder.join("model.glb").is_file(){
            vertices_and_indices = Self::load_gltf_model(folder.join("model.glb"));

        }else{
            vertices_and_indices = (vec![], vec![]);
        }

        let vertices = vertices_and_indices.0;
        let indices = vertices_and_indices.1;
        let render_object = RenderObject::new(self, vertices, indices, texture, false);
        let output = self.objects.len();
        self.objects.push(render_object);
        return output;
    }



    pub(crate) fn load_gltf_model(path: std::path::PathBuf) -> (Vec<Vertex>, Vec<u32>) {
        println!("loading {:}", path.to_str().unwrap());
        let (gltf, buffers, _) = gltf::import(path).unwrap();
        // let materials = material_result.unwrap();


        let mut out_vertices = vec![];
        let mut out_indices = vec![];
        // let mut positions = vec![];
        for node in gltf.nodes(){
            let transformation_matrix = Matrix4::from(node.transform().matrix()) * Matrix4::from_axis_angle(&Vector3::x_axis(), std::f32::consts::PI);
            let transformation_position = transformation_matrix.column(3).xyz();
            match node.mesh(){
                None => {}
                Some(mesh) => {
                    let mut texture_type = 0; //I think primitve index should line up with texture type... maybe?
                    for primitive in mesh.primitives() {
                        println!("Texture type: {:}", texture_type);
                        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                        let indices = reader.read_indices().unwrap().into_u32().collect::<Vec<_>>();
                        let positions = reader.read_positions().unwrap().collect::<Vec<_>>();
                        let normals = reader.read_normals().unwrap().collect::<Vec<_>>();
                        let texture_coordinates = reader.read_tex_coords(0).unwrap().into_f32().collect::<Vec<_>>();

                        out_indices.extend_from_slice(&indices);
                        for i in 0..positions.len(){
                            let position = Vector3::from(positions[i]);
                            let position = transformation_matrix.transform_vector(&position);
                            out_vertices.push(Vertex {
                                position,
                                normal:Vector3::zeros(),
                                tangent:Vector4::zeros(),
                                texture_coordinate:Vector2::zeros(),
                                texture_type,
                                elevation: 0.0,
                                aridity: 0.0,
                                population: 0.0,
                                warmest_temperature: 0.0,
                                coldest_temperature: 0.0
                            });
                        }

                        for triangle in indices.chunks(3) {
                            for i in 0..3{
                                let normal = normals[triangle[i] as usize];
                                let normal = Vector3::from(normal);
                                let normal = transformation_matrix.transform_vector(&normal) - transformation_position;

                                let texture_coordinate = texture_coordinates[triangle[i] as usize ];
                                let texture_coordinate = Vector2::from(texture_coordinate);

                                let index1 = triangle[(i+0)%3] as usize;
                                let index2 = triangle[(i+1)%3] as usize;
                                let index3 = triangle[(i+2)%3] as usize;

                                let position3 = transformation_matrix.transform_vector(&Vector3::from(positions[index3]));
                                let position1 = transformation_matrix.transform_vector(&Vector3::from(positions[index1]));
                                let position2 = transformation_matrix.transform_vector(&Vector3::from(positions[index2]));

                                // let position1 = &Vector3::from(positions[index1]);
                                // let position2 = &Vector3::from(positions[index2]);
                                // let position3 = &Vector3::from(positions[index3]);



                                let edge1 = position2 - position1;
                                let edge2 = position3 - position1;
                                let delta_uv1 = Vector2::from(texture_coordinates[index2]) - Vector2::from(texture_coordinates[index1]);
                                let delta_uv2 = Vector2::from(texture_coordinates[index3]) - Vector2::from(texture_coordinates[index1]);

                                let f = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y);

                                let mut tangent = Vector4::new(0.0,0.0,0.0,-1.0);

                                tangent.x = f * (delta_uv2.y * edge1.x - delta_uv1.y * edge2.x);
                                tangent.y = f * (delta_uv2.y * edge1.y - delta_uv1.y * edge2.y);
                                tangent.z = f * (delta_uv2.y * edge1.z - delta_uv1.y * edge2.z);

                                let normal_tangent = tangent.xyz().normalize();

                                tangent = Vector4::new(
                                    normal_tangent.x,
                                    normal_tangent.y,
                                    normal_tangent.z,
                                    tangent.w
                                );
                                out_vertices[index1].normal = normal;
                                out_vertices[index1].texture_coordinate = texture_coordinate;
                                out_vertices[index1].tangent = tangent;

                            }
                        }
                        texture_type += 1;
                    }
                }
            }
        }

        return (out_vertices,out_indices);
    }

    fn create_depth_resources(&mut self) {
        let depth_format = self.find_depth_format();

        let (depth_image, depth_image_memory) = self.create_image_with_memory(
            self.surface_capabilities.unwrap().current_extent.width,
            self.surface_capabilities.unwrap().current_extent.height,
            1,
            self.msaa_samples,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
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
            self
                .device
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

    fn get_surface_support(&mut self) {
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

    fn get_surface_capabilities(&mut self) {
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

    fn create_descriptor_pool(&mut self) {
        let pool_sizes = [
            vk::DescriptorPoolSizeBuilder::new()
                .descriptor_count(self.swapchain_image_views.as_ref().unwrap().len() as u32)
                ._type(vk::DescriptorType::UNIFORM_BUFFER),
            vk::DescriptorPoolSizeBuilder::new()
                .descriptor_count(self.swapchain_image_views.as_ref().unwrap().len() as u32)
                ._type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        ];

        let compute_pool_sizes = [vk::DescriptorPoolSizeBuilder::new()
            .descriptor_count(self.swapchain_image_views.as_ref().unwrap().len() as u32)
            ._type(vk::DescriptorType::STORAGE_IMAGE)];

        let pool_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .pool_sizes(&pool_sizes)
            .max_sets(self.swapchain_image_views.as_ref().unwrap().len() as u32);
        let compute_pool_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .pool_sizes(&compute_pool_sizes)
            .max_sets(self.swapchain_image_views.as_ref().unwrap().len() as u32);

        self.descriptor_pool = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_descriptor_pool(&pool_info, None)
            }
            .unwrap(),
        );
        self.compute_descriptor_pool = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_descriptor_pool(&compute_pool_info, None)
            }
            .unwrap(),
        );
    }

    fn create_descriptor_set_layout(&mut self) {
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

    fn create_descriptor_sets(&mut self) {
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

        println!(
            "Num descriptor sets: {:?}",
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
                            .image_view(self.compute_image_views[0])
                            .sampler(self.compute_samplers[0]),
                    );
                    let normal = self.fallback_texture
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

                    let roughness = self.fallback_texture
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

                let descriptor_writes = [
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

                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .update_descriptor_sets(&descriptor_writes, &[]);
                }
            });
    }

    fn create_index_buffer(&mut self) {
        let buffer_size: vk::DeviceSize = (std::mem::size_of::<u32>() * self.indices.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = self.create_buffer_with_memory(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let mut destination_pointer = std::ptr::null::<*mut c_void>() as *mut c_void;
        unsafe {
            self.device.as_ref().unwrap().map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                None,
                &mut destination_pointer as *mut *mut c_void,
            )
        }
        .unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.indices.as_ptr() as *mut c_void,
                destination_pointer,
                buffer_size as usize,
            )
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .unmap_memory(staging_buffer_memory)
        };

        let (index_buffer, index_buffer_memory) = self.create_buffer_with_memory(
            buffer_size + 40,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        self.index_buffer = Some(index_buffer);
        self.index_buffer_memory = Some(index_buffer_memory);
        self.copy_buffer(staging_buffer, self.index_buffer.unwrap(), buffer_size);

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(Some(staging_buffer), None)
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .free_memory(Some(staging_buffer_memory), None)
        };
    }

    fn create_vertex_buffer(&mut self) {
        let buffer_size: vk::DeviceSize =
            (std::mem::size_of::<Vertex>() * self.vertices.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = self.create_buffer_with_memory(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let mut destination_pointer = std::ptr::null::<*mut c_void>() as *mut c_void;
        unsafe {
            self.device.as_ref().unwrap().map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                None,
                &mut destination_pointer as *mut *mut c_void,
            )
        }
        .unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.vertices.as_ptr() as *mut c_void,
                destination_pointer,
                buffer_size as usize,
            )
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .unmap_memory(staging_buffer_memory)
        };

        let (vertex_buffer, vertex_buffer_memory) = self.create_buffer_with_memory(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        self.vertex_buffer = Some(vertex_buffer);
        self.vertex_buffer_memory = Some(vertex_buffer_memory);
        self.copy_buffer(staging_buffer, self.vertex_buffer.unwrap(), buffer_size);

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(Some(staging_buffer), None)
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .free_memory(Some(staging_buffer_memory), None)
        };
    }

    pub(crate) fn update_vertex_and_index_buffers(&mut self) {
        if self.vertex_buffer.is_some() || self.vertex_buffer_memory.is_some() {
            self.destroy_vertex_buffer();
            self.create_vertex_buffer();
        }
        if self.index_buffer.is_some() || self.index_buffer_memory.is_some() {
            self.destroy_index_buffer();
            self.create_index_buffer();
        }
        unsafe{
            self.device.as_ref().unwrap().reset_descriptor_pool(
                self.descriptor_pool.unwrap(),
                None,
            ).unwrap();
        }

        self.create_descriptor_sets();
    }

    fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> u32 {
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

    fn copy_buffer(
        &self,
        source_buffer: vk::Buffer,
        destination_buffer: vk::Buffer,
        size: vk::DeviceSize,
    ) {
        let command_buffer = self.begin_single_time_commands();
        unsafe {
            self.device.as_ref().unwrap().cmd_copy_buffer(
                command_buffer,
                source_buffer,
                destination_buffer,
                &[vk::BufferCopyBuilder::new().size(size)],
            )
        };
        self.end_single_time_commands(command_buffer);
    }

    fn create_buffer_with_memory(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_info = vk::BufferCreateInfoBuilder::new()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_buffer(&buffer_info, None)
        }
        .unwrap();

        let memory_requirements = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_buffer_memory_requirements(buffer)
        };

        let allocate_info = vk::MemoryAllocateInfoBuilder::new()
            .allocation_size(memory_requirements.size)
            .memory_type_index(
                self.find_memory_type(memory_requirements.memory_type_bits, properties),
            );

        let buffer_memory = unsafe {
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
                .bind_buffer_memory(buffer, buffer_memory, 0)
        }
        .unwrap();

        return (buffer, buffer_memory);
    }

    pub(crate) fn draw_frame(&mut self) {
        let fences = [self.in_flight_fence.unwrap()];
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .wait_for_fences(&fences, true, u64::MAX)
        }
        .unwrap();
        unsafe { self.device.as_ref().unwrap().reset_fences(&fences) }.unwrap();

        // unsafe{
        //     self.device.as_ref().unwrap().free_command_buffers(
        //         self.command_pool.unwrap(),
        //         self.command_buffers.as_ref().unwrap(),
        //     );
        // }
        //
        // self.create_command_buffers();
        self.run_commands();

        let image_index: u32;

        match unsafe {
            self.device.as_ref().unwrap().acquire_next_image_khr(
                self.swapchain.unwrap(),
                u64::MAX,
                Some(self.image_available_semaphore.unwrap()),
                None,
            )
        }
        .result()
        {
            Ok(index) => image_index = index,
            Err(e) => {
                if e == vk::Result::ERROR_OUT_OF_DATE_KHR || e == vk::Result::SUBOPTIMAL_KHR {
                    unsafe { self.device.as_ref().unwrap().device_wait_idle() }.unwrap();
                    self.cleanup_swapchain();
                    self.recreate_swapchain();
                    return;
                } else {
                    panic!("acquire_next_image error");
                }
            }
        };

        let wait_semaphores = [self.image_available_semaphore.unwrap()];
        let signal_semaphores = [self.render_finished_semaphore.unwrap()];
        let wait_stages = vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.command_buffers.as_ref().unwrap()[image_index as usize]];
        let submits = [vk::SubmitInfoBuilder::new()
            .command_buffers(&command_buffers)
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(&signal_semaphores)
            .wait_dst_stage_mask(&wait_stages)];

        unsafe {
            self.device.as_ref().unwrap().queue_submit(
                self.main_queue.unwrap(),
                &submits,
                Some(self.in_flight_fence.unwrap()),
            )
        }
        .unwrap();

        let swapchains = [self.swapchain.unwrap()];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHRBuilder::new()
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .wait_semaphores(&signal_semaphores);

        match unsafe {
            self.device
                .as_ref()
                .unwrap()
                .queue_present_khr(self.main_queue.unwrap(), &present_info)
        }
        .result()
        {
            Ok(_) => {}
            Err(e) => {
                if e == vk::Result::ERROR_OUT_OF_DATE_KHR || e == vk::Result::SUBOPTIMAL_KHR {
                    unsafe { self.device.as_ref().unwrap().device_wait_idle() }.unwrap();
                    self.cleanup_swapchain();
                    self.recreate_swapchain();
                }
            }
        }
    }

    fn create_swapchain(&mut self) {
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

        let surface_format = surface_formats
            .into_iter()
            .find(|format| {
                let format_matches = format.format == vk::Format::R8G8B8A8_SRGB
                    || format.format == vk::Format::B8G8R8_SRGB
                    || format.format == vk::Format::R8G8B8_SRGB
                    || format.format == vk::Format::B8G8R8A8_SRGB
                    || format.format == vk::Format::A8B8G8R8_SRGB_PACK32;
                let color_space_matches = format.color_space == vk::ColorSpaceKHR::COLORSPACE_SRGB_NONLINEAR_KHR ||
                    format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR;
                return format_matches && color_space_matches;
            });

        self.surface_format = Some(
            match surface_format {
                None=>{
                    println!("ERROR: Unable to find surface format, using default");
                    default_surface_format},
                Some(found_format) => {found_format}
            }
        );
        println!("Selected Surface format: {:?}", self.surface_format.unwrap());

        // surface_formats
        //     .into_iter()
        //     .for_each(|format| {
        //         println!("Format Properties: {:?}", unsafe{self.instance.as_ref().unwrap().get_physical_device_format_properties(self.physical_device.unwrap(),format.format)}.optimal_tiling_features);
        //     });

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
    }

    fn create_swapchain_image_views(&mut self) {
        self.swapchain_images = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_swapchain_images_khr(self.swapchain.unwrap(), None)
        }
        .unwrap();

        self.swapchain_image_views = Some(
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
                .collect::<Vec<_>>(),
        );
    }

    fn load_shader(&self, path: std::path::PathBuf) -> vk::ShaderModule {
        let shader_code = erupt::utils::decode_spv(&std::fs::read(path).unwrap()).unwrap();
        return VulkanData::create_shader_module(self.device.as_ref().unwrap(), shader_code);
    }

    fn load_shaders(&mut self) {
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

    fn create_render_pass(&mut self) {
        let push_constant_range = vk::PushConstantRangeBuilder::new()
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
        let push_constant_ranges = [push_constant_range];

        let descriptor_set_layouts = [self.descriptor_set_layout.unwrap()];
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

        let color_attachment = vk::AttachmentDescriptionBuilder::new()
            .format(self.surface_format.unwrap().format)
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let color_attachment_reference = vk::AttachmentReferenceBuilder::new()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attachment = vk::AttachmentDescriptionBuilder::new()
            .format(self.find_depth_format())
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL);
        let depth_attachment_reference = vk::AttachmentReferenceBuilder::new()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);


        let color_attachment_resolve = vk::AttachmentDescriptionBuilder::new()
            .format(self.surface_format.unwrap().format)
            .samples(vk::SampleCountFlagBits::_1)
            .load_op(if self.msaa_samples != vk::SampleCountFlagBits::_1 {
                vk::AttachmentLoadOp::DONT_CARE
            } else {
                vk::AttachmentLoadOp::CLEAR
            })
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_resolve_reference = vk::AttachmentReferenceBuilder::new()
            .attachment(if self.msaa_samples != vk::SampleCountFlagBits::_1 {
                2
            } else {
                0
            })
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_attachment_references;
        let color_attachment_resolve_references;

        if self.msaa_samples != vk::SampleCountFlagBits::_1 {
            color_attachment_resolve_references = vec![color_attachment_resolve_reference];
            color_attachment_references = vec![color_attachment_reference];
        } else {
            color_attachment_resolve_references = vec![];
            color_attachment_references = vec![color_attachment_resolve_reference];
        };

        let mut main_subpass = vk::SubpassDescriptionBuilder::new()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_references)
            .depth_stencil_attachment(&depth_attachment_reference);
        if self.msaa_samples != vk::SampleCountFlagBits::_1 {
            main_subpass = main_subpass.resolve_attachments(&color_attachment_resolve_references);
        }

        let dependencies = [vk::SubpassDependencyBuilder::new()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )];
        let attachments = if self.msaa_samples != vk::SampleCountFlagBits::_1 {
            vec![color_attachment, depth_attachment, color_attachment_resolve]
        } else {
            vec![color_attachment_resolve, depth_attachment]
        };
        let subpasses = [main_subpass];

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
    fn create_post_process_render_pass(&mut self) {

        let color_attachment = vk::AttachmentDescriptionBuilder::new()
            .format(self.surface_format.unwrap().format)
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_attachment_reference = vk::AttachmentReferenceBuilder::new()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attachment = vk::AttachmentDescriptionBuilder::new()
            .format(self.find_depth_format())
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let depth_attachment_reference = vk::AttachmentReferenceBuilder::new()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachment_resolve = vk::AttachmentDescriptionBuilder::new()
            .format(self.surface_format.unwrap().format)
            .samples(vk::SampleCountFlagBits::_1)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_resolve_reference = vk::AttachmentReferenceBuilder::new()
            .attachment(if self.msaa_samples != vk::SampleCountFlagBits::_1 {
                2
            } else {
                0
            })
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_attachment_references;
        let color_attachment_resolve_references;

        if self.msaa_samples != vk::SampleCountFlagBits::_1 {
            color_attachment_resolve_references = vec![color_attachment_resolve_reference];
            color_attachment_references = vec![color_attachment_reference];
        } else {
            color_attachment_resolve_references = vec![];
            color_attachment_references = vec![color_attachment_resolve_reference];
        };

        let mut main_subpass = vk::SubpassDescriptionBuilder::new()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_references)
            .depth_stencil_attachment(&depth_attachment_reference);
        if self.msaa_samples != vk::SampleCountFlagBits::_1 {
            main_subpass = main_subpass.resolve_attachments(&color_attachment_resolve_references);
        }


        let dependencies = [vk::SubpassDependencyBuilder::new()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )];
        let attachments = if self.msaa_samples != vk::SampleCountFlagBits::_1 {
            vec![color_attachment, depth_attachment, color_attachment_resolve]
        } else {
            vec![color_attachment_resolve, depth_attachment]
        };

        let subpasses = [main_subpass];

        let render_pass_info = vk::RenderPassCreateInfoBuilder::new()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        self.post_process_render_pass = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_render_pass(&render_pass_info, None)
            }
            .unwrap(),
        );
    }

    fn create_compute_pipelines(&mut self) {
        let ubo_layout_binding = vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);

        let sampler_layout_binding = vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(1)
            .descriptor_count(NUM_MODELS as u32)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);

        let layout_bindings = [
            ubo_layout_binding,
            sampler_layout_binding,
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(2)
                .descriptor_count(self.compute_images.len() as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(3)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(4)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info =
            vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&layout_bindings);

        self.compute_descriptor_set_layout = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_descriptor_set_layout(&layout_info, None)
            }
            .unwrap(),
        );

        let descriptor_set_layouts = [self.compute_descriptor_set_layout.unwrap()];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfoBuilder::new()
            .descriptor_pool(self.compute_descriptor_pool.unwrap())
            .set_layouts(&descriptor_set_layouts);

        self.compute_descriptor_sets = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .allocate_descriptor_sets(&descriptor_set_allocate_info)
            }
            .unwrap(),
        );

        for i in 0..self.fullscreen_quads.len() {
            let new_image = CombinedImage {
                image: self.compute_images[0],
                image_view: self.compute_image_views[0],
                sampler: self.compute_samplers[0],
                allocation: self.compute_image_allocations[0],
                width: self.surface_capabilities.unwrap().current_extent.width,
                height: self.surface_capabilities.unwrap().current_extent.height,
            };
            self.textures[self.fullscreen_quads[i].texture_index].albedo = Some(new_image);
        }

        for descriptor_set in self.compute_descriptor_sets.as_ref().unwrap() {
            println!("compute descriptor set construction started");
            let buffer_infos = vec![vk::DescriptorBufferInfoBuilder::new()
                .buffer(self.uniform_buffers[0])
                .offset(0)
                .range((std::mem::size_of::<UniformBufferObject>()) as vk::DeviceSize)];

            let mut image_infos = vec![]; //TODO: make this into a function so it stays up to date with main image_infos
            for texture in &self.textures {
                let albedo = texture.albedo.as_ref().unwrap_or(
                    self.fallback_texture
                        .as_ref()
                        .unwrap()
                        .albedo
                        .as_ref()
                        .unwrap(),
                );
                image_infos.push(
                    vk::DescriptorImageInfoBuilder::new()
                        .image_view(albedo.image_view)
                        .sampler(albedo.sampler)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                );
            }
            let mut storage_info = vec![];
            for image_view in &self.compute_image_views {
                storage_info.push(
                    vk::DescriptorImageInfoBuilder::new()
                        .image_view(*image_view)
                        .image_layout(vk::ImageLayout::GENERAL),
                );
            }
            let depth_info = vec![
                vk::DescriptorImageInfoBuilder::new()
                    .image_view(self.depth_image_view.unwrap())
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL)
                    .sampler(self.depth_sampler.unwrap()),
            ];

            let descriptor_writes = [
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
                    .image_info(&image_infos),
                vk::WriteDescriptorSetBuilder::new()
                    .dst_set(*descriptor_set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&storage_info),
                vk::WriteDescriptorSetBuilder::new()
                    .dst_set(*descriptor_set)
                    .dst_binding(4)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&depth_info),

            ];

            println!("Updating compute descriptor sets");

            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .update_descriptor_sets(&descriptor_writes, &[]);
            }
        }

        let push_constant_range = vk::PushConstantRangeBuilder::new()
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let push_constant_ranges = [push_constant_range];

        let compute_pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .push_constant_ranges(&push_constant_ranges)
            .set_layouts(&descriptor_set_layouts)
            .build();

        println!("Creating compute pipeline layout");

        self.compute_pipeline_layout = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_pipeline_layout(&compute_pipeline_layout_info, None)
            }
            .unwrap(),
        );

        let main_c_string = CString::new("main").unwrap();

        let compute_pipeline_stage_info = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::COMPUTE)
            .module(self.load_shader(std::path::PathBuf::from("shaders/comp.spv")))
            .name(main_c_string.as_c_str());

        let compute_pipeline_infos = [vk::ComputePipelineCreateInfoBuilder::new()
            .stage(*compute_pipeline_stage_info)
            .layout(self.compute_pipeline_layout.unwrap())];
        self.compute_pipelines = unsafe {
            self.device.as_ref().unwrap().create_compute_pipelines(
                None,
                &compute_pipeline_infos,
                None,
            )
        }
        .unwrap();
        println!("compute pipelines: {:?}", self.compute_pipelines.len());
    }

    fn destroy_compute_pipelines(&mut self) {
        unsafe {
            self.device.as_ref().unwrap().destroy_descriptor_set_layout(
                Some(self.compute_descriptor_set_layout.unwrap()),
                None,
            );
            self.device
                .as_ref()
                .unwrap()
                .destroy_descriptor_pool(Some(self.compute_descriptor_pool.unwrap()), None);
            self.device
                .as_ref()
                .unwrap()
                .destroy_pipeline_layout(Some(self.compute_pipeline_layout.unwrap()), None);
            for pipeline in &self.compute_pipelines {
                self.device
                    .as_ref()
                    .unwrap()
                    .destroy_pipeline(Some(*pipeline), None);
            }
        }
    }

    fn create_graphics_pipelines(&mut self) {
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

        let vert_shader_stage_create_info = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::VERTEX)
            .module(self.vert_shader_module.unwrap())
            .name(vert_entry_string.as_c_str());
        let frag_shader_stage_create_info = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::FRAGMENT)
            .module(self.frag_shader_module.unwrap())
            .name(frag_entry_string.as_c_str());
        let shader_stages = vec![vert_shader_stage_create_info, frag_shader_stage_create_info];

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfoBuilder::new()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
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
            .layout(self.pipeline_layout.unwrap())
            .render_pass(self.render_pass.unwrap())
            .subpass(0)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_state_info)];

        self.graphics_pipelines = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_graphics_pipelines(None, &pipeline_infos, None)
        }
        .unwrap();
    }

    fn create_framebuffers(&mut self) {
        self.swapchain_framebuffers = Some(
            self.swapchain_image_views
                .as_ref()
                .unwrap()
                .iter()
                .map(|image_view| {
                    let attachments = if self.msaa_samples != vk::SampleCountFlagBits::_1 {
                        vec![
                            self.color_image_view.unwrap(),
                            self.depth_image_view.unwrap(),
                            *image_view,
                        ]
                    } else {
                        vec![*image_view, self.depth_image_view.unwrap()]
                    };
                    let framebuffer_create_info = vk::FramebufferCreateInfoBuilder::new()
                        .render_pass(self.render_pass.unwrap())
                        .attachments(&attachments)
                        .width(self.surface_capabilities.unwrap().current_extent.width)
                        .height(self.surface_capabilities.unwrap().current_extent.height)
                        .layers(1);
                    return unsafe {
                        self.device
                            .as_ref()
                            .unwrap()
                            .create_framebuffer(&framebuffer_create_info, None)
                    }
                    .unwrap();
                })
                .collect(),
        );
    }

    fn create_command_buffers(&mut self) {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
            .command_pool(self.command_pool.unwrap())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(self.swapchain_framebuffers.as_ref().unwrap().len() as u32);

        self.command_buffers = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .allocate_command_buffers(&command_buffer_allocate_info)
            }
            .unwrap(),
        );

        println!(
            "Num command buffers: {:?}",
            self.command_buffers.as_ref().unwrap().len()
        );
    }

    fn run_commands(&self) {
        self.command_buffers
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .for_each(|(i, command_buffer)| {
                let command_buffer_begin_info = vk::CommandBufferBeginInfoBuilder::new();
                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .begin_command_buffer(*command_buffer, &command_buffer_begin_info)
                }
                .unwrap();

                let clear_colors = vec![
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0],
                        },
                    },
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                ];

                let render_pass_info = vk::RenderPassBeginInfoBuilder::new()
                    .render_pass(self.render_pass.unwrap())
                    .framebuffer(self.swapchain_framebuffers.as_ref().unwrap()[i])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.surface_capabilities.unwrap().current_extent,
                    })
                    .clear_values(&clear_colors);

                unsafe {
                    self.device.as_ref().unwrap().cmd_begin_render_pass(
                        *command_buffer,
                        &render_pass_info,
                        vk::SubpassContents::INLINE,
                    )
                };

                for pipeline in &self.graphics_pipelines {
                    unsafe {
                        self.device.as_ref().unwrap().cmd_bind_pipeline(
                            *command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            *pipeline,
                        )
                    };
                }

                let vertex_buffers = [self.vertex_buffer.unwrap()];
                let offsets = [0 as vk::DeviceSize];
                unsafe {
                    self.device.as_ref().unwrap().cmd_bind_vertex_buffers(
                        *command_buffer,
                        0,
                        &vertex_buffers,
                        &offsets,
                    )
                };

                unsafe {
                    self.device.as_ref().unwrap().cmd_bind_index_buffer(
                        *command_buffer,
                        self.index_buffer.unwrap(),
                        0 as vk::DeviceSize,
                        vk::IndexType::UINT32,
                    )
                };

                unsafe {
                    self.device.as_ref().unwrap().cmd_bind_descriptor_sets(
                        *command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout.unwrap(),
                        0,
                        self.descriptor_sets.as_ref().unwrap(),
                        &[],
                    )
                };

                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_set_depth_test_enable_ext(*command_buffer, false)
                };

                self.cubemap.as_ref().unwrap().draw(
                    self.device.as_ref().unwrap(),
                    *command_buffer,
                    self.pipeline_layout.unwrap(),
                    1.0,
                );

                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_set_depth_test_enable_ext(*command_buffer, true)
                };


                for object in &self.objects {
                    object.draw(
                        self.device.as_ref().unwrap(),
                        *command_buffer,
                        self.pipeline_layout.unwrap(),
                        1.0,
                    );
                }

                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_end_render_pass(*command_buffer)
                };

                //begin compute
                unsafe {
                    self.device.as_ref().unwrap().cmd_bind_pipeline(
                        *command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        self.compute_pipelines[0],
                    )
                };

                unsafe {
                    self.device.as_ref().unwrap().cmd_bind_descriptor_sets(
                        *command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        self.compute_pipeline_layout.unwrap(),
                        0,
                        &self.compute_descriptor_sets.as_ref().unwrap(),
                        &[],
                    )
                };

                for i in 0..self.compute_images.len() {
                    let barrier = vk::ImageMemoryBarrierBuilder::new()
                        .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .src_access_mask(
                            vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ,
                        )
                        .dst_access_mask(
                            vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ,
                        )
                        .image(self.compute_images[i])
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });

                    unsafe {
                        self.device.as_ref().unwrap().cmd_pipeline_barrier(
                            *command_buffer,
                            Some(vk::PipelineStageFlags::ALL_COMMANDS),
                            Some(vk::PipelineStageFlags::ALL_COMMANDS),
                            None,
                            &[],
                            &[],
                            &[barrier],
                        );
                    }
                }

                let push_constant = PushConstants {
                    model: self.cubemap.as_ref().unwrap().render_object.view.try_inverse().unwrap(),
                    view: self.objects[0].view.try_inverse().unwrap(),
                    proj: self.objects[0].proj,
                    texture_index: 0,
                    constant: 1.0,
                    bitfield: 0,
                };
                unsafe {
                    self.device.as_ref().unwrap().cmd_push_constants(
                        *command_buffer,
                        self.compute_pipeline_layout.unwrap(),
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        size_of::<PushConstants>() as u32,
                        &push_constant as *const _ as *const c_void,
                    );
                };

                unsafe {
                    self.device.as_ref().unwrap().cmd_dispatch(
                        *command_buffer,
                        (self.surface_capabilities.unwrap().current_extent.width as f64 / 8.0)
                            .ceil() as u32,
                        (self.surface_capabilities.unwrap().current_extent.height as f64 / 8.0)
                            .ceil() as u32,
                        1,
                    )
                };
                for i in 0..self.compute_images.len() {
                    let barrier = vk::ImageMemoryBarrierBuilder::new()
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .src_access_mask(
                            vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ,
                        )
                        .dst_access_mask(
                            vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ,
                        )
                        .image(self.compute_images[i])
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });

                    unsafe {
                        self.device.as_ref().unwrap().cmd_pipeline_barrier(
                            *command_buffer,
                            Some(vk::PipelineStageFlags::ALL_COMMANDS),
                            Some(vk::PipelineStageFlags::ALL_COMMANDS),
                            None,
                            &[],
                            &[],
                            &[barrier],
                        );
                    }
                }
                //end compute
                unsafe {
                    self.device.as_ref().unwrap().cmd_bind_descriptor_sets(
                        *command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout.unwrap(),
                        0,
                        self.descriptor_sets.as_ref().unwrap(),
                        &[],
                    )
                };


                let clear_colors = vec![];

                let render_pass_info = vk::RenderPassBeginInfoBuilder::new()
                    .render_pass(self.post_process_render_pass.unwrap())
                    .framebuffer(self.swapchain_framebuffers.as_ref().unwrap()[i])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.surface_capabilities.unwrap().current_extent,
                    })
                    .clear_values(&clear_colors);
                unsafe {
                    self.device.as_ref().unwrap().cmd_begin_render_pass(
                        *command_buffer,
                        &render_pass_info,
                        vk::SubpassContents::INLINE,
                    )
                };
                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_set_depth_test_enable_ext(*command_buffer, false)
                };

                for object in &self.fullscreen_quads {
                    object.draw(
                        self.device.as_ref().unwrap(),
                        *command_buffer,
                        self.pipeline_layout.unwrap(),
                        1.0,
                    );
                }
                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_set_depth_test_enable_ext(*command_buffer, true)
                };
                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_end_render_pass(*command_buffer)
                };
                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .end_command_buffer(*command_buffer)
                }
                .unwrap();
            });
    }

    fn recreate_swapchain(&mut self) {
        self.get_surface_capabilities();


        while self.surface_capabilities.unwrap().current_extent.height == 0 ||
            self.surface_capabilities.unwrap().current_extent.width == 0 {
            self.get_surface_capabilities();
            unsafe {
                self.device.as_ref().unwrap().device_wait_idle().unwrap();
            }
            std::thread::sleep(Duration::from_millis(100));
        }


        self.create_color_resources();
        self.create_depth_resources();
        self.create_swapchain();
        self.create_swapchain_image_views();
        self.create_render_pass();
        self.create_graphics_pipelines();

        self.create_framebuffers();
        self.create_uniform_buffers();
        self.create_descriptor_pool();
        self.create_compute_images();

        self.create_compute_pipelines();
        self.create_descriptor_sets();
        self.create_command_buffers();
    }

    fn create_color_resources(&mut self) {
        let color_format = self.surface_format.unwrap().format;
        let (color_image, color_image_memory) = self.create_image_with_memory(
            self.surface_capabilities.unwrap().current_extent.width,
            self.surface_capabilities.unwrap().current_extent.height,
            1,
            self.msaa_samples,
            color_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        self.color_image = Some(color_image);
        self.color_image_memory = Some(color_image_memory);
        self.color_image_view =
            Some(self.create_image_view(color_image, color_format, vk::ImageAspectFlags::COLOR, 1));
    }

    fn create_cubemap_resources(&mut self) {
        let cubemap_folder = PathBuf::from("cubemap_space");

        self.cubemap = Some(Cubemap::new(
            self,
            cubemap_folder.join("StandardCubeMap.png"),
        ));
        self.irradiance_maps.push(CombinedImage::new(
            self,
            cubemap_folder.join("irradiance.hdr"),
            vk::ImageViewType::CUBE,
            vk::Format::R32G32B32A32_SFLOAT,
            false,
        ).unwrap());
                self.environment_maps.push(CombinedImage::new(
                    self,
                    cubemap_folder.join("environment.hdr"),
                    vk::ImageViewType::CUBE,
                    vk::Format::R32G32B32A32_SFLOAT,
                    false,
                ).unwrap());

        self.brdf_lut =
            CombinedImage::new(
                self,
                PathBuf::from("brdf_lut.png"),
                vk::ImageViewType::_2D,
                vk::Format::R8G8B8A8_UNORM,
                false
            );

    }

    fn create_command_pool(&mut self) {
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

    fn find_depth_format(&self) -> vk::Format {
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

    fn create_shader_module(device: &DeviceLoader, spv_code: Vec<u32>) -> vk::ShaderModule {
        let shader_module_create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&spv_code);
        unsafe { device.create_shader_module(&shader_module_create_info, None) }.unwrap()
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            self.device.as_ref().unwrap().device_wait_idle().unwrap();
            self.destroy_compute_images();
            self.destroy_compute_pipelines();
            self.swapchain_framebuffers
                .as_ref()
                .unwrap()
                .into_iter()
                .for_each(|framebuffer| {
                    self.device
                        .as_ref()
                        .unwrap()
                        .destroy_framebuffer(Some(*framebuffer), None)
                });
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
                self.command_buffers.as_ref().unwrap(),
            );

            for pipeline in &self.graphics_pipelines {
                self.device
                    .as_ref()
                    .unwrap()
                    .destroy_pipeline(Some(*pipeline), None)
            }
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
            .as_ref()
            .unwrap()
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
    }

    pub(crate) fn cleanup(&mut self) {
        unsafe { self.device.as_ref().unwrap().device_wait_idle() }.unwrap();

        self.cleanup_swapchain();
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_descriptor_set_layout(Some(self.descriptor_set_layout.unwrap()), None);
            self.destroy_vertex_buffer();
            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(Some(self.index_buffer.unwrap()), None);
            self.device
                .as_ref()
                .unwrap()
                .free_memory(Some(self.index_buffer_memory.unwrap()), None);
        }

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_semaphore(Some(self.render_finished_semaphore.unwrap()), None);
            self.device
                .as_ref()
                .unwrap()
                .destroy_semaphore(Some(self.image_available_semaphore.unwrap()), None);
            self.device
                .as_ref()
                .unwrap()
                .destroy_fence(Some(self.in_flight_fence.unwrap()), None);
        }
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_shader_module(Some(self.vert_shader_module.unwrap()), None)
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_shader_module(Some(self.frag_shader_module.unwrap()), None)
        };
        unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .destroy_surface_khr(Some(self.surface.unwrap()), None)
        };
        unsafe { self.device.as_ref().unwrap().destroy_device(None) };
        unsafe { self.instance.as_ref().unwrap().destroy_instance(None) };
    }

    fn destroy_vertex_buffer(&mut self) {
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
    fn destroy_index_buffer(&mut self) {
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
    pub(crate) fn get_projection_matrix(&self) -> Matrix4<f32> {
        let surface_width = self.surface_capabilities.unwrap().current_extent.width as f32;
        let surface_height = self.surface_capabilities.unwrap().current_extent.height as f32;
        let projection =
            nalgebra::Perspective3::new(surface_width / surface_height,90f32.to_radians() , 1_000_000.0, 20_000_000.0)
                .to_homogeneous();
        return projection;
    }
    pub(crate) fn get_cubemap_projection_matrix(&self) -> Matrix4<f32> {
        let surface_width = self.surface_capabilities.unwrap().current_extent.width as f32;
        let surface_height = self.surface_capabilities.unwrap().current_extent.height as f32;
        let projection =
            nalgebra::Perspective3::new(surface_width / surface_height, 90f32.to_radians(), 0.01, 10.0)
                .to_homogeneous();
        return projection;
    }
    pub(crate) fn transfer_data_to_gpu(&mut self) {
        let random: [f32; NUM_RANDOM] =
            get_random_vector(&self.rng, NUM_RANDOM).try_into().unwrap();
        for i in 0..NUM_RANDOM {
            self.uniform_buffer_object.random[i] =
                Vector4::new(random[i], random[i], random[i], random[i]);
        }

        // for i in 0..1{
        //     // let random_vec = vec![self.rng.u8(..); (self.cpu_images[0].image.width * self.cpu_images[0].image.height) as usize];
        //     let random_vec = std::iter::repeat_with(||self.rng.u8(..)).take((self.cpu_images[0].image.width * self.cpu_images[0].image.height) as usize).collect();
        //     self.cpu_images[i].write_data(random_vec);
        // }

        for i in 0..self.uniform_buffer_pointers.len() {
            unsafe {
                self.uniform_buffer_pointers[i].copy_from_nonoverlapping(
                    &self.uniform_buffer_object as *const UniformBufferObject as *const u8,
                    std::mem::size_of::<UniformBufferObject>(),
                );
            };
        }
    }

    fn create_uniform_buffers(&mut self) {
        let device_size = (std::mem::size_of::<UniformBufferObject>()) as vk::DeviceSize;

        let buffer_create_info = vk::BufferCreateInfoBuilder::new()
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .size(device_size)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_create_info = vk_mem_erupt::AllocationCreateInfo {
            usage: vk_mem_erupt::MemoryUsage::CpuToGpu,
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE,
            ..Default::default()
        };

        let (buffer, allocation, _) = self
            .allocator
            .as_ref()
            .unwrap()
            .create_buffer(&buffer_create_info, &allocation_create_info)
            .unwrap();
        self.uniform_buffers.push(buffer);
        self.uniform_buffer_allocations.push(allocation);
        self.uniform_buffer_pointers.push(
            self.allocator
                .as_ref()
                .unwrap()
                .map_memory(&allocation)
                .unwrap(),
        );
    }

    fn generate_mipmaps(
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
            println!("i: {:}", i);
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

    fn begin_single_time_commands(&self) -> vk::CommandBuffer {
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

    fn end_single_time_commands(&self, command_buffer: vk::CommandBuffer) {
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

    fn transition_image_layout(
        &self,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        subresource_range: vk::ImageSubresourceRange,
    ) {
        let command_buffer = self.begin_single_time_commands();

        let mut barrier = vk::ImageMemoryBarrierBuilder::new()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(subresource_range);

        let source_stage: vk::PipelineStageFlags;
        let destination_stage: vk::PipelineStageFlags;

        match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => {
                barrier = barrier.src_access_mask(vk::AccessFlags::empty());
                barrier = barrier.dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
                destination_stage = vk::PipelineStageFlags::TRANSFER;
            }
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => {
                barrier = barrier.src_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                barrier = barrier.dst_access_mask(vk::AccessFlags::SHADER_READ);

                source_stage = vk::PipelineStageFlags::TRANSFER;
                destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
            }
            _ => panic!("Transition not supported"),
        }

        unsafe {
            self.device.as_ref().unwrap().cmd_pipeline_barrier(
                command_buffer,
                Some(source_stage),
                Some(destination_stage),
                None,
                &[],
                &[],
                &[barrier],
            )
        };

        self.end_single_time_commands(command_buffer);
    }

    fn create_image_view(
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

    fn copy_buffer_to_image(&self, buffer: vk::Buffer, image: vk::Image, width: u32, height: u32) {
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

    fn create_image_with_memory(
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
