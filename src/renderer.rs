use ash::{vk, Device, Instance, Entry};
use crate::support::*;
use crate::cube::*;
use crate::game::*;
use cgmath::{Matrix4, Vector3, SquareMatrix, Deg, Vector2, Vector4, Point3};
use ash::extensions::khr::{Surface, Swapchain};
use image::GenericImageView;
use winit::window::Window;
use std::ffi::{c_void, CString, CStr};
use std::convert::TryInto;
use ash::extensions::ext::DebugUtils;
use crate::octree::Node;
use crate::{main, octree};
use ash::vk::{CommandBuffer, PipelineLayout};
const BASE_VOXEL_SIZE: f32 = 128.0;

struct CombinedImage{
    image: vk::Image,
    image_view: vk::ImageView,
    sampler: vk::Sampler,
    allocation: vk_mem::Allocation,
    index: u32,
    width: u32,
    height: u32,
}

pub (crate) struct RenderObject {
    vertex_start: u32,
    vertex_count: u32,
    index_start: u32,
    index_count: u32,
    texture: Option<CombinedImage>,
    pub(crate) model: Matrix4<f32>,
    pub(crate) view: Matrix4<f32>,
    pub(crate) proj: Matrix4<f32>,
}

impl RenderObject {
    fn new(vulkan_data: &mut VulkanData, vertices: &Vec<Vertex>, indices: &Vec<u32>, texture_path: Option<std::path::PathBuf> ) -> Self{

        let vertex_start = vulkan_data.vertices.len() as u32;
        let index_start = vulkan_data.indices.len() as u32;

        vulkan_data.vertices.extend(vertices);

        for index in indices{
            vulkan_data.indices.push(index+vertex_start);
        }


        let mut texture = None;
        if texture_path.is_some() {
            let dynamic_image = image::io::Reader::open(texture_path.unwrap()).unwrap().decode().unwrap();

            let width = dynamic_image.width();
            let height = dynamic_image.height();
            let pixels = dynamic_image.into_rgba8().into_raw();

            let format = vk::Format::R8G8B8A8_SRGB;

            let image_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .extent(vk::Extent3D{
                    width,
                    height,
                    depth: 1
                })
                .mip_levels(1)
                .array_layers(1)
                .format(format)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .samples(vk::SampleCountFlags::TYPE_1)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo{
                usage: vk_mem::MemoryUsage::GpuOnly,
                required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                ..Default::default()
            };

            let (image, allocation, _) = vulkan_data.allocator.as_ref().unwrap().create_image(&image_info,&allocation_info).unwrap();

            let image_view = vulkan_data.create_image_view(image,format,vk::ImageAspectFlags::COLOR, 1);

            let sampler_info = vk::SamplerCreateInfo::builder()
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

            let sampler = unsafe{vulkan_data.device.as_ref().unwrap().create_sampler(&sampler_info,None).unwrap()};


            let buffer_info = vk::BufferCreateInfo::builder()
                .size(pixels.len() as u64)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC);

            let buffer_allocation_create_info = vk_mem::AllocationCreateInfo{
                usage: vk_mem::MemoryUsage::CpuOnly,
                flags: vk_mem::AllocationCreateFlags::MAPPED,
                ..Default::default()
            };
            let (staging_buffer, staging_buffer_allocation, staging_buffer_allocation_info) = vulkan_data.allocator.as_ref().unwrap().create_buffer(&buffer_info,&buffer_allocation_create_info).unwrap();
            unsafe{staging_buffer_allocation_info.get_mapped_data().copy_from_nonoverlapping(pixels.as_ptr(), pixels.len())};
            vulkan_data.transition_image_layout(image,vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,1);
            vulkan_data.copy_buffer_to_image(staging_buffer,image, width, height);
            vulkan_data.allocator.as_ref().unwrap().destroy_buffer(staging_buffer, &staging_buffer_allocation);
            vulkan_data.transition_image_layout(image,vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, 1   );

            texture = Some(CombinedImage{
                image,
                image_view,
                sampler,
                allocation,
                index: vulkan_data.current_texture_index,
                width,
                height
            });
            vulkan_data.current_texture_index += 1

        }


        let object = RenderObject {
            vertex_start,
            vertex_count: vertices.len() as u32,
            index_start,
            index_count: indices.len() as u32,
            texture,
            model: Matrix4::identity(),
            view: Matrix4::identity(),
            proj: Matrix4::identity()
        };
        return object;
    }
}

impl Drawable for RenderObject {
    fn draw(&self, device: &Device, command_buffer: vk::CommandBuffer, pipeline_layout: vk::PipelineLayout, frag_constant: f32) {

        unsafe {
            let texture_index = match self.texture.as_ref(){
                Some(image) => image.index,
                None => 0
            };
            let push_constant = PushConstants{
                model: self.model,
                view: self.view,
                proj: self.proj,
                texture_index,
                constant: frag_constant,
            };
            let constants = std::slice::from_raw_parts(
                (&push_constant as *const PushConstants) as *const u8,
                std::mem::size_of::<PushConstants>(),
            );
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                constants,
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
            unsafe {
                device.cmd_draw(
                    command_buffer,
                    self.vertex_count,
                    1,
                    self.vertex_start,
                    0,
                )
            };
        }
    }
}

trait Drawable {
    fn draw(&self, device: &Device, command_buffer: vk::CommandBuffer, pipeline_layout: vk::PipelineLayout, frag_constant: f32);
}
struct Cubemap{
    positive_x: RenderObject,
    negative_x: RenderObject,
    positive_y: RenderObject,
    negative_y: RenderObject,
    positive_z: RenderObject,
    negative_z: RenderObject,
}

impl Cubemap{
    fn new(
        vulkan_data: &mut VulkanData,
        positive_x: std::path::PathBuf,
        negative_x: std::path::PathBuf,
        positive_y: std::path::PathBuf,
        negative_y: std::path::PathBuf,
        positive_z: std::path::PathBuf,
        negative_z: std::path::PathBuf,
    ) -> Self{

        let indices = &QUAD_INDICES.to_vec();

        let cubemap = Cubemap{
            positive_x: RenderObject::new(vulkan_data, &POSITIVE_X_VERTICES.to_vec(), indices, Some(positive_x)),
            negative_x: RenderObject::new(vulkan_data, &NEGATIVE_X_VERTICES.to_vec(), indices, Some(negative_x)),
            positive_y: RenderObject::new(vulkan_data, &POSITIVE_Y_VERTICES.to_vec(), indices, Some(positive_y)),
            negative_y: RenderObject::new(vulkan_data, &NEGATIVE_Y_VERTICES.to_vec(), indices, Some(negative_y)),
            positive_z: RenderObject::new(vulkan_data, &POSITIVE_Z_VERTICES.to_vec(), indices, Some(positive_z)),
            negative_z: RenderObject::new(vulkan_data, &NEGATIVE_Z_VERTICES.to_vec(), indices, Some(negative_z)),
        };

        return cubemap;
    }
    fn get_image_infos(&self) -> Vec<vk::DescriptorImageInfo>{

        return vec![
            vk::DescriptorImageInfo::builder().image_view(self.positive_x.texture.as_ref().unwrap().image_view).sampler(self.positive_x.texture.as_ref().unwrap().sampler).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).build(),
            vk::DescriptorImageInfo::builder().image_view(self.negative_x.texture.as_ref().unwrap().image_view).sampler(self.negative_x.texture.as_ref().unwrap().sampler).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).build(),
            vk::DescriptorImageInfo::builder().image_view(self.positive_y.texture.as_ref().unwrap().image_view).sampler(self.positive_y.texture.as_ref().unwrap().sampler).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).build(),
            vk::DescriptorImageInfo::builder().image_view(self.negative_y.texture.as_ref().unwrap().image_view).sampler(self.negative_y.texture.as_ref().unwrap().sampler).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).build(),
            vk::DescriptorImageInfo::builder().image_view(self.positive_z.texture.as_ref().unwrap().image_view).sampler(self.positive_y.texture.as_ref().unwrap().sampler).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).build(),
            vk::DescriptorImageInfo::builder().image_view(self.negative_z.texture.as_ref().unwrap().image_view).sampler(self.negative_y.texture.as_ref().unwrap().sampler).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).build(),

        ];

    }


}

impl Cubemap{
    fn process(&mut self,camera: &Camera, proj: Matrix4<f32>) {
        let view = camera.get_view_matrix_no_translation();
        self.positive_x.view = view;
        self.negative_x.view = view;
        self.positive_y.view = view;
        self.negative_y.view = view;
        self.positive_z.view = view;
        self.negative_z.view = view;

        self.positive_x.model = Matrix4::identity();
        self.negative_x.model = Matrix4::identity();
        self.positive_y.model = Matrix4::identity();
        self.negative_y.model = Matrix4::identity();
        self.positive_z.model = Matrix4::identity();
        self.negative_z.model = Matrix4::identity();

        self.positive_x.proj = proj;
        self.negative_x.proj = proj;
        self.positive_y.proj = proj;
        self.negative_y.proj = proj;
        self.positive_z.proj = proj;
        self.negative_z.proj = proj;

    }
}
impl Drawable for Cubemap{
    fn draw(&self, device: &Device, command_buffer: vk::CommandBuffer, pipeline_layout: vk::PipelineLayout, frag_constant: f32) {

        self.positive_x.draw(device, command_buffer, pipeline_layout,frag_constant);
        self.negative_x.draw(device, command_buffer, pipeline_layout,frag_constant);
        self.positive_y.draw(device, command_buffer, pipeline_layout,frag_constant);
        self.negative_y.draw(device, command_buffer, pipeline_layout,frag_constant);
        self.positive_z.draw(device, command_buffer, pipeline_layout,frag_constant);
        self.negative_z.draw(device, command_buffer, pipeline_layout,frag_constant);

    }
}

struct Voxel{
    position: Vector3<f32>,
    size: f32,
    positive_x: RenderObject,
    negative_x: RenderObject,
    positive_y: RenderObject,
    negative_y: RenderObject,
    positive_z: RenderObject,
    negative_z: RenderObject,
}

impl Drawable for Voxel{
    fn draw(&self, device: &Device, command_buffer: CommandBuffer, pipeline_layout: PipelineLayout, frag_constant: f32) {
        self.positive_x.draw(device, command_buffer, pipeline_layout,frag_constant);
        self.negative_x.draw(device, command_buffer, pipeline_layout,frag_constant);
        self.positive_y.draw(device, command_buffer, pipeline_layout,frag_constant);
        self.negative_y.draw(device, command_buffer, pipeline_layout,frag_constant);
        self.positive_z.draw(device, command_buffer, pipeline_layout,frag_constant);
        self.negative_z.draw(device, command_buffer, pipeline_layout,frag_constant);

    }
}

struct Voxels{
    voxels: Vec<Voxel>,
    positive_x_vertex_offset: u32,
    negative_x_vertex_offset: u32,
    positive_y_vertex_offset: u32,
    negative_y_vertex_offset: u32,
    positive_z_vertex_offset: u32,
    negative_z_vertex_offset: u32,
    positive_x_index_offset: u32,
    negative_x_index_offset: u32,
    positive_y_index_offset: u32,
    negative_y_index_offset: u32,
    positive_z_index_offset: u32,
    negative_z_index_offset: u32,

}
impl Voxels{
    fn new(vulkan_data: &mut VulkanData) -> Self{
        let indices = &QUAD_INDICES.to_vec();
        let voxel = Voxel{
            position: Vector3::new(0.0,0.0,-5.0),
            size: 1.0,
            positive_x: RenderObject::new(vulkan_data, &POSITIVE_X_VERTICES.to_vec(), indices, None),
            negative_x: RenderObject::new(vulkan_data, &NEGATIVE_X_VERTICES.to_vec(), indices, None),
            positive_y: RenderObject::new(vulkan_data, &POSITIVE_Y_VERTICES.to_vec(), indices, None),
            negative_y: RenderObject::new(vulkan_data, &NEGATIVE_Y_VERTICES.to_vec(), indices, None),
            positive_z: RenderObject::new(vulkan_data, &POSITIVE_Z_VERTICES.to_vec(), indices, None),
            negative_z: RenderObject::new(vulkan_data, &NEGATIVE_Z_VERTICES.to_vec(), indices, None),
        };

        return Self{
            positive_x_vertex_offset: voxel.positive_x.vertex_start,
            negative_x_vertex_offset: voxel.negative_x.vertex_start,
            positive_y_vertex_offset: voxel.positive_y.vertex_start,
            negative_y_vertex_offset: voxel.negative_y.vertex_start,
            positive_z_vertex_offset: voxel.positive_z.vertex_start,
            negative_z_vertex_offset: voxel.negative_z.vertex_start,
            positive_x_index_offset: voxel.positive_x.index_start,
            negative_x_index_offset: voxel.negative_x.index_start,
            positive_y_index_offset: voxel.positive_y.index_start,
            negative_y_index_offset: voxel.negative_y.index_start,
            positive_z_index_offset: voxel.positive_z.index_start,
            negative_z_index_offset: voxel.negative_z.index_start,
            voxels: vec![voxel],
        };

    }
    fn add_voxel(&mut self, vulkan_data: &mut VulkanData, position: Vector3<f32>, size: f32){
        let new_voxel = Voxel{
            position,
            size,
            positive_x: RenderObject {
                vertex_start: self.positive_x_vertex_offset,
                vertex_count: POSITIVE_X_VERTICES.len() as u32,
                index_start: self.positive_x_index_offset,
                index_count: QUAD_INDICES.len() as u32,
                texture: None,
                model: Matrix4::identity(),
                view: Matrix4::identity(),
                proj: Matrix4::identity()
            },
            negative_x: RenderObject {
                vertex_start: self.negative_x_vertex_offset,
                vertex_count: NEGATIVE_X_VERTICES.len() as u32,
                index_start: self.negative_x_index_offset,
                index_count: QUAD_INDICES.len() as u32,
                texture: None,
                model: Matrix4::identity(),
                view: Matrix4::identity(),
                proj: Matrix4::identity()

            },
            positive_y: RenderObject {
                vertex_start: self.positive_y_vertex_offset,
                vertex_count: POSITIVE_Y_VERTICES.len() as u32,
                index_start: self.positive_y_index_offset,
                index_count: QUAD_INDICES.len() as u32,
                texture: None,
                model: Matrix4::identity(),
                view: Matrix4::identity(),
                proj: Matrix4::identity()

            },
            negative_y: RenderObject {
                vertex_start: self.negative_y_vertex_offset,
                vertex_count: NEGATIVE_Y_VERTICES.len() as u32,
                index_start: self.negative_y_index_offset,
                index_count: QUAD_INDICES.len() as u32,
                texture: None,
                model: Matrix4::identity(),
                view: Matrix4::identity(),
                proj: Matrix4::identity()

            },
            positive_z: RenderObject {
                vertex_start: self.positive_z_vertex_offset,
                vertex_count: POSITIVE_Z_VERTICES.len() as u32,
                index_start: self.positive_z_index_offset,
                index_count: QUAD_INDICES.len() as u32,
                texture: None,
                model: Matrix4::identity(),
                view: Matrix4::identity(),
                proj: Matrix4::identity()

            },
            negative_z: RenderObject {
                vertex_start: self.negative_z_vertex_offset,
                vertex_count: NEGATIVE_Z_VERTICES.len() as u32,
                index_start: self.negative_z_index_offset,
                index_count: QUAD_INDICES.len() as u32,
                texture: None,
                model: Matrix4::identity(),
                view: Matrix4::identity(),
                proj: Matrix4::identity()
            },
        };
        self.voxels.push(new_voxel);
    }

    fn process(&mut self, view_matrix: Matrix4<f32>, projection_matrix: Matrix4<f32>){
        for mut voxel in 0..self.voxels.len(){
            self.voxels[voxel].positive_x.model = Matrix4::from_translation(self.voxels[voxel].position) * Matrix4::from_scale(self.voxels[voxel].size);
            self.voxels[voxel].negative_x.model = Matrix4::from_translation(self.voxels[voxel].position) * Matrix4::from_scale(self.voxels[voxel].size);
            self.voxels[voxel].positive_y.model = Matrix4::from_translation(self.voxels[voxel].position) * Matrix4::from_scale(self.voxels[voxel].size);
            self.voxels[voxel].negative_y.model = Matrix4::from_translation(self.voxels[voxel].position) * Matrix4::from_scale(self.voxels[voxel].size);
            self.voxels[voxel].positive_z.model = Matrix4::from_translation(self.voxels[voxel].position) * Matrix4::from_scale(self.voxels[voxel].size);
            self.voxels[voxel].negative_z.model = Matrix4::from_translation(self.voxels[voxel].position) * Matrix4::from_scale(self.voxels[voxel].size);

            self.voxels[voxel].positive_x.view = view_matrix;
            self.voxels[voxel].negative_x.view = view_matrix;
            self.voxels[voxel].positive_y.view = view_matrix;
            self.voxels[voxel].negative_y.view = view_matrix;
            self.voxels[voxel].positive_z.view = view_matrix;
            self.voxels[voxel].negative_z.view = view_matrix;

            self.voxels[voxel].positive_x.proj = projection_matrix;
            self.voxels[voxel].negative_x.proj = projection_matrix;
            self.voxels[voxel].positive_y.proj = projection_matrix;
            self.voxels[voxel].negative_y.proj = projection_matrix;
            self.voxels[voxel].positive_z.proj = projection_matrix;
            self.voxels[voxel].negative_z.proj = projection_matrix;

        }
    }
}


impl Drawable for Voxels{
    fn draw(&self, device: &Device, command_buffer: CommandBuffer, pipeline_layout: PipelineLayout, frag_constant: f32) {
        for voxel in &self.voxels{
            voxel.draw(device, command_buffer, pipeline_layout,frag_constant);
        }
    }
}
pub(crate) struct VulkanData {
    rng: fastrand::Rng,
    instance: Option<Instance>,
    entry: Option<Entry>,
    pub(crate) device: Option<Device>,
    physical_device: Option<vk::PhysicalDevice>,
    allocator: Option<vk_mem::Allocator>,
    main_queue: Option<vk::Queue>,
    main_queue_index: Option<u32>,
    surface_loader: Option<Surface>,
    surface: Option<vk::SurfaceKHR>,
    surface_format: Option<vk::SurfaceFormatKHR>,
    pub(crate) surface_capabilities: Option<vk::SurfaceCapabilitiesKHR>,
    swapchain_loader: Option<Swapchain>,
    swapchain: Option<vk::SwapchainKHR>,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Option<Vec<vk::ImageView>>,
    compute_images: Vec<vk::Image>,
    compute_image_allocations: Vec<vk_mem::Allocation>,
    compute_image_views: Vec<vk::ImageView>,
    compute_samplers: Vec<vk::Sampler>,
    vert_shader_module: Option<vk::ShaderModule>,
    frag_shader_module: Option<vk::ShaderModule>,
    pipeline_layout: Option<vk::PipelineLayout>,
    compute_pipeline_layout: Option<vk::PipelineLayout>,
    render_pass: Option<vk::RenderPass>,
    graphics_pipelines: Vec<vk::Pipeline>,
    compute_pipelines: Vec<vk::Pipeline>,
    swapchain_framebuffers: Option<Vec<vk::Framebuffer>>,
    command_pool: Option<vk::CommandPool>,
    command_buffers: Option<Vec<vk::CommandBuffer>>,
    image_available_semaphore: Option<vk::Semaphore>,
    render_finished_semaphore: Option<vk::Semaphore>,
    in_flight_fence: Option<vk::Fence>,
    vertex_buffer: Option<vk::Buffer>,
    vertex_buffer_memory: Option<vk::DeviceMemory>,
    index_buffer: Option<vk::Buffer>,
    index_buffer_memory: Option<vk::DeviceMemory>,
    mip_levels: u32,
    depth_image: Option<vk::Image>,
    depth_image_memory: Option<vk::DeviceMemory>,
    depth_image_view: Option<vk::ImageView>,
    color_image: Option<vk::Image>,
    color_image_memory: Option<vk::DeviceMemory>,
    color_image_view: Option<vk::ImageView>,
    cubemap: Option<Cubemap>,
    fullscreen_quads: Vec<RenderObject>,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    uniform_buffer_pointers: Vec<*mut u8>,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffer_allocations: Vec<vk_mem::Allocation>,
    pub(crate) uniform_buffer_object: UniformBufferObject,
    storage_buffer_pointer: Option<*mut u8>,
    storage_buffer: Option<vk::Buffer>,
    storage_buffer_allocation: Option<vk_mem::Allocation>,
    pub(crate)nodes: Vec<Node>,
    voxels: Option<Voxels>,
    msaa_samples: vk::SampleCountFlags,
    descriptor_pool: Option<vk::DescriptorPool>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_sets: Option<Vec<vk::DescriptorSet>>,
    compute_descriptor_pool: Option<vk::DescriptorPool>,
    compute_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    compute_descriptor_sets: Option<Vec<vk::DescriptorSet>>,
    last_frame_instant: std::time::Instant,
    current_texture_index: u32,
    pub(crate) objects: Vec<RenderObject>,
    pub(crate) player_object_index: usize,
}

fn get_random_vector(rng: &fastrand::Rng, length:usize) -> Vec<f32>{
    let mut vector = Vec::new();
    for _ in 0..length{
        vector.push(rng.f32());
    }
    return vector;
}

impl VulkanData {
    pub(crate) fn new(nodes: Vec<Node>) -> Self {

        let uniform_buffer_object = UniformBufferObject {
            random: [Vector4::new(0.0f32, 0.0f32, 0.0f32, 0.0f32,); NUM_RANDOM],
            player_index: 0,
            value2: 0,
            value3: 0,
            value4: 0,
            mouse_position: Vector2::new(0.0, 0.0),

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
            surface_loader: None,
            surface: None,
            surface_format: None,
            surface_capabilities: None,
            swapchain_loader: None,
            swapchain: None,
            swapchain_images: vec![],
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
            graphics_pipelines: vec![],
            compute_pipelines: vec![],
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
            depth_image_memory: None,
            depth_image_view: None,
            color_image: None,
            color_image_memory: None,
            color_image_view: None,
            msaa_samples: vk::SampleCountFlags::TYPE_1,
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
            nodes,
            last_frame_instant: std::time::Instant::now(),
            cubemap: None,
            uniform_buffer_pointers: vec![],
            uniform_buffers: vec![],
            uniform_buffer_allocations: vec![],
            current_texture_index: 0,
            objects: vec![],
            fullscreen_quads: vec![],
            player_object_index: 0,
            voxels: None
        };
    }
    pub(crate) fn init_vulkan(&mut self, window: &Window) {
        let validation_layer_names =
            [CString::new("VK_LAYER_KHRONOS_validation").expect("CString conversion failed")];
        let validation_layer_names_raw = validation_layer_names
            .iter()
            .map(|c_string| c_string.as_ptr())
            .collect::<Vec<_>>();
        // validation_layers.remove(0);
        // println!("{:}", validation_layers[0]);

        self.entry = Some(unsafe { ash::EntryCustom::new()}.unwrap());

        self.entry.as_ref().unwrap().enumerate_instance_extension_properties().unwrap().into_iter().for_each(|extension_property| {
            println!("Supported Extension: {:}", String::from_utf8_lossy( &unsafe{std::mem::transmute::<[i8; 256], [u8;256]>(extension_property.extension_name)}));
        });


        println!("Verts: {:}", self.vertices.len());

        let app_info = vk::ApplicationInfo::builder();
        let mut surface_extensions = ash_window::enumerate_required_extensions(window).unwrap();
        surface_extensions.push(DebugUtils::name());
        // surface_extensions.push(vk::ImgFilterCubicFn::name());

        let raw_extensions = surface_extensions
            .iter()
            .map(|extension| extension.to_owned().as_ptr())
            .collect::<Vec<_>>();

        let enables = [vk::ValidationFeatureEnableEXT::BEST_PRACTICES];
        let mut features = vk::ValidationFeaturesEXT::builder().enabled_validation_features(&enables);

        let create_info = vk::InstanceCreateInfo::builder()
            .enabled_extension_names(&raw_extensions)
            .application_info(&app_info)
            .enabled_layer_names(&validation_layer_names_raw)
            .push_next(&mut features);
        self.instance = Some(
            unsafe {
                self.entry
                    .as_ref()
                    .unwrap()
                    .create_instance(&create_info, None)
            }
                .expect("Failed to create instance"),
        );


        let physical_devices =
            unsafe { self.instance.as_ref().unwrap().enumerate_physical_devices() }.unwrap();
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

        // self.msaa_samples = self.get_max_usable_sample_count();
        self.msaa_samples = vk::SampleCountFlags::TYPE_1;

        println!("Samples: {:?}", self.msaa_samples);

        self.main_queue_index = Some(
            unsafe {
                self.instance
                    .as_ref()
                    .unwrap()
                    .get_physical_device_queue_family_properties(self.physical_device.unwrap())
            }
                .iter()
                .position(|queue| queue.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .unwrap() as u32,
        );
        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(self.main_queue_index.unwrap())
            .queue_priorities(&[1.0f32]);
        let queue_create_infos = &[queue_create_info.build()];

        let device_features = vk::PhysicalDeviceFeatures::builder().sampler_anisotropy(true);

        let mut extended_dynamic_state_features = vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::builder().extended_dynamic_state(true);

        let device_extension_names_raw = vec![Swapchain::name().as_ptr(),vk::ExtExtendedDynamicStateFn::name().as_ptr()];
        let device_create_info = vk::DeviceCreateInfo::builder()
            .push_next(&mut extended_dynamic_state_features)
            .queue_create_infos(queue_create_infos)
            .enabled_layer_names(&validation_layer_names_raw)
            .enabled_extension_names(&device_extension_names_raw)
            .enabled_features(&device_features);
        self.device = Some(
            unsafe {
                self.instance.as_ref().unwrap().create_device(
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
        self.create_image_views();
        println!("Creating uniform buffers");
        self.create_uniform_buffers();
        self.create_storage_buffer();

        self.create_descriptor_set_layout();
        self.create_render_pass();


        self.create_command_pool();


        // self.load_obj_model(std::path::PathBuf::from("models/viking_room/viking_room.obj"), std::path::PathBuf::from("models/viking_room/viking_room.png"));
        self.player_object_index = self.load_obj_model(std::path::PathBuf::from("models/person/person.obj"), std::path::PathBuf::from("models/person/person.png"));
        self.uniform_buffer_object.player_index = self.player_object_index as u32;

        self.create_color_resources();
        self.create_depth_resources();
        self.create_cubemap_resources();
        self.create_compute_images();
        for _ in 0..1 {
            let mut quad = RenderObject::new(self, &POSITIVE_Z_VERTICES.to_vec(), &QUAD_INDICES.to_vec(), None);
            quad.texture = Some(CombinedImage {
                image: self.compute_images[0],
                image_view: self.compute_image_views[0],
                sampler: self.compute_samplers[0],
                allocation: self.compute_image_allocations[0],
                index: self.current_texture_index,
                width: self.surface_capabilities.unwrap().current_extent.width,
                height: self.surface_capabilities.unwrap().current_extent.height
            });
            self.current_texture_index += 1;
            self.fullscreen_quads.push(quad);
        }

        self.create_voxels();
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

    fn create_voxels(&mut self){
        let mut voxels = Voxels::new(self);
        for node_index in 0..self.nodes.len(){
            if self.nodes[node_index].node_type == octree::node_types::REFLECTIVE{
                let mut voxel_position =  Vector3::new(0.0, -BASE_VOXEL_SIZE, 0.0);
                for i in 0..self.nodes[node_index].current_depth + 1{
                    let branch = self.nodes[node_index].path[i as usize];
                    let voxel_step_size = 0.5*BASE_VOXEL_SIZE* 0.5f32.powi(i as i32);
                    voxel_position += Vector3::new(
                        ((branch % 2) as f32) * voxel_step_size,
                        ((branch/4) as f32) * voxel_step_size,
                        (((branch/2) % 2) as f32) * voxel_step_size
                    );

                }
                voxels.add_voxel(self,
                                 voxel_position + Vector3::new(0.5*BASE_VOXEL_SIZE * 0.5f32.powi(self.nodes[node_index].current_depth as i32),
                                                               0.5*BASE_VOXEL_SIZE * 0.5f32.powi(self.nodes[node_index].current_depth as i32),
                                                               0.5*BASE_VOXEL_SIZE * 0.5f32.powi(self.nodes[node_index].current_depth as i32)),
                                 0.5*BASE_VOXEL_SIZE * 0.5f32.powi(self.nodes[node_index].current_depth as i32))
            }
        }
        self.voxels = Some(voxels);

    }

    fn create_compute_images(&mut self){
        for _ in 0..1{
            let image_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .extent(vk::Extent3D{
                    width: self.surface_capabilities.unwrap().current_extent.width,
                    height: self.surface_capabilities.unwrap().current_extent.height,
                    depth: 1
                })
                .mip_levels(1)
                .array_layers(1)
                .format(self.surface_format.unwrap().format)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
                .samples(vk::SampleCountFlags::TYPE_1)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let allocation_info = vk_mem::AllocationCreateInfo{
                usage: vk_mem::MemoryUsage::GpuOnly,
                ..Default::default()
            };

            let (image, allocation, _) = self.allocator.as_ref().unwrap().create_image(&image_info, &allocation_info).unwrap();

            let image_view = self.create_image_view(image,image_info.format,vk::ImageAspectFlags::COLOR, 1);

            let sampler_info = vk::SamplerCreateInfo::builder()
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

            let sampler = unsafe{self.device.as_ref().unwrap().create_sampler(&sampler_info,None).unwrap()};

            let command_buffer = self.begin_single_time_commands();

            let barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange{
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1
                });

            unsafe{
                self.device.as_ref().unwrap().cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier.build()]
                )
            }

            self.end_single_time_commands(command_buffer);

            self.compute_images.push(image);
            self.compute_image_allocations.push(allocation);
            self.compute_image_views.push(image_view);
            self.compute_samplers.push(sampler);

        }



    }

    fn destroy_compute_images(&mut self){
        for i in 0..self.compute_images.len(){
            let image = self.compute_images.remove(i);
            let allocation = self.compute_image_allocations.remove(i);
            let image_view = self.compute_image_views.remove(i);
            let sampler = self.compute_samplers.remove(i);
            self.allocator.as_ref().unwrap().destroy_image(image,&allocation);
            unsafe{self.device.as_ref().unwrap().destroy_image_view(image_view, None)};
            unsafe{self.device.as_ref().unwrap().destroy_sampler(sampler, None)};
        }
    }

    fn create_allocator(&mut self){
        let allocator_create_info = vk_mem::AllocatorCreateInfo{
            physical_device: self.physical_device.unwrap(),
            device: self.device.as_ref().unwrap().clone(),
            instance: self.instance.as_ref().unwrap().clone(),
            flags: vk_mem::AllocatorCreateFlags::NONE,
            preferred_large_heap_block_size: 0,
            frame_in_use_count: 0,
            heap_size_limits: None
        };
        self.allocator = Some(vk_mem::Allocator::new(&allocator_create_info).unwrap());

    }

    fn create_surface(&mut self, window: &Window){
        self.surface = Some(
            unsafe {
                ash_window::create_surface(
                    self.entry.as_ref().unwrap(),
                    self.instance.as_ref().unwrap(),
                    window,
                    None,
                )
            }
                .unwrap(),
        );
        self.surface_loader = Some(Surface::new(
            self.entry.as_ref().unwrap(),
            self.instance.as_ref().unwrap(),
        ));
    }

    fn create_sync_objects(&mut self){
        let semaphore_create_info = vk::SemaphoreCreateInfo::builder();
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
            vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
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

    fn get_max_usable_sample_count(&self) -> vk::SampleCountFlags {
        let physical_device_properties = unsafe{ self.instance.as_ref().unwrap().get_physical_device_properties(self.physical_device.unwrap())};
        let counts = physical_device_properties.limits.framebuffer_color_sample_counts & physical_device_properties.limits.framebuffer_depth_sample_counts;

        if counts.contains(vk::SampleCountFlags::TYPE_64){return vk::SampleCountFlags::TYPE_64}
        else if counts.contains(vk::SampleCountFlags::TYPE_32){return vk::SampleCountFlags::TYPE_32}
        else if counts.contains(vk::SampleCountFlags::TYPE_16){return vk::SampleCountFlags::TYPE_16}
        else if counts.contains(vk::SampleCountFlags::TYPE_8){return vk::SampleCountFlags::TYPE_8}
        else if counts.contains(vk::SampleCountFlags::TYPE_4){return vk::SampleCountFlags::TYPE_4}
        else if counts.contains(vk::SampleCountFlags::TYPE_2){return vk::SampleCountFlags::TYPE_2}
        else if counts.contains(vk::SampleCountFlags::TYPE_1){return vk::SampleCountFlags::TYPE_1}
        else {panic!("No samples found???")}
    }
    // fn load_gltf_model(&mut self, path: std::path::PathBuf){
    //
    // }

    pub(crate) fn load_obj_model(&mut self, path: std::path::PathBuf, texture_path: std::path::PathBuf) -> usize{


        let (models, _) = tobj::load_obj(path, &tobj::LoadOptions{
            single_index: true  ,
            triangulate: false,
            ignore_points: false,
            ignore_lines: false,
        }).unwrap();
        // let materials = material_result.unwrap();

        let mut vertices = vec![];

        for model in models{
            for index in model.mesh.indices{
                // self.indices.push(index);
                vertices.push(Vertex{
                    position: Vector3 {
                        x: model.mesh.positions[(index*3 + 0) as usize],
                        y: model.mesh.positions[(index*3 + 1) as usize],
                        z: model.mesh.positions[(index*3 + 2) as usize]
                    },
                    color: Vector3 {
                        x: 1.0,
                        y: 1.0,
                        z: 1.0
                    },
                    texture_coordinate: Vector2 {
                        x: model.mesh.texcoords[(index*2 + 0) as usize],
                        y: 1.0 - model.mesh.texcoords[(index*2 + 1) as usize]
                    }
                });
            }

        }
        let new_object = RenderObject::new(self, &vertices, &vec![], Some(texture_path));
        self.objects.push(new_object);
        return self.objects.len() - 1;
    }

    fn create_depth_resources(&mut self){

        let depth_format = self.find_depth_format();

        let (depth_image, depth_image_memory) = self.create_image_with_memory(
            self.surface_capabilities.unwrap().current_extent.width,
            self.surface_capabilities.unwrap().current_extent.height,
            1,
            self.msaa_samples,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        );

        self.depth_image = Some(depth_image);
        self.depth_image_memory = Some(depth_image_memory);
        self.depth_image_view = Some(self.create_image_view(depth_image, vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH, 1));


    }

    fn get_surface_support (&mut self){
        unsafe{ self.surface_loader.as_ref().unwrap().get_physical_device_surface_support(self.physical_device.unwrap(), self.main_queue_index.unwrap(),self.surface.unwrap())}.unwrap();
    }

    fn get_surface_capabilities(&mut self) {
        self.surface_capabilities = Some(
            unsafe {
                self.surface_loader
                    .as_ref()
                    .unwrap()
                    .get_physical_device_surface_capabilities(
                        self.physical_device.unwrap(),
                        self.surface.unwrap(),
                    )
            }
                .unwrap(),
        );
    }

    fn create_descriptor_pool(&mut self){
        let pool_sizes = [
            vk::DescriptorPoolSize::builder().descriptor_count(self.swapchain_image_views.as_ref().unwrap().len() as u32).ty(vk::DescriptorType::UNIFORM_BUFFER).build(),
            vk::DescriptorPoolSize::builder().descriptor_count(self.swapchain_image_views.as_ref().unwrap().len() as u32).ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).build()
        ];

        let compute_pool_sizes = [
            vk::DescriptorPoolSize::builder().descriptor_count(self.swapchain_image_views.as_ref().unwrap().len() as u32).ty(vk::DescriptorType::STORAGE_IMAGE).build()
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::builder().pool_sizes(&pool_sizes).max_sets(self.swapchain_image_views.as_ref().unwrap().len() as u32);
        let compute_pool_info = vk::DescriptorPoolCreateInfo::builder().pool_sizes(&compute_pool_sizes).max_sets(self.swapchain_image_views.as_ref().unwrap().len() as u32);


        self.descriptor_pool = Some(unsafe{self.device.as_ref().unwrap().create_descriptor_pool(&pool_info, None)}.unwrap());
        self.compute_descriptor_pool = Some(unsafe{self.device.as_ref().unwrap().create_descriptor_pool(&compute_pool_info, None)}.unwrap());

    }

    fn create_descriptor_set_layout(&mut self) {

        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
        let sampler_layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(NUM_MODELS as u32)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);



        let layout_bindings = [
            ubo_layout_binding.build(),
            sampler_layout_binding.build(),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&layout_bindings);

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

    fn create_descriptor_sets(&mut self){
        let layouts = [self.descriptor_set_layout.unwrap()];
        let allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool.unwrap())
            .set_layouts(&layouts);

        self.descriptor_sets = Some(unsafe{self.device.as_ref().unwrap().allocate_descriptor_sets(&allocate_info)}.unwrap());

        println!("Num descriptor sets: {:?}", self.descriptor_sets.as_ref().unwrap().len());

        self.descriptor_sets.as_ref().unwrap().into_iter().for_each(|descriptor_set|{
            let buffer_infos = vec![vk::DescriptorBufferInfo::builder().buffer(self.uniform_buffers[0]).offset(0).range((std::mem::size_of::<UniformBufferObject>() ) as vk::DeviceSize).build()];
            let mut image_infos = vec![];
            for object in &self.objects{
                image_infos.push(vk::DescriptorImageInfo::builder()
                    .image_view(object.texture.as_ref().unwrap().image_view)
                    .sampler(object.texture.as_ref().unwrap().sampler)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .build());
            }
            image_infos.extend(self.cubemap.as_ref().unwrap().get_image_infos());
            image_infos.push(
                vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(self.compute_image_views[0])
                    .sampler(self.compute_samplers[0])
                    .build());
            let textures_left = NUM_MODELS - image_infos.len();
            for _ in 0..textures_left{
                image_infos.push(
                    vk::DescriptorImageInfo::builder()
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(self.compute_image_views[0])
                        .sampler(self.compute_samplers[0])
                        .build());
            }
            let descriptor_writes = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(*descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_infos)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(*descriptor_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_infos)
                    .build()

            ];

            unsafe{
                self.device.as_ref().unwrap().update_descriptor_sets(&descriptor_writes, &[]);
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

        let destination_pointer = unsafe {
            self.device.as_ref().unwrap().map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
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
                .destroy_buffer(staging_buffer, None)
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .free_memory(staging_buffer_memory, None)
        };
    }

    fn create_vertex_buffer(&mut self) {
        let buffer_size: vk::DeviceSize = (std::mem::size_of::<Vertex>() * self.vertices.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = self.create_buffer_with_memory(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let destination_pointer = unsafe {
            self.device.as_ref().unwrap().map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
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
            buffer_size*4,
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
                .destroy_buffer(staging_buffer, None)
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .free_memory(staging_buffer_memory, None)
        };
    }

    pub(crate) fn update_vertex_buffer(&mut self) {
        let buffer_size: vk::DeviceSize = (std::mem::size_of::<Vertex>() * self.vertices.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = self.create_buffer_with_memory(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let destination_pointer = unsafe {
            self.device.as_ref().unwrap().map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
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

        self.copy_buffer(staging_buffer, self.vertex_buffer.unwrap(), buffer_size);

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(staging_buffer, None)
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .free_memory(staging_buffer_memory, None)
        };
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
                &[vk::BufferCopy::builder().size(size).build()],
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
        let buffer_info = vk::BufferCreateInfo::builder()
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

        let allocate_info = vk::MemoryAllocateInfo::builder()
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

        self.run_commands();


        let image_index: u32;

        match unsafe {
            self.swapchain_loader.as_ref().unwrap().acquire_next_image(
                self.swapchain.unwrap(),
                u64::MAX,
                self.image_available_semaphore.unwrap(),
                vk::Fence::null(),
            )
        } {
            Ok((index, _)) => image_index = index,
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
        let submits = [vk::SubmitInfo::builder()
            .command_buffers(&command_buffers)
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(&signal_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .build()];

        unsafe {
            self.device.as_ref().unwrap().queue_submit(
                self.main_queue.unwrap(),
                &submits,
                self.in_flight_fence.unwrap(),
            )
        }
            .unwrap();

        let swapchains = [self.swapchain.unwrap()];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .wait_semaphores(&signal_semaphores);

        match unsafe {
            self.swapchain_loader
                .as_ref()
                .unwrap()
                .queue_present(self.main_queue.unwrap(), &present_info)
        } {
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

    fn create_swapchain(&mut self){
        let surface_formats = unsafe {
            self.surface_loader
                .as_ref()
                .unwrap()
                .get_physical_device_surface_formats(
                    self.physical_device.unwrap(),
                    self.surface.unwrap(),
                )
        }
            .unwrap();



        let default_surface_format = surface_formats[0];
        self.surface_format = Some(*
                                       surface_formats
                                           .iter()
                                           .find(|format| {
                                               return format.format == vk::Format::R8G8B8A8_SRGB
                                                   && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR;
                                           })
                                           .unwrap_or(&default_surface_format),
        );
        // surface_formats
        //     .into_iter()
        //     .for_each(|format| {
        //         println!("Format Properties: {:?}", unsafe{self.instance.as_ref().unwrap().get_physical_device_format_properties(self.physical_device.unwrap(),format.format)}.optimal_tiling_features);
        //     });




        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.surface.unwrap())
            .min_image_count(self.surface_capabilities.unwrap().min_image_count)
            .image_color_space(self.surface_format.unwrap().color_space)
            .image_format(self.surface_format.unwrap().format)
            .image_extent(self.surface_capabilities.unwrap().current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(self.surface_capabilities.unwrap().current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::IMMEDIATE)
            .clipped(true);

        self.swapchain_loader = Some(Swapchain::new(
            self.instance.as_ref().unwrap(),
            self.device.as_ref().unwrap(),
        ));
        self.swapchain = Some(
            unsafe {
                self.swapchain_loader
                    .as_ref()
                    .unwrap()
                    .create_swapchain(&swapchain_create_info, None)
            }
                .unwrap(),
        );

    }

    fn create_image_views(&mut self){
        self.swapchain_images = unsafe {
            self.swapchain_loader
                .as_ref()
                .unwrap()
                .get_swapchain_images(self.swapchain.unwrap())
        }
            .unwrap();

        self.swapchain_image_views = Some(
            self.swapchain_images
                .iter()
                .map(|image| {
                    return self.create_image_view(*image, self.surface_format.unwrap().format, vk::ImageAspectFlags::COLOR, 1)
                })
                .collect::<Vec<_>>(),
        );


    }

    fn load_shader(&self, path:std::path::PathBuf) -> vk::ShaderModule{
        let mut shader_file = std::fs::File::open(path).unwrap();
        let shader_code = ash::util::read_spv(&mut shader_file).unwrap();
        return VulkanData::create_shader_module(
            &self.device.as_ref().unwrap(),
            shader_code);
    }

    fn load_shaders(&mut self){
        let mut vert_shader_file = std::fs::File::open("shaders/vert.spv").unwrap();
        let mut frag_shader_file = std::fs::File::open("shaders/frag.spv").unwrap();
        let vert_shader_code = ash::util::read_spv(&mut vert_shader_file).unwrap();
        let frag_shader_code = ash::util::read_spv(&mut frag_shader_file).unwrap();
        self.vert_shader_module = Some(VulkanData::create_shader_module(
            &self.device.as_ref().unwrap(),
            vert_shader_code,
        ));
        self.frag_shader_module = Some(VulkanData::create_shader_module(
            &self.device.as_ref().unwrap(),
            frag_shader_code,
        ));
    }

    fn create_render_pass(&mut self){

        let push_constant_range = vk::PushConstantRange::builder()
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
        let push_constant_ranges = [push_constant_range.build()];

        let descriptor_set_layouts = [self.descriptor_set_layout.unwrap()];
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
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

        let color_attachment = vk::AttachmentDescription::builder()
            .format(self.surface_format.unwrap().format)
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_attachment_reference = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attchement = vk::AttachmentDescription::builder()
            .format(self.find_depth_format())
            .samples(self.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let depth_attachment_reference = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachment_resolve = vk::AttachmentDescription::builder()
            .format(self.surface_format.unwrap().format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(if self.msaa_samples != vk::SampleCountFlags::TYPE_1{
                vk::AttachmentLoadOp::DONT_CARE
            }else{
                vk::AttachmentLoadOp::CLEAR
            })
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_resolve_reference = vk::AttachmentReference::builder()
            .attachment(if self.msaa_samples != vk::SampleCountFlags::TYPE_1{
                2
            }else{
                0
            })
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);


        let color_attachment_references;
        let color_attachment_resolve_references;

        if self.msaa_samples != vk::SampleCountFlags::TYPE_1{
            color_attachment_resolve_references = vec![color_attachment_resolve_reference.build()];
            color_attachment_references = vec![color_attachment_reference.build()];

        }else{
            color_attachment_resolve_references = vec![];
            color_attachment_references = vec![color_attachment_resolve_reference.build()];

        };



        let mut main_subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_references)
            .depth_stencil_attachment(&depth_attachment_reference);
        if self.msaa_samples != vk::SampleCountFlags::TYPE_1{
            main_subpass = main_subpass.resolve_attachments(&color_attachment_resolve_references);
        }
        let dependencies = [
            vk::SubpassDependency::builder()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT| vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .build(),

        ];
        let attachments = if self.msaa_samples != vk::SampleCountFlags::TYPE_1{
            vec![color_attachment.build(), depth_attchement.build(), color_attachment_resolve.build(), ]
        }else{
            vec![color_attachment_resolve.build(), depth_attchement.build(), ]
        };

        let subpasses = [main_subpass.build()];

        let render_pass_info = vk::RenderPassCreateInfo::builder()
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

    fn create_compute_pipelines(&mut self){

        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);

        let sampler_layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(NUM_MODELS as u32)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);

        let layout_bindings = [
            ubo_layout_binding.build(),
            sampler_layout_binding.build(),
            vk::DescriptorSetLayoutBinding::builder().binding(2).descriptor_count(self.compute_images.len() as u32).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).stage_flags(vk::ShaderStageFlags::COMPUTE).build(),
            vk::DescriptorSetLayoutBinding::builder().binding(3).descriptor_count(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).stage_flags(vk::ShaderStageFlags::COMPUTE).build(),

        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&layout_bindings);

        self.compute_descriptor_set_layout = Some(unsafe {self.device
            .as_ref()
            .unwrap()
            .create_descriptor_set_layout(&layout_info, None)}.unwrap());


        let descriptor_set_layouts =[self.compute_descriptor_set_layout.unwrap()
        ];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.compute_descriptor_pool.unwrap())
            .set_layouts(&descriptor_set_layouts);

        self.compute_descriptor_sets = Some(unsafe{self.device.as_ref().unwrap().allocate_descriptor_sets(&descriptor_set_allocate_info)}.unwrap());

        for descriptor_set in self.compute_descriptor_sets.as_ref().unwrap(){
            println!("compute descriptor set construction started");
            let buffer_infos = vec![
                vk::DescriptorBufferInfo::builder().buffer(self.uniform_buffers[0]).offset(0).range((std::mem::size_of::<UniformBufferObject>()) as vk::DeviceSize).build(),
            ];

            let storage_buffer_infos = vec![
                vk::DescriptorBufferInfo::builder().buffer(self.storage_buffer.unwrap()).offset(0).range((std::mem::size_of::<Node>() * self.nodes.len()) as vk::DeviceSize).build()
            ];

            let mut image_infos = vec![];
            for object in &self.objects{
                image_infos.push(vk::DescriptorImageInfo::builder()
                    .image_view(object.texture.as_ref().unwrap().image_view)
                    .sampler(object.texture.as_ref().unwrap().sampler)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .build());
            }
            image_infos.extend(self.cubemap.as_ref().unwrap().get_image_infos());
            image_infos.push(
                vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(self.compute_image_views[0])
                    .sampler(self.compute_samplers[0])
                    .build());

            let mut storage_info: Vec<vk::DescriptorImageInfo> = vec![];
            for image_view in &self.compute_image_views{
                storage_info.push(vk::DescriptorImageInfo::builder()
                    .image_view(*image_view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .build());
            }

            let descriptor_writes = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(*descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_infos)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(*descriptor_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_infos)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(*descriptor_set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&storage_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(*descriptor_set)
                    .dst_binding(3)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&storage_buffer_infos)
                    .build()
            ];

            println!("Updating compute descriptor sets");

            unsafe{
                self.device.as_ref().unwrap().update_descriptor_sets(&descriptor_writes, &[]);
            }


        }

        let push_constant_range = vk::PushConstantRange::builder()
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let push_constant_ranges = [push_constant_range.build()];


        let compute_pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&push_constant_ranges)
            .set_layouts(&descriptor_set_layouts).build();

        println!("Creating compute pipeline layout");

        self.compute_pipeline_layout = Some(unsafe{
            self.device.as_ref().unwrap().create_pipeline_layout(&compute_pipeline_layout_info,None)
        }.unwrap());

        let main_c_string = CString::new("main").unwrap();

        let compute_pipeline_stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.load_shader(std::path::PathBuf::from("shaders/comp.spv")))
            .name(main_c_string.as_c_str());

        let compute_pipeline_infos = [vk::ComputePipelineCreateInfo::builder()
            .stage(compute_pipeline_stage_info.build())
            .layout(self.compute_pipeline_layout.unwrap()).build()];
        self.compute_pipelines = unsafe{self.device.as_ref().unwrap().create_compute_pipelines(vk::PipelineCache::null(), &compute_pipeline_infos,None)}.unwrap();
        println!("compute pipelines: {:?}", self.compute_pipelines.len());

        for i in 0..self.fullscreen_quads.len(){

            self.fullscreen_quads[i].texture.as_mut().unwrap().image = self.compute_images[0];
            self.fullscreen_quads[i].texture.as_mut().unwrap().image_view = self.compute_image_views[0];
            self.fullscreen_quads[i].texture.as_mut().unwrap().sampler = self.compute_samplers[0];
            self.fullscreen_quads[i].texture.as_mut().unwrap().allocation = self.compute_image_allocations[0];
            self.fullscreen_quads[i].texture.as_mut().unwrap().width = self.surface_capabilities.unwrap().current_extent.width;
            self.fullscreen_quads[i].texture.as_mut().unwrap().height = self.surface_capabilities.unwrap().current_extent.height;

        }
    }

    fn destroy_compute_pipelines(&mut self){
        unsafe{
            self.device.as_ref().unwrap().destroy_descriptor_set_layout(self.compute_descriptor_set_layout.unwrap(), None);
            self.device.as_ref().unwrap().destroy_descriptor_pool(self.compute_descriptor_pool.unwrap(), None);
            self.device.as_ref().unwrap().destroy_pipeline_layout(self.compute_pipeline_layout.unwrap(), None);
            for pipeline in &self.compute_pipelines{
                self.device.as_ref().unwrap().destroy_pipeline(*pipeline, None);

            }
        }
    }


    fn create_graphics_pipelines(&mut self){
        let binding_descriptions = vec![Vertex::get_binding_description()];
        let attribute_descriptions = Vertex::get_attribute_description();

        let vertex_input_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let dynamic_states = [vk::DynamicState::DEPTH_TEST_ENABLE_EXT];

        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_states);

        let viewport = vk::Viewport::builder()
            .x(0.0f32)
            .y(0.0f32)
            .width(self.surface_capabilities.unwrap().current_extent.width as f32)
            .height(self.surface_capabilities.unwrap().current_extent.height as f32)
            .min_depth(0.0f32)
            .max_depth(1.0f32);

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(self.surface_capabilities.unwrap().current_extent);

        let viewports = [viewport.build()];
        let scissors = [scissor.build()];
        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0f32);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(self.msaa_samples);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_blend_attachments = [color_blend_attachment.build()];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::OR)
            .attachments(&color_blend_attachments);

        let vert_entry_string = CString::new("main").unwrap();
        let frag_entry_string = CString::new("main").unwrap();

        let vert_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(self.vert_shader_module.unwrap())
            .name(vert_entry_string.as_c_str());
        let frag_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(self.frag_shader_module.unwrap())
            .name(frag_entry_string.as_c_str());
        let shader_stages = vec![
            vert_shader_stage_create_info.build(),
            frag_shader_stage_create_info.build(),
        ];

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false);

        let pipeline_infos = [
            vk::GraphicsPipelineCreateInfo::builder()
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
                .dynamic_state(&dynamic_state_info)
                .build(),
        ];

        self.graphics_pipelines =
            unsafe {
                self.device.as_ref().unwrap().create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &pipeline_infos,
                    None,
                )
            }
                .unwrap();
    }


    fn create_framebuffers(&mut self){
        self.swapchain_framebuffers = Some(
            self.swapchain_image_views
                .as_ref()
                .unwrap()
                .iter()
                .map(|image_view| {
                    let attachments = if self.msaa_samples != vk::SampleCountFlags::TYPE_1{
                        vec![self.color_image_view.unwrap(),self.depth_image_view.unwrap(), *image_view]
                    }else{
                        vec![*image_view,self.depth_image_view.unwrap()]
                    };
                    let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
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

    fn create_command_buffers(&mut self){

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
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

        println!("Num command buffers: {:?}", self.command_buffers.as_ref().unwrap().len());


    }

    fn run_commands(&self){
        let extended_dynamic_state = ash::extensions::ext::ExtendedDynamicState::new(self.instance.as_ref().unwrap(),self.device.as_ref().unwrap());

        self.command_buffers
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .for_each(|(i, command_buffer)| {
                let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder();
                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .begin_command_buffer(*command_buffer, &command_buffer_begin_info)
                }
                    .unwrap();

                //begin compute

                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_bind_pipeline(*command_buffer,vk::PipelineBindPoint::COMPUTE, self.compute_pipelines[0])};

                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_bind_descriptor_sets(*command_buffer,
                                                  vk::PipelineBindPoint::COMPUTE,
                                                  self.compute_pipeline_layout.unwrap(),
                                                  0,
                                                  &self.compute_descriptor_sets.as_ref().unwrap(),
                                                  &[])
                };

                for i in 0..self.compute_images.len(){
                    let barrier = vk::ImageMemoryBarrier::builder()
                        .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .src_access_mask(vk::AccessFlags::MEMORY_WRITE| vk::AccessFlags::MEMORY_READ)
                        .dst_access_mask(vk::AccessFlags::MEMORY_WRITE| vk::AccessFlags::MEMORY_READ)
                        .image(self.compute_images[i])
                        .subresource_range(vk::ImageSubresourceRange{
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1
                        });

                    unsafe{
                        self.device.as_ref().unwrap().cmd_pipeline_barrier(*command_buffer,
                                                                           vk::PipelineStageFlags::ALL_COMMANDS,
                                                                           vk::PipelineStageFlags::ALL_COMMANDS,
                                                                           vk::DependencyFlags::empty(),
                                                                           &[],
                                                                           &[],
                                                                           &[barrier.build()]);
                    }

                }


                let push_constant = PushConstants{
                    model: Matrix4::identity(),
                    view: self.objects[self.player_object_index].view,
                    proj: Matrix4::identity(),
                    texture_index: self.objects[0].texture.as_ref().unwrap().index,
                    constant: 1.0
                };
                unsafe {
                    let constants = std::slice::from_raw_parts(
                        (&push_constant as *const PushConstants) as *const u8,
                        std::mem::size_of::<PushConstants>(),
                    );
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_push_constants(*command_buffer,self.compute_pipeline_layout.unwrap(),vk::ShaderStageFlags::COMPUTE,0,constants)};


                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_dispatch(*command_buffer, (self.surface_capabilities.unwrap().current_extent.width as f64/8.0).ceil() as u32, (self.surface_capabilities.unwrap().current_extent.height as f64/8.0).ceil() as u32, 1)};
                for i in 0..self.compute_images.len() {
                    let barrier = vk::ImageMemoryBarrier::builder()
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ)
                        .dst_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ)
                        .image(self.compute_images[i])
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1
                        });

                    unsafe {
                        self.device.as_ref().unwrap().cmd_pipeline_barrier(*command_buffer,
                                                                           vk::PipelineStageFlags::ALL_COMMANDS,
                                                                           vk::PipelineStageFlags::ALL_COMMANDS,
                                                                           vk::DependencyFlags::empty(),
                                                                           &[],
                                                                           &[],
                                                                           &[barrier.build()]);
                    }
                }
                //end compute

                let clear_colors = vec![vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                }, vk::ClearValue{depth_stencil:vk::ClearDepthStencilValue{ depth: 1.0, stencil: 0 }}];

                let render_pass_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(self.render_pass.unwrap())
                    .framebuffer(self.swapchain_framebuffers.as_ref().unwrap()[i])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.surface_capabilities.unwrap().current_extent,
                    }).clear_values(&clear_colors)
                    ;

                unsafe {
                    self.device.as_ref().unwrap().cmd_begin_render_pass(
                        *command_buffer,
                        &render_pass_info,
                        vk::SubpassContents::INLINE,)
                };



                for pipeline in &self.graphics_pipelines{
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

                unsafe{ extended_dynamic_state.cmd_set_depth_test_enable(*command_buffer,false)};
                for object in &self.fullscreen_quads{
                    object.draw(self.device.as_ref().unwrap(),*command_buffer,self.pipeline_layout.unwrap(),1.0);
                }
                unsafe{ extended_dynamic_state.cmd_set_depth_test_enable(*command_buffer,true)};

                self.voxels.as_ref().unwrap().draw(self.device.as_ref().unwrap(), *command_buffer, self.pipeline_layout.unwrap(),0.0);


                for object in &self.objects{
                    object.draw(self.device.as_ref().unwrap(), *command_buffer, self.pipeline_layout.unwrap(),1.0);
                }

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
        self.create_color_resources();
        self.create_depth_resources();
        self.create_swapchain();
        self.create_image_views();
        self.create_render_pass();
        self.create_graphics_pipelines();

        self.create_framebuffers();
        self.create_uniform_buffers();
        self.create_descriptor_pool();
        self.create_compute_images();

        self.create_descriptor_sets();
        self.create_compute_pipelines();
        self.create_command_buffers();

    }



    fn create_color_resources(&mut self){
        let color_format = self.surface_format.unwrap().format;
        let (color_image, color_image_memory) = self.create_image_with_memory(
            self.surface_capabilities.unwrap().current_extent.width,
            self.surface_capabilities.unwrap().current_extent.height,
            1,
            self.msaa_samples,
            color_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        );
        self.color_image = Some(color_image);
        self.color_image_memory = Some(color_image_memory);
        self.color_image_view = Some(self.create_image_view(color_image,color_format,vk::ImageAspectFlags::COLOR,1));
    }

    fn create_cubemap_resources(&mut self){
        self.cubemap = Some(Cubemap::new(
            self,
            std::path::PathBuf::from("cubemap/px.png"),
            std::path::PathBuf::from("cubemap/nx.png"),
            std::path::PathBuf::from("cubemap/py.png"),
            std::path::PathBuf::from("cubemap/ny.png"),
            std::path::PathBuf::from("cubemap/pz.png"),
            std::path::PathBuf::from("cubemap/nz.png"),
        ));

    }

    fn create_command_pool(&mut self) {
        let command_pool_create_info =
            vk::CommandPoolCreateInfo::builder().flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER).queue_family_index(self.main_queue_index.unwrap());
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

    fn find_supported_format(&self, candidates: Vec<vk::Format>, tiling: vk::ImageTiling, features: vk::FormatFeatureFlags) -> vk::Format{
        return candidates.into_iter().find(|format| {
            let properties = unsafe {self.instance.as_ref().unwrap().get_physical_device_format_properties(self.physical_device.unwrap(),*format)};
            match tiling{
                vk::ImageTiling::LINEAR => {
                    return properties.linear_tiling_features.contains(features);
                }
                vk::ImageTiling::OPTIMAL => {
                    return properties.optimal_tiling_features.contains(features);
                }
                _ => panic!("No supported format or something idk I'm tired")
            }
        }
        ).unwrap();
    }

    fn find_depth_format(&self) -> vk::Format{
        return self.find_supported_format(vec![vk::Format::D32_SFLOAT, vk::Format::D24_UNORM_S8_UINT, vk::Format::D24_UNORM_S8_UINT], vk::ImageTiling::OPTIMAL, vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT);
    }


    fn create_shader_module(device: &Device, spv_code: Vec<u32>) -> vk::ShaderModule {
        let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(&spv_code);
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
                .for_each(|framebuffer| unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .destroy_framebuffer(*framebuffer, None)
                });
            self.device.as_ref().unwrap().destroy_image_view(self.depth_image_view.unwrap(), None);
            self.device.as_ref().unwrap().destroy_image(self.depth_image.unwrap(), None);
            self.device.as_ref().unwrap().free_memory(self.depth_image_memory.unwrap(), None);

            self.device.as_ref().unwrap().free_command_buffers(self.command_pool.unwrap(),self.command_buffers.as_ref().unwrap());

            for pipeline in &self.graphics_pipelines {
                self.device
                    .as_ref()
                    .unwrap()
                    .destroy_pipeline(*pipeline, None)
            };
        }

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_pipeline_layout(self.pipeline_layout.unwrap(), None)
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_render_pass(self.render_pass.unwrap(), None)
        };
        self.swapchain_image_views
            .as_ref()
            .unwrap()
            .iter()
            .for_each(|image_view| unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .destroy_image_view(*image_view, None)
            });
        unsafe {
            self.swapchain_loader
                .as_ref()
                .unwrap()
                .destroy_swapchain(self.swapchain.unwrap(), None)
        };
    }

    pub(crate) fn cleanup(&mut self) {
        unsafe { self.device.as_ref().unwrap().device_wait_idle() }.unwrap();

        self.cleanup_swapchain();
        unsafe {

            self.device
                .as_ref()
                .unwrap()
                .destroy_descriptor_set_layout(self.descriptor_set_layout.unwrap(), None);
            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(self.vertex_buffer.unwrap(), None);
            self.device
                .as_ref()
                .unwrap()
                .free_memory(self.vertex_buffer_memory.unwrap(), None);
            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(self.index_buffer.unwrap(), None);
            self.device
                .as_ref()
                .unwrap()
                .free_memory(self.index_buffer_memory.unwrap(), None);
        }

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_semaphore(self.render_finished_semaphore.unwrap(), None);
            self.device
                .as_ref()
                .unwrap()
                .destroy_semaphore(self.image_available_semaphore.unwrap(), None);
            self.device
                .as_ref()
                .unwrap()
                .destroy_fence(self.in_flight_fence.unwrap(), None);
        }
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_shader_module(self.vert_shader_module.unwrap(), None)
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_shader_module(self.frag_shader_module.unwrap(), None)
        };
        unsafe {
            self.surface_loader
                .as_ref()
                .unwrap()
                .destroy_surface(self.surface.unwrap(), None)
        };
        unsafe { self.device.as_ref().unwrap().destroy_device(None) };
        unsafe { self.instance.as_ref().unwrap().destroy_instance(None) };
    }
    pub(crate) fn process(&mut self, camera: &Camera) {

        let surface_width = self.surface_capabilities.unwrap().current_extent.width as f32;
        let surface_height = self.surface_capabilities.unwrap().current_extent.height as f32;
        let projection = cgmath::perspective(Deg(90.0), surface_width/surface_height, 0.1, 100.0);

        self.cubemap.as_mut().unwrap().process( camera,projection);

        for object in 0..self.objects.len(){
            self.objects[object].view = camera.get_view_matrix();
            self.objects[object].proj = projection;

        }
        for object in 0..self.fullscreen_quads.len(){
            self.objects[object].view = camera.get_view_matrix();
            self.objects[object].proj = projection;

        }

        self.voxels.as_mut().unwrap().process(camera.get_view_matrix(), projection);


        let random: [f32;NUM_RANDOM] = get_random_vector(&self.rng, NUM_RANDOM).try_into().unwrap();
        for i in 0..NUM_RANDOM{
            self.uniform_buffer_object.random[i] = Vector4::new(random[i],random[i],random[i],random[i],);
        }

        for i in 0..self.uniform_buffer_pointers.len(){
            unsafe{
                self.uniform_buffer_pointers[i].copy_from_nonoverlapping(&self.uniform_buffer_object as *const UniformBufferObject as *const u8, std::mem::size_of::<UniformBufferObject>());
            };
        }
        unsafe{
            self.storage_buffer_pointer.as_ref().unwrap().copy_from_nonoverlapping(self.nodes.as_ptr() as *const u8, self.nodes.len()*std::mem::size_of::<Node>());
        }


    }

    fn  create_uniform_buffers(&mut self) {
        let device_size = (std::mem::size_of::<UniformBufferObject>()) as vk::DeviceSize;

        let buffer_create_info = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER).size(device_size)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::CpuToGpu,
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
            ..Default::default()
        };

        let (buffer, allocation, _) = self.allocator.as_ref().unwrap().create_buffer(&buffer_create_info, &allocation_create_info).unwrap();
        self.uniform_buffers.push(buffer);
        self.uniform_buffer_allocations.push(allocation);
        self.uniform_buffer_pointers.push(self.allocator.as_ref().unwrap().map_memory(&allocation).unwrap());
    }

    fn create_storage_buffer(&mut self){
        let device_size = (std::mem::size_of::<Node>()*self.nodes.len()) as vk::DeviceSize;

        let buffer_create_info = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER).size(device_size)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_create_info = vk_mem::AllocationCreateInfo{
            usage: vk_mem::MemoryUsage::CpuToGpu,
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL| vk::MemoryPropertyFlags::HOST_VISIBLE,
            ..Default::default()
        };

        let (buffer, allocation, _) = self.allocator.as_ref().unwrap().create_buffer(&buffer_create_info,&allocation_create_info).unwrap();
        self.storage_buffer = Some(buffer);
        self.storage_buffer_allocation = Some(allocation);
        self.storage_buffer_pointer = Some(self.allocator.as_ref().unwrap().map_memory(&allocation).unwrap());

    }

    fn generate_mipmaps(&self, image: vk::Image, texture_width: u32, texture_height:u32, mip_levels: u32){
        let command_buffer = self.begin_single_time_commands();

        let mut barriers = [vk::ImageMemoryBarrier::builder()
            .image(image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange{
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            }).build()];


        let mut mip_width = texture_width as i32;
        let mut mip_height = texture_height as i32;

        let mut i = 1;
        while i < mip_levels && mip_height > 1 && mip_width > 1{
            barriers[0].subresource_range.base_mip_level = i - 1;
            barriers[0].old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barriers[0].new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barriers[0].src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barriers[0].dst_access_mask = vk::AccessFlags::TRANSFER_READ;


            unsafe {
                self.device.as_ref().unwrap().cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers
                )
            };

            let blit = vk::ImageBlit::builder()
                .src_offsets([vk::Offset3D{ x: 0,y: 0,z: 0 }, vk::Offset3D{ x: mip_width,y: mip_height,z: 1 }])
                .src_subresource(vk::ImageSubresourceLayers{
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i - 1,
                    base_array_layer: 0,
                    layer_count: 1
                })
                .dst_offsets([vk::Offset3D{x: 0, y: 0, z: 0}, vk::Offset3D{x: mip_width/2,y: mip_height/2, z: 1}])
                .dst_subresource(vk::ImageSubresourceLayers{
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: 1
                });
            unsafe{self.device.as_ref().unwrap().cmd_blit_image(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,&[blit.build()],vk::Filter::LINEAR)};

            barriers[0].old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barriers[0].new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barriers[0].src_access_mask = vk::AccessFlags::TRANSFER_READ;
            barriers[0].dst_access_mask = vk::AccessFlags::SHADER_READ;

            unsafe{self.device.as_ref().unwrap().cmd_pipeline_barrier(command_buffer,
                                                                      vk::PipelineStageFlags::TRANSFER,
                                                                      vk::PipelineStageFlags::FRAGMENT_SHADER,
                                                                      vk::DependencyFlags::empty(),
                                                                      &[],
                                                                      &[],
                                                                      &barriers)};


            mip_width /= 2;
            mip_height /= 2;
            i+= 1;
        }
        barriers[0].subresource_range.base_mip_level = mip_levels - 1;
        barriers[0].old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barriers[0].new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barriers[0].src_access_mask = vk::AccessFlags::TRANSFER_READ;
        barriers[0].dst_access_mask = vk::AccessFlags::SHADER_READ;

        unsafe{self.device.as_ref().unwrap().cmd_pipeline_barrier(command_buffer,
                                                                  vk::PipelineStageFlags::TRANSFER,
                                                                  vk::PipelineStageFlags::FRAGMENT_SHADER,
                                                                  vk::DependencyFlags::empty(),
                                                                  &[],
                                                                  &[],
                                                                  &barriers)};

        self.end_single_time_commands(command_buffer);


    }

    fn begin_single_time_commands(&self) -> vk::CommandBuffer{
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool.unwrap())
            .command_buffer_count(1);
        let command_buffer = unsafe{ self.device.as_ref().unwrap().allocate_command_buffers(&allocate_info)}.unwrap()[0];
        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe{self.device.as_ref().unwrap().begin_command_buffer(command_buffer, &begin_info)}.unwrap();
        return command_buffer;
    }

    fn end_single_time_commands(&self, command_buffer: vk::CommandBuffer){
        let command_buffers = [command_buffer];
        unsafe{self.device.as_ref().unwrap().end_command_buffer(command_buffer)}.unwrap();
        let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);
        unsafe{self.device.as_ref().unwrap().queue_submit(self.main_queue.unwrap(),&[submit_info.build()],vk::Fence::null())}.unwrap();
        unsafe{self.device.as_ref().unwrap().queue_wait_idle(self.main_queue.unwrap())}.unwrap();

        unsafe{self.device.as_ref().unwrap().free_command_buffers(self.command_pool.unwrap(), &command_buffers)};

    }

    fn transition_image_layout(&self, image: vk::Image, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, mip_levels: u32){
        let command_buffer = self.begin_single_time_commands();

        let mut barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange{
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1
            });

        let source_stage: vk::PipelineStageFlags;
        let destination_stage: vk::PipelineStageFlags;

        match (old_layout, new_layout){
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL)=>{

                barrier = barrier.src_access_mask(vk::AccessFlags::empty());
                barrier = barrier.dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
                destination_stage = vk::PipelineStageFlags::TRANSFER;
            }
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)=>{

                barrier = barrier.src_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                barrier = barrier.dst_access_mask(vk::AccessFlags::SHADER_READ);

                source_stage = vk::PipelineStageFlags::TRANSFER;
                destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
            }
            _ => panic!("Transition not supported")
        }

        unsafe{self.device.as_ref().unwrap().cmd_pipeline_barrier(
            command_buffer,
            source_stage,
            destination_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier.build()]
        )};



        self.end_single_time_commands(command_buffer);
    }

    fn create_image_view(&self, image: vk::Image, format: vk::Format, aspect_flags: vk::ImageAspectFlags, mip_levels: u32) -> vk::ImageView{
        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange{
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1
            });

        return unsafe{ self.device.as_ref().unwrap().create_image_view(&view_info,None)}.unwrap();

    }

    fn copy_buffer_to_image(&self, buffer: vk::Buffer, image: vk::Image, width: u32, height: u32){
        let command_buffer = self.begin_single_time_commands();

        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers{
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1
            })
            .image_offset(vk::Offset3D{
                x: 0,
                y: 0,
                z: 0
            })
            .image_extent(vk::Extent3D{
                width,
                height,
                depth: 1
            });

        unsafe{self.device.as_ref().unwrap().cmd_copy_buffer_to_image(command_buffer,buffer,image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region.build()])};

        self.end_single_time_commands(command_buffer);
    }

    fn create_image_with_memory(&self, width: u32, height:u32, mip_levels: u32, num_samples: vk::SampleCountFlags, format: vk::Format, tiling: vk::ImageTiling, usage:vk::ImageUsageFlags, properties: vk::MemoryPropertyFlags) -> (vk::Image, vk::DeviceMemory){

        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D{
                width,
                height,
                depth: 1
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .samples(num_samples)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe{self.device.as_ref().unwrap().create_image(&image_info,None)}.unwrap();

        let memory_requirements = unsafe{ self.device.as_ref().unwrap().get_image_memory_requirements(image)};

        let allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(memory_requirements.size)
            .memory_type_index(self.find_memory_type(memory_requirements.memory_type_bits,properties));

        let image_memory = unsafe{ self.device.as_ref().unwrap().allocate_memory(&allocate_info, None)}.unwrap();

        unsafe{self.device.as_ref().unwrap().bind_image_memory(image, image_memory,0)}.unwrap();
        return (image, image_memory);

    }
}
