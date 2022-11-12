use std::{ffi::CStr, sync::Arc};

use erupt::{SmallVec, vk, InstanceLoader, DeviceLoader, ExtendableFromConst, EntryLoader};
use nalgebra::{Vector3, Matrix4, Vector4, Vector2};
use winit::window::Window;

use crate::{support::{Light, UniformBufferObject, ShaderStorageBufferObject, NUM_RANDOM, NUM_MODELS, Vertex, PostProcessUniformBufferObject}, renderer::{combination_types::{TextureSet, CombinedSampledImage}, cpu_image::CpuImage}};

use super::{ui::UiData, drawables::{Cubemap, RenderObject}, buffers::MappedBuffer, combination_types::CombinedImage};

pub mod draw;
pub mod descriptors;
pub mod model_loading;
pub mod utilities;
pub mod main_pipeline;

const MSAA_ENABLED: bool = false;

//TODO: Clean this up, maybe split it into multiple structs so it's less of a mess
pub struct VulkanData {
    pub(crate) rng: rand::rngs::ThreadRng,
    pub(crate) instance: Option<InstanceLoader>,
    pub(crate) entry: Option<EntryLoader>,
    pub device: Option<DeviceLoader>,
    pub(crate) physical_device: Option<vk::PhysicalDevice>,
    pub(crate) allocator: Option<vk_mem_erupt::Allocator>,
    pub(crate) main_queue: Option<vk::Queue>,
    pub(crate) main_queue_index: Option<u32>,
    pub(crate) surface: Option<vk::SurfaceKHR>,
    pub(crate) surface_format: Option<vk::SurfaceFormatKHR>,
    pub surface_capabilities: Option<vk::SurfaceCapabilitiesKHR>,
    pub(crate) swapchain_created: bool,
    pub(crate) swapchain: Option<vk::SwapchainKHR>,
    pub(crate) image_index: u32,
    pub(crate) swapchain_images: SmallVec<vk::Image>,
    pub(crate) swapchain_image_views: Vec<vk::ImageView>,
    pub(crate) vert_shader_module: Option<vk::ShaderModule>,
    pub(crate) frag_shader_module: Option<vk::ShaderModule>,
    pub(crate) pipeline_layout: Option<vk::PipelineLayout>,
    pub(crate) postprocess_subpass_pipeline_layout: Option<vk::PipelineLayout>,
    pub(crate) render_pass: Option<vk::RenderPass>,
    pub(crate) graphics_pipelines: SmallVec<vk::Pipeline>,
    pub(crate) framebuffers: Vec<vk::Framebuffer>,
    pub(crate) command_pool: Option<vk::CommandPool>,
    pub(crate) command_buffer: Option<vk::CommandBuffer>,
    pub(crate) image_available_semaphore: Option<vk::Semaphore>,
    pub(crate) render_finished_semaphore: Option<vk::Semaphore>,
    pub(crate) in_flight_fence: Option<vk::Fence>,
    pub(crate) vertex_buffer: Option<vk::Buffer>,
    pub(crate) vertex_buffer_memory: Option<vk::DeviceMemory>,
    pub ui_data: UiData,
    pub(crate) index_buffer: Option<vk::Buffer>,
    pub(crate) index_buffer_memory: Option<vk::DeviceMemory>,
    pub(crate) mip_levels: u32,
    pub(crate) depth_image: Option<vk::Image>,
    pub(crate) depth_sampler: Option<vk::Sampler>,
    pub(crate) depth_image_memory: Option<vk::DeviceMemory>,
    pub(crate) depth_image_view: Option<vk::ImageView>,
    pub(crate) normals_image: Option<CombinedImage>,
    pub(crate) rough_metal_ao_image: Option<CombinedImage>,
    pub(crate) albedo_image: Option<vk::Image>,
    pub(crate) albedo_image_allocation: Option<vk_mem_erupt::Allocation>,
    pub(crate) albedo_image_view: Option<vk::ImageView>,
    pub(crate) color_resolve_image: Option<vk::Image>,
    pub(crate) color_resolve_image_memory: Option<vk::DeviceMemory>,
    pub(crate) color_resolve_image_view: Option<vk::ImageView>,
    pub(crate) cubemap: Option<Cubemap>,
    // fullscreen_quads: Vec<RenderObject>,
    pub(crate) vertices: Vec<Vertex>,
    pub(crate) indices: Vec<u32>,
    pub(crate) uniform_buffer_pointers: Vec<*mut u8>,
    pub(crate) uniform_buffers: Vec<vk::Buffer>,
    pub(crate) uniform_buffer_allocations: Vec<vk_mem_erupt::Allocation>,
    pub uniform_buffer_object: UniformBufferObject,
    pub post_process_ubo: Option<MappedBuffer<PostProcessUniformBufferObject>>,
    pub(crate) storage_buffer: Option<vk::Buffer>,
    pub(crate) storage_buffer_allocation: Option<vk_mem_erupt::Allocation>,
    pub storage_buffer_object: Box<ShaderStorageBufferObject>,
    pub(crate) current_boneset: usize,
    pub(crate) msaa_samples: vk::SampleCountFlagBits,
    pub(crate) descriptor_pool: Option<vk::DescriptorPool>,
    pub(crate) descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    pub(crate) postprocess_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    pub(crate) descriptor_sets: Option<SmallVec<vk::DescriptorSet>>,
    pub(crate) postprocess_descriptor_sets: Option<SmallVec<vk::DescriptorSet>>,
    pub(crate) last_frame_instant: std::time::Instant,
    pub objects: Vec<RenderObject>,
    pub(crate) textures: Vec<TextureSet>,
    pub(crate) cubemaps: Vec<CombinedSampledImage>,
    pub(crate) irradiance_maps: Vec<CombinedSampledImage>,
    pub(crate) environment_maps: Vec<CombinedSampledImage>,
    pub(crate) brdf_lut: Option<CombinedSampledImage>,
    pub(crate) fallback_texture: Option<TextureSet>,
    pub(crate) cpu_images: Vec<CpuImage>,
    pub(crate) images_3d: Vec<CombinedSampledImage>,
}



impl VulkanData{
    pub fn new() -> Self {
        assert!(!MSAA_ENABLED);
        let lights = [
            Light::new(Vector3::zeros(), Vector3::new(1.0, 1.0, 1.0)),
            Light::new(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(254.0 / 255.0, 196.0 / 255.0, 127.0 / 255.0),
            ),
        ];
        let uniform_buffer_object = UniformBufferObject {
            view: Matrix4::identity(),
            proj: Matrix4::identity(),
            random: [Vector4::new(0.0f32, 0.0f32, 0.0f32, 0.0f32); NUM_RANDOM],
            lights,
            cubemap_index: 0,
            num_lights: 100,
            map_mode: 0,
            exposure: 1.0,
            mouse_position: Vector2::new(0.0, 0.0),
            screen_size: Vector2::zeros(),
            time: 0.0,
            player_position: Vector3::zeros(),
        };

        let storage_buffer_object = ShaderStorageBufferObject::new_boxed();

        let indices = vec![];
        let vertices = vec![];

        return VulkanData {
            rng: rand::thread_rng(),
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
            swapchain_created: false,
            image_index: 0,
            swapchain: None,
            swapchain_images: SmallVec::new(),
            swapchain_image_views: vec![],
            vert_shader_module: None,
            frag_shader_module: None,
            pipeline_layout: None,
            postprocess_subpass_pipeline_layout: None,
            render_pass: None,
            graphics_pipelines: SmallVec::new(),
            framebuffers: vec![],
            command_pool: None,
            command_buffer: None,
            image_available_semaphore: None,
            render_finished_semaphore: None,
            in_flight_fence: None,
            vertex_buffer: None,
            vertex_buffer_memory: None,
            ui_data: UiData {
                vertex_buffer: None,
                vertex_allocation: None,
                vertex_allocation_info: None,
                vertex_pointer: None,
                index_buffer: None,
                index_allocation: None,
                index_allocation_info: None,
                index_pointer: None,
                num_indices: 0,
                pipeline: None,
                pipeline_layout: None,
                descriptor_set: None,
                descriptor_set_layout: None,
                image: None,
            },
            index_buffer: None,
            index_buffer_memory: None,
            mip_levels: 1,
            depth_image: None,
            depth_sampler: None,
            depth_image_memory: None,
            depth_image_view: None,
            albedo_image: None,
            albedo_image_view: None,
            albedo_image_allocation: None,
            normals_image: None,
            rough_metal_ao_image: None,
            color_resolve_image: None,
            color_resolve_image_memory: None,
            color_resolve_image_view: None,
            msaa_samples: vk::SampleCountFlagBits::_1,
            vertices,
            indices,
            descriptor_pool: None,
            descriptor_set_layout: None,
            postprocess_descriptor_set_layout: None,
            descriptor_sets: None,
            postprocess_descriptor_sets: None,
            uniform_buffer_object,
            storage_buffer: None,
            storage_buffer_allocation: None,
            storage_buffer_object,
            last_frame_instant: std::time::Instant::now(),
            cubemap: None,
            uniform_buffer_pointers: vec![],
            uniform_buffers: vec![],
            uniform_buffer_allocations: vec![],
            objects: vec![],
            // fullscreen_quads: vec![],
            textures: vec![],
            cubemaps: vec![],
            irradiance_maps: vec![],
            environment_maps: vec![],
            brdf_lut: None,
            fallback_texture: None,
            cpu_images: vec![],
            current_boneset: 0,
            // planet_textures: vec![],
            images_3d: vec![],
            post_process_ubo: None,
        };
    }
    pub fn init_vulkan<F>(&mut self, window: &Window, custom_resources: F)
    where
        F: FnOnce(&mut Self),
    {
        let mut validation_layer_names = vec![];

        #[cfg(feature = "validation-layers")]
        validation_layer_names.push(erupt::cstr!("VK_LAYER_KHRONOS_validation"));

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
                        CStr::from_ptr(extension_property.extension_name.as_ptr())
                            .to_string_lossy()
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

        if MSAA_ENABLED {
            self.msaa_samples = self.get_max_usable_sample_count();
        } else {
            self.msaa_samples = vk::SampleCountFlagBits::_1;
        }

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

        let device_features = vk::PhysicalDeviceFeaturesBuilder::new()
            .sampler_anisotropy(true)
            .wide_lines(true);

        let mut multiview_features =
            vk::PhysicalDeviceMultiviewFeaturesBuilder::new().multiview(true);
        // let mut multiview_properties = vk1_1::PhysicalDeviceMultiviewPropertiesBuilder::new().max_multiview_instance_index(6).max_multiview_view_count(6);

        let mut extended_dynamic_state_features =
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXTBuilder::new()
                .extended_dynamic_state(true);

        let mut line_rasterization_features =
            erupt::extensions::ext_line_rasterization::PhysicalDeviceLineRasterizationFeaturesEXTBuilder::new().smooth_lines(true);

        let device_extension_names_raw = vec![erupt::extensions::khr_swapchain::KHR_SWAPCHAIN_EXTENSION_NAME,
                                              erupt::extensions::ext_extended_dynamic_state::EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,
                                              erupt::extensions::ext_line_rasterization::EXT_LINE_RASTERIZATION_EXTENSION_NAME,
                                              ];
        let device_create_info = vk::DeviceCreateInfoBuilder::new()
            .extend_from(&mut extended_dynamic_state_features)
            .extend_from(&mut line_rasterization_features)
            .extend_from(&mut multiview_features)
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
        self.create_buffers();

        self.create_descriptor_set_layout();
        self.create_ui_descriptor_set_layout();
        self.create_render_pass();

        self.create_command_pool();
        self.transfer_data_to_storage_buffer(&self.storage_buffer_object);

        self.fallback_texture = Some(TextureSet {
            albedo: CombinedSampledImage::new(
                self,
                "models/fallback/albedo.png".parse().unwrap(),
                vk::ImageViewType::_2D,
                vk::Format::R8G8B8A8_SRGB,
                false,
            ),
            normal: CombinedSampledImage::new(
                self,
                "models/fallback/normal.png".parse().unwrap(),
                vk::ImageViewType::_2D,
                vk::Format::R8G8B8A8_UNORM,
                false,
            ),
            roughness_metalness_ao: CombinedSampledImage::new(
                self,
                "models/fallback/rough_metal_ao.png".parse().unwrap(),
                vk::ImageViewType::_2D,
                vk::Format::R8G8B8A8_UNORM,
                false,
            ),
        });

        for _ in 0..NUM_MODELS {
            self.cpu_images.push(CpuImage::new(self, 128, 128))
        }
        self.create_color_resources();
        self.create_normals_image();
        self.create_rough_metal_ao_image();
        self.create_depth_resources();
        self.create_cubemap_resources();
        self.create_array_image_resources();

        println!("Loading Shaders");
        self.load_shaders();
        self.create_graphics_pipelines();

        self.create_framebuffers();

        self.create_vertex_buffer();
        self.create_index_buffer();

        println!("Creating Descriptor pools");
        self.create_descriptor_pool();

        // println!("Running test compute shader");
        // self.run_test_shader();

        println!("Creating descriptor sets");
        self.create_descriptor_sets();

        custom_resources(self);
        self.update_descriptor_sets();
        println!("Creating UI pipelines and buffers");
        self.create_ui_data();
        self.update_ui_descriptors();
        self.create_ui_pipeline();

        println!("Creating command buffer");
        self.create_command_buffers();
        println!("Creating sync objects");
        self.create_sync_objects();

        println!("Finished init");
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
}