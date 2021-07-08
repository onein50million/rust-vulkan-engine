use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::vk::PhysicalDevice;
use ash::{vk, Device, Entry, Instance};
use cgmath::{Matrix4, Point3, SquareMatrix, Vector2, Vector3, Deg, Transform, InnerSpace, Quaternion, Euler, Angle, Zero, One, Matrix3};
use std::env;
use std::ffi::{c_void, CStr, CString};
use std::ops::{Mul, Add};
use winit::event::{DeviceEvent, Event, VirtualKeyCode, WindowEvent, ElementState};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};
use image::{GenericImageView, Rgb, Rgba};
use image::buffer::Pixels;
use std::any::Any;

const FRAMERATE_TARGET: f64 = 280.0;
const MSAA_SAMPLES: vk::SampleCountFlags = vk::SampleCountFlags::TYPE_1;


//TODO: see what happens if you take this off
#[derive(Copy, Clone)]
#[repr(C)]
struct Vertex {
    position: Vector3<f32>,
    color: Vector3<f32>,
    texture_coordinae: Vector2<f32>,
}

#[repr(C)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}

struct Player {
    position: Vector3<f64>,
    yaw: f64,
    pitch: f64,
    move_speed: f64,
}

impl Player{
    fn get_view_matrix(&self) -> Matrix4<f32>{
        let matrix = Matrix4::<f32>::from_translation(Vector3::new(self.position.x as f32, self.position.y as f32, self.position.z as f32))
                * Matrix4::<f32>::from_angle_y(Deg(-self.yaw as f32))
                * Matrix4::<f32>::from_angle_x(Deg(self.pitch as f32));

        // println!("matrix: {:?}", matrix);

        // let vector_distance  = self.pitch.to_radians().cos();

        // return Matrix4::look_to_rh(
        //     Point3::new(self.position.x as f32, self.position.y as f32, self.position.z as f32,),
        //     Vector3::new((vector_distance * self.yaw.to_radians().sin()) as f32,(vector_distance * self.yaw.to_radians().cos()) as f32, self.pitch.to_radians().sin() as f32),
        //     Vector3::new(0.0,0.0,1.0));

        return matrix.inverse_transform().unwrap();
    }

    fn process_inputs(&mut self, inputs: &Inputs, delta_time: f64){
        let movement = (Matrix3::from_angle_y(Deg(-self.yaw))) * Vector3::new(
            inputs.right - inputs.left,
            inputs.down - inputs.up,
            inputs.backward - inputs.forward,
            );
        self.position += movement*self.move_speed * delta_time;
        // println!("position: {:?}", self.position)
    }

}


impl Vertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        return vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build();
    }
    fn get_attribute_description() -> Vec<vk::VertexInputAttributeDescription> {
        let attribute_descriptions = vec![
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(12), //might be off, could be fun to see what happens when it's off
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(24), //might be off, could be fun to see what happens when it's off

        ];

        return attribute_descriptions
            .into_iter()
            .map(|attribute_description| attribute_description.build())
            .collect();
    }
}

struct Inputs{
    forward: f64,
    backward: f64,
    left: f64,
    right: f64,
    up: f64,
    down: f64
}

struct VulkanData {
    instance: Option<Instance>,
    entry: Option<Entry>,
    device: Option<Device>,
    physical_device: Option<PhysicalDevice>,
    main_queue: Option<vk::Queue>,
    main_queue_index: Option<u32>,
    surface_loader: Option<Surface>,
    surface: Option<vk::SurfaceKHR>,
    surface_format: Option<vk::SurfaceFormatKHR>,
    surface_capabilities: Option<vk::SurfaceCapabilitiesKHR>,
    swapchain_loader: Option<Swapchain>,
    swapchain: Option<vk::SwapchainKHR>,
    swapchain_image_views: Option<Vec<vk::ImageView>>,
    vert_shader_module: Option<vk::ShaderModule>,
    frag_shader_module: Option<vk::ShaderModule>,
    pipeline_layout: Option<vk::PipelineLayout>,
    render_pass: Option<vk::RenderPass>,
    graphics_pipeline: Option<vk::Pipeline>,
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
    texture_image: Option<vk::Image>,
    texture_image_memory: Option<vk::DeviceMemory>,
    texture_image_view: Option<vk::ImageView>,
    texture_sampler: Option<vk::Sampler>,
    depth_image: Option<vk::Image>,
    depth_image_memory: Option<vk::DeviceMemory>,
    depth_image_view: Option<vk::ImageView>,
    vertices: [Vertex; 8],
    indices: Vec<u16>,
    descriptor_pool: Option<vk::DescriptorPool>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_sets: Option<Vec<vk::DescriptorSet>>,
    uniform_buffer_object: UniformBufferObject,
    last_frame_instant: std::time::Instant,
    mouse_buffer: Vector2<f64>,
    inputs: Inputs,
    player: Player,
    focused: bool,
}

impl VulkanData {
    fn new() -> Self {
        let vertices: [Vertex; 8] = [
            Vertex {
                position: Vector3 {
                    x: -0.5,
                    y: -0.5,
                    z: 0.0,
                },
                color: Vector3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                texture_coordinae: Vector2 { x: 1.0, y: 0.0 }
            },
            Vertex {
                position: Vector3 {
                    x: 0.5,
                    y: -0.5,
                    z: 0.0,
                },
                color: Vector3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                texture_coordinae: Vector2 { x: 0.0, y: 0.0 }
            },
            Vertex {
                position: Vector3 {
                    x: 0.5,
                    y: 0.5,
                    z: 0.0,
                },
                color: Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                texture_coordinae: Vector2 { x: 0.0, y: 1.0 }
            },
            Vertex {
                position: Vector3 {
                    x: -0.5,
                    y: 0.5,
                    z: 0.0,
                },
                color: Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                texture_coordinae: Vector2 { x: 1.0, y: 1.0 }
            },
            //second square
            Vertex {
                position: Vector3 {
                    x: -0.5,
                    y: -0.5,
                    z: 0.5,
                },
                color: Vector3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                texture_coordinae: Vector2 { x: 1.0, y: 0.0 }
            },
            Vertex {
                position: Vector3 {
                    x: 0.5,
                    y: -0.5,
                    z: 0.5,
                },
                color: Vector3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                texture_coordinae: Vector2 { x: 0.0, y: 0.0 }
            },
            Vertex {
                position: Vector3 {
                    x: 0.5,
                    y: 0.5,
                    z: 0.5,
                },
                color: Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                texture_coordinae: Vector2 { x: 0.0, y: 1.0 }
            },
            Vertex {
                position: Vector3 {
                    x: -0.5,
                    y: 0.5,
                    z: 0.5,
                },
                color: Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                texture_coordinae: Vector2 { x: 1.0, y: 1.0 }
            },
        ];

        let uniform_buffer_object = UniformBufferObject {
            model: Matrix4::identity(),
            view: Matrix4::look_at_rh(
                Point3 {
                    x: 3.0,
                    y: 3.0,
                    z: -3.0,
                },
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
            ),
            proj: cgmath::perspective(cgmath::Deg(1.0), 1.0, 0.1, 10.0),
        };

        return VulkanData {
            instance: None,
            entry: None,
            device: None,
            physical_device: None,
            main_queue: None,
            main_queue_index: None,
            surface_loader: None,
            surface: None,
            surface_format: None,
            surface_capabilities: None,
            swapchain_loader: None,
            swapchain: None,
            swapchain_image_views: None,
            vert_shader_module: None,
            frag_shader_module: None,
            pipeline_layout: None,
            render_pass: None,
            graphics_pipeline: None,
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
            texture_image: None,
            texture_image_memory: None,
            texture_image_view: None,
            texture_sampler: None,
            depth_image: None,
            depth_image_memory: None,
            depth_image_view: None,
            vertices,
            indices: vec![0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4],
            descriptor_pool: None,
            descriptor_set_layout: None,
            descriptor_sets: None,
            uniform_buffer_object,
            last_frame_instant: std::time::Instant::now(),
            mouse_buffer: Vector2::new(0.0, 0.0),
            inputs: Inputs {
                forward: 0.0,
                backward: 0.0,
                left: 0.0,
                right: 0.0,
                up: 0.0,
                down: 0.0
            },
            player: Player{position: Vector3::new(2.0, 0.0, 1.0), yaw: 0.0, pitch: 0.0, move_speed: 5.0 },
            focused: true
        };
    }
    fn init_vulkan(&mut self, window: &Window) {
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

        let device_extension_names_raw = vec![Swapchain::name().as_ptr()];
        let device_create_info = vk::DeviceCreateInfo::builder()
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


        self.get_surface_support();
        self.get_surface_capabilities();

        self.create_command_pool();


        self.create_texture_sampler();
        self.create_texture_image();
        self.create_texture_image_view();

        self.create_descriptor_set_layout();


        self.create_vertex_buffer();
        self.create_index_buffer();

        self.create_depth_resources();

        self.recreate_swaphain();

        self.create_descriptor_pool();
        self.create_descriptor_sets();


        self.process();



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

    fn create_depth_resources(&mut self){

        let depth_format = self.find_depth_format();

        let (depth_image, depth_image_memory) = self.create_image(
            self.surface_capabilities.unwrap().current_extent.width,
            self.surface_capabilities.unwrap().current_extent.height,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        );

        self.depth_image = Some(depth_image);
        self.depth_image_memory = Some(depth_image_memory);
        self.depth_image_view = Some(self.create_image_view(depth_image,vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH));


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
        let pool_sizes = [vk::DescriptorPoolSize::builder().descriptor_count(self.swapchain_image_views.as_ref().unwrap().len() as u32).ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).build()];

        let pool_info = vk::DescriptorPoolCreateInfo::builder().pool_sizes(&pool_sizes).max_sets(self.swapchain_image_views.as_ref().unwrap().len() as u32);

        self.descriptor_pool = Some(unsafe{self.device.as_ref().unwrap().create_descriptor_pool(&pool_info, None)}.unwrap());
    }

    fn create_descriptor_set_layout(&mut self) {

        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .immutable_samplers(&[]);

        let sampler_layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let layout_bindings = [ubo_layout_binding.build(), sampler_layout_binding.build()];

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

        self.descriptor_sets.as_ref().unwrap().into_iter().for_each(|descriptor_set|{

            let image_infos = [vk::DescriptorImageInfo::builder().image_view(self.texture_image_view.unwrap()).sampler(self.texture_sampler.unwrap()).build()];
            let descriptor_writes = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(*descriptor_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_infos)
                    .build()
            ];

            unsafe{self.device.as_ref().unwrap().update_descriptor_sets(&descriptor_writes, &[])};
        });

    }

    fn create_index_buffer(&mut self) {
        let buffer_size: vk::DeviceSize = std::mem::size_of_val(&self.indices) as u64;
        let (staging_buffer, staging_buffer_memory) = self.create_buffer(
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

        let (index_buffer, index_buffer_memory) = self.create_buffer(
            buffer_size,
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
        let buffer_size: vk::DeviceSize = std::mem::size_of_val(&self.vertices) as u64;
        let (staging_buffer, staging_buffer_memory) = self.create_buffer(
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

        let (vertex_buffer, vertex_buffer_memory) = self.create_buffer(
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

    fn create_buffer(
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

    fn draw_frame(&mut self) {
        let fences = [self.in_flight_fence.unwrap()];
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .wait_for_fences(&fences, true, u64::MAX)
        }
        .unwrap();
        unsafe { self.device.as_ref().unwrap().reset_fences(&fences) }.unwrap();

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
                    self.recreate_swaphain();
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
                    self.recreate_swaphain();
                }
            }
        }
    }

    fn recreate_swaphain(&mut self) {
        self.create_command_pool();

        self.get_surface_capabilities();

        self.create_depth_resources();

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

        surface_formats
            .into_iter()
            .for_each(|format| {
                println!("Format Properties: {:?}", unsafe{self.instance.as_ref().unwrap().get_physical_device_format_properties(self.physical_device.unwrap(),format.format)}.optimal_tiling_features);
            });




        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.surface.unwrap())
            .min_image_count(self.surface_capabilities.unwrap().min_image_count)
            .image_color_space(self.surface_format.unwrap().color_space)
            .image_format(self.surface_format.unwrap().format)
            .image_extent(self.surface_capabilities.unwrap().current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
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

        let swapchain_images = unsafe {
            self.swapchain_loader
                .as_ref()
                .unwrap()
                .get_swapchain_images(self.swapchain.unwrap())
        }
        .unwrap();

        self.swapchain_image_views = Some(
            swapchain_images
                .into_iter()
                .map(|image| {
                    return self.create_image_view(image, swapchain_create_info.image_format, vk::ImageAspectFlags::COLOR)
                })
                .collect::<Vec<_>>(),
        );

        let mut vert_shader_file = std::fs::File::open("vert.spv").unwrap();
        let mut frag_shader_file = std::fs::File::open("frag.spv").unwrap();
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

        let binding_descriptions = [Vertex::get_binding_description()];
        let attribute_descriptions = Vertex::get_attribute_description();

        let vertex_input_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport::builder()
            .x(0.0f32)
            .y(0.0f32)
            .width(swapchain_create_info.image_extent.width as f32)
            .height(swapchain_create_info.image_extent.height as f32)
            .min_depth(0.0f32)
            .max_depth(1.0f32);

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(swapchain_create_info.image_extent);

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
            .rasterization_samples(MSAA_SAMPLES);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false);

        let color_blend_attachments = [color_blend_attachment.build()];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&color_blend_attachments);

        let push_constant_range = vk::PushConstantRange::builder()
            .offset(0)
            .size(std::mem::size_of::<UniformBufferObject>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let descriptor_set_layouts = [self.descriptor_set_layout.unwrap()];
        let push_constant_ranges = [push_constant_range.build()];
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
            .format(swapchain_create_info.image_format)
            .samples(MSAA_SAMPLES)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_reference = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_attachment_references = [color_attachment_reference.build()];

        let depth_attchement = vk::AttachmentDescription::builder()
            .format(self.find_depth_format())
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let depth_attachment_reference = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_references)
            .depth_stencil_attachment(&depth_attachment_reference);

        let dependencies = [vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT| vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .build()];

        let attachments = [color_attachment.build(), depth_attchement.build()];

        let subpasses = [subpass.build()];

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .subpasses(&subpasses)
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

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
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
            .depth_stencil_state(&depth_stencil);

        self.graphics_pipeline = Some(
            unsafe {
                self.device.as_ref().unwrap().create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info.build()],
                    None,
                )
            }
            .unwrap()[0],
        );

        self.swapchain_framebuffers = Some(
            self.swapchain_image_views
                .as_ref()
                .unwrap()
                .iter()
                .map(|image_view| {
                    let attachments = vec![*image_view, self.depth_image_view.unwrap()];
                    let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                        .render_pass(self.render_pass.unwrap())
                        .attachments(&attachments)
                        .width(swapchain_create_info.image_extent.width)
                        .height(swapchain_create_info.image_extent.height)
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

    fn create_color_resources(&mut self){
        // self.surface_capabilities.unwrap().current_extent
    }

    fn create_command_pool(&mut self) {
        let command_pool_create_info =
            vk::CommandPoolCreateInfo::builder().queue_family_index(self.main_queue_index.unwrap());
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

    fn has_stencil_format(format: vk::Format) -> bool{
        return format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT;
    }

    fn create_texture_sampler(&mut self){
        let physical_device_properties = unsafe{self.instance.as_ref().unwrap().get_physical_device_properties(self.physical_device.unwrap())};

        println!("Max Anisotropy: {:}", physical_device_properties.limits.max_sampler_anisotropy);

        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(physical_device_properties.limits.max_sampler_anisotropy)
            .border_color(vk::BorderColor::INT_OPAQUE_WHITE)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);

        self.texture_sampler = Some(unsafe{self.device.as_ref().unwrap().create_sampler(&sampler_info, None)}.unwrap());

    }

    fn create_shader_module(device: &Device, spv_code: Vec<u32>) -> vk::ShaderModule {
        let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(&spv_code);
        unsafe { device.create_shader_module(&shader_module_create_info, None) }.unwrap()
    }

    fn cleanup_swapchain(&mut self) {


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
        unsafe {
            self.device.as_ref().unwrap().destroy_image_view(self.depth_image_view.unwrap(), None);
            self.device.as_ref().unwrap().destroy_image(self.depth_image.unwrap(), None);
            self.device.as_ref().unwrap().free_memory(self.depth_image_memory.unwrap(), None);

            self.device
                .as_ref()
                .unwrap()
                .destroy_pipeline(self.graphics_pipeline.unwrap(), None)
        };
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
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_command_pool(self.command_pool.unwrap(), None)
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

    fn cleanup(&mut self) {
        unsafe { self.device.as_ref().unwrap().device_wait_idle() }.unwrap();

        self.cleanup_swapchain();
        unsafe {


            self.device.as_ref().unwrap().destroy_sampler(self.texture_sampler.unwrap(), None);

            self.device
                .as_ref()
                .unwrap()
                .destroy_image_view(self.texture_image_view.unwrap(), None);

            self.device
                .as_ref()
                .unwrap()
                .destroy_image(self.texture_image.unwrap(), None);

            self.device
                .as_ref()
                .unwrap()
                .free_memory(self.texture_image_memory.unwrap(), None);


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
    fn process(&mut self) {
        let delta_time = self.last_frame_instant.elapsed().as_secs_f64();
        self.last_frame_instant = std::time::Instant::now();
        // println!("fps: {}", 1.0/delta_time);

        let surface_width = self.surface_capabilities.unwrap().current_extent.width as f32;
        let surface_height = self.surface_capabilities.unwrap().current_extent.height as f32;

        // self.player.rotation.x += delta_time as f32;
        self.player.process_inputs(&self.inputs,delta_time);

        self.player.yaw += self.mouse_buffer.x*0.1;
        self.player.pitch = (self.player.pitch +  self.mouse_buffer.y*0.1).clamp(-80.0,80.0);

        self.mouse_buffer = Vector2::zero();

        // println!("pitch: {:?} yaw: {:?}", self.player.pitch, self.player.yaw);

        self.uniform_buffer_object.proj =
            cgmath::perspective(Deg(90.0), surface_width / surface_height, 0.1, 100.0);
        // self.uniform_buffer_object.model = self
        //     .uniform_buffer_object
        //     .model
        //     .mul(Matrix4::from_angle_x(cgmath::Deg(100.0 * delta_time as f32)));

        self.uniform_buffer_object.view = self.player.get_view_matrix() ;
        // self.mouse_buffer = Vector2{ x: 0.0, y: 0.0 };

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

                let clear_colors = vec![vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                }, vk::ClearValue{depth_stencil:vk::ClearDepthStencilValue{ depth: 1.0, stencil: 0 }}];

                unsafe {
                    self.device.as_ref().unwrap().cmd_begin_render_pass(
                        *command_buffer,
                        &vk::RenderPassBeginInfo::builder() //not sure how I feel about stacking all this together
                            .render_pass(self.render_pass.unwrap())
                            .framebuffer(self.swapchain_framebuffers.as_ref().unwrap()[i])
                            .render_area(vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: self.surface_capabilities.unwrap().current_extent,
                            })
                            .clear_values(&clear_colors),
                        vk::SubpassContents::INLINE,
                    )
                };

                unsafe {
                    self.device.as_ref().unwrap().cmd_bind_pipeline(
                        *command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.graphics_pipeline.unwrap(),
                    )
                };

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
                        vk::IndexType::UINT16,
                    )
                };

                unsafe {
                    let constants = std::slice::from_raw_parts(
                        (&self.uniform_buffer_object as *const UniformBufferObject) as *const u8,
                        std::mem::size_of::<UniformBufferObject>(),
                    );
                    self.device.as_ref().unwrap().cmd_push_constants(
                        *command_buffer,
                        self.pipeline_layout.unwrap(),
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        constants,
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
                    self.device.as_ref().unwrap().cmd_draw_indexed(
                        *command_buffer,
                        self.indices.len() as u32,
                        1,
                        0,
                        0,
                        0,
                    )
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
    fn create_texture_image(&mut self) {
        let reader = image::io::Reader::open("texture.png").unwrap();
        let reader_decode =  reader.decode();
        let dynamic_image = reader_decode.unwrap();
        let texture_image = dynamic_image.as_rgba8().unwrap();
        let pixels = texture_image.as_raw();
        let image_size: vk::DeviceSize  = (texture_image.width() * texture_image.height() * 4) as u64;

        println!("Image size: {:}", image_size);

        let (staging_buffer, staging_buffer_memory) = self.create_buffer(image_size,vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);

        let destination_pointer = unsafe {self.device.as_ref().unwrap().map_memory(staging_buffer_memory, 0, image_size,vk::MemoryMapFlags::empty())}.unwrap();
        // unsafe {std::ptr::write_bytes(destination_pointer, 100, image_size as usize)}
        // unsafe {std::ptr::copy_nonoverlapping(pixels, destination_pointer as *mut Vec<u8>, pixels.len())};
        unsafe{destination_pointer.copy_from_nonoverlapping(pixels.as_ptr() as *const c_void,image_size as usize)}

        unsafe{self.device.as_ref().unwrap().unmap_memory(staging_buffer_memory)}

        let (texture, texture_memory) = self.create_image(texture_image.width(), texture_image.height(), vk::Format::R8G8B8A8_SRGB, vk::ImageTiling::OPTIMAL, vk::ImageUsageFlags::TRANSFER_DST|vk::ImageUsageFlags::SAMPLED, vk::MemoryPropertyFlags::DEVICE_LOCAL);
        self.texture_image = Some(texture);
        self.texture_image_memory = Some(texture_memory);

        self.transition_image_layout(self.texture_image.unwrap(), vk::Format::R8G8B8A8_SRGB, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        self.copy_buffer_to_image(staging_buffer,self.texture_image.unwrap(), texture_image.width(), texture_image.height());
        self.transition_image_layout(self.texture_image.unwrap(), vk::Format::R8G8B8A8_SRGB, vk::ImageLayout::TRANSFER_DST_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        unsafe{self.device.as_ref().unwrap().destroy_buffer(staging_buffer, None)};
        unsafe{self.device.as_ref().unwrap().free_memory(staging_buffer_memory, None)};
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

    fn transition_image_layout(&self, image: vk::Image, format: vk::Format, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout){
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
                level_count: 1,
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

    fn create_texture_image_view(&mut self){
        let view_info = vk::ImageViewCreateInfo::builder()
            .image(self.texture_image.unwrap())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .subresource_range(vk::ImageSubresourceRange{
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1
            });

        self.texture_image_view = Some(unsafe{ self.device.as_ref().unwrap().create_image_view(&view_info,None)}.unwrap());
    }

    fn create_image_view(&self, image: vk::Image, format: vk::Format, aspect_flags: vk::ImageAspectFlags) -> vk::ImageView{
        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange{
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: 1,
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

    fn create_image(&self, width: u32, height:u32, format: vk::Format, tiling: vk::ImageTiling, usage:vk::ImageUsageFlags, properties: vk::MemoryPropertyFlags) -> (vk::Image, vk::DeviceMemory){


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
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .samples(vk::SampleCountFlags::TYPE_1)
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



fn main() {
    let mut frametime: std::time::Instant = std::time::Instant::now();
    let mut time_since_last_frame: f64 = 0.0; //seconds

    let path = env::current_dir().unwrap();
    println!("The current directory is {}", path.display());

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .build(&event_loop)
        .expect("Failed to build window");
    window.set_cursor_grab(true).unwrap();
    window.set_cursor_visible(false);

    let mut vulkan_data = VulkanData::new();
    vulkan_data.init_vulkan(&window);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event,
                ..
            } => {
                match event {
                    WindowEvent::Focused(is_focused) => vulkan_data.focused = is_focused,
                    WindowEvent::CloseRequested => {
                        unsafe { vulkan_data.device.as_ref().unwrap().device_wait_idle() }.unwrap();
                        close_app(&mut vulkan_data, control_flow);
                    }
                    _ => {}
                }

            }
            Event::MainEventsCleared => {
                //app update code
                // window.request_redraw();
                let elapsed_time = frametime.elapsed().as_secs_f64();
                frametime = std::time::Instant::now();

                if time_since_last_frame > 1.0 / FRAMERATE_TARGET {
                    vulkan_data.draw_frame();
                    vulkan_data.process();
                    time_since_last_frame -= 1.0 / FRAMERATE_TARGET;
                }
                time_since_last_frame += elapsed_time;
            }
            Event::DeviceEvent {
                device_id: _,
                event: device_event,
            } => match device_event {
                DeviceEvent::MouseMotion { delta } => {
                    vulkan_data.mouse_buffer.x += delta.0;
                    vulkan_data.mouse_buffer.y += delta.1;
                }
                DeviceEvent::Key(key) => {
                    match key.virtual_keycode {
                        Some(keycode) => {
                            if keycode == VirtualKeyCode::Escape && key.state == ElementState::Released && vulkan_data.focused{
                                close_app(&mut vulkan_data, control_flow);
                            }
                            if keycode == VirtualKeyCode::W{
                                vulkan_data.inputs.forward = match key.state{
                                    ElementState::Released => 0.0,
                                    ElementState::Pressed => 1.0
                                };
                            }if keycode == VirtualKeyCode::S{
                                vulkan_data.inputs.backward = match key.state{
                                    ElementState::Released => 0.0,
                                    ElementState::Pressed => 1.0
                                };
                            }if keycode == VirtualKeyCode::A{
                                vulkan_data.inputs.left = match key.state{
                                    ElementState::Released => 0.0,
                                    ElementState::Pressed => 1.0
                                };
                            }if keycode == VirtualKeyCode::D{
                                vulkan_data.inputs.right = match key.state{
                                    ElementState::Released => 0.0,
                                    ElementState::Pressed => 1.0
                                };
                            }if keycode == VirtualKeyCode::Space{
                                vulkan_data.inputs.up = match key.state{
                                    ElementState::Released => 0.0,
                                    ElementState::Pressed => 1.0
                                };
                            }if keycode == VirtualKeyCode::LControl{
                                vulkan_data.inputs.down = match key.state{
                                    ElementState::Released => 0.0,
                                    ElementState::Pressed => 1.0
                                };
                            }

                        }
                        _ => {}
                    }
                },
                _ => {}
            },
            _ => {}
        }
    })
}

fn close_app(vulkan_data: &mut VulkanData, control_flow: &mut ControlFlow) {
    *control_flow = ControlFlow::Exit;
    vulkan_data.cleanup();
}
