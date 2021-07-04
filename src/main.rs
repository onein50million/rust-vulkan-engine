use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::vk::PhysicalDevice;
use ash::{vk, Device, Entry, Instance};
use cgmath::Vector3;
use std::env;
use std::ffi::{CStr, CString};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};
use winit::window::CursorIcon::VerticalText;

//TODO: see what happens if you take this off
#[repr(C)]
struct Vertex {
    position: Vector3<f32>,
    color: Vector3<f32>,
}

const vertices: [Vertex; 3] = [
    Vertex {
        position: Vector3 {
            x: -0.5,
            y: 0.0,
            z: 0.0,
        },
        color: Vector3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
    },
    Vertex {
        position: Vector3 {
            x: 0.5,
            y: 0.0,
            z: 0.0,
        },
        color: Vector3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
    },
    Vertex {
        position: Vector3 {
            x: 0.0,
            y: 0.5,
            z: 0.0,
        },
        color: Vector3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
    },
];

impl Vertex{
    fn get_binding_description() -> vk::VertexInputBindingDescription{
        return unsafe{vk::VertexInputBindingDescription::builder().binding(0).stride(std::mem::size_of::<Vertex>() as u32).input_rate(vk::VertexInputRate::VERTEX).build() };
    }
    fn get_attribute_description() -> Vec<vk::VertexInputAttributeDescription>{
        let attribute_descriptions = vec![
            vk::VertexInputAttributeDescription::builder().binding(0).location(0).format(vk::Format::R32G32B32_SFLOAT).offset(0),
            vk::VertexInputAttributeDescription::builder().binding(0).location(1).format(vk::Format::R32G32B32_SFLOAT).offset(12), //might be off, could be fun to see what happens when it's off
        ];

        return attribute_descriptions.into_iter().map(|attribute_description| attribute_description.build()).collect();
    }
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
    allocator: Option<vk_mem::Allocator>,
    staging_buffer: Option<vk::Buffer>,
    vertex_buffer: Option<vk::Buffer>,

}

impl VulkanData {
    fn new() -> Self {
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
            allocator: None,
            staging_buffer: None,
            vertex_buffer: None
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

        self.entry = Some(unsafe { ash::EntryCustom::new() }.unwrap());

        let app_info = vk::ApplicationInfo::builder();
        let mut surface_extensions = ash_window::enumerate_required_extensions(window).unwrap();
        surface_extensions.push(DebugUtils::name());

        let raw_extensions = surface_extensions
            .iter()
            .map(|extension| extension.to_owned().as_ptr())
            .collect::<Vec<_>>();

        let create_info = vk::InstanceCreateInfo::builder()
            .enabled_extension_names(&raw_extensions)
            .application_info(&app_info)
            .enabled_layer_names(&validation_layer_names_raw);
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

        let device_extension_names_raw = vec![Swapchain::name().as_ptr()];
        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(queue_create_infos)
            .enabled_layer_names(&validation_layer_names_raw)
            .enabled_extension_names(&device_extension_names_raw);
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

        self.recreate_swaphain();

        let _surface_support = unsafe {
            self.surface_loader
                .as_ref()
                .unwrap()
                .get_physical_device_surface_support(
                    self.physical_device.unwrap(),
                    self.main_queue_index.unwrap(),
                    self.surface.unwrap(),
                )
        };

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

        let allocator_create_info = vk_mem::AllocatorCreateInfo{ physical_device: self.physical_device.unwrap(),
            device:self.device.unwrap(),
            instance: self.instance.unwrap(),
            ..Default::default()};
        self.allocator = Some(vk_mem::Allocator::new(&allocator_create_info).unwrap());
        let allocation_create_info = vk_mem::AllocationCreateInfo{usage: vk_mem::MemoryUsage::GpuOnly,..Default::default()};
        let (vertex_buffer, vertex_buffer_allocation, _) =
            allocator.create_buffer(
                vk::BufferCreateInfo::builder().size(16*1024).usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST).build(), &allocation_create_info
            ).unwrap();
        let (staging_buffer, staging_buffer_allocation, _) =
            allocator.create_buffer(
                vk::BufferCreateInfo::builder().size(16*1024).usage(vk::BufferUsageFlags:: | vk::BufferUsageFlags::TRANSFER_SRC).build(), &allocation_create_info
            ).unwrap();
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
        let surface_capabilities = unsafe {
            self.surface_loader
                .as_ref()
                .unwrap()
                .get_physical_device_surface_capabilities(
                    self.physical_device.unwrap(),
                    self.surface.unwrap(),
                )
        }
        .unwrap();
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
        self.surface_format = Some(
            *surface_formats
                .iter()
                .find(|format| {
                    return format.format == vk::Format::B8G8R8A8_SRGB
                        && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR;
                })
                .unwrap(),
        );

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.surface.unwrap())
            .min_image_count(surface_capabilities.min_image_count)
            .image_color_space(self.surface_format.unwrap().color_space)
            .image_format(self.surface_format.unwrap().format)
            .image_extent(surface_capabilities.current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_capabilities.current_transform)
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
                .iter()
                .map(|image| {
                    let image_view_create_info = vk::ImageViewCreateInfo::builder()
                        .format(swapchain_create_info.image_format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(*image)
                        .view_type(vk::ImageViewType::TYPE_2D);
                    return unsafe {
                        self.device
                            .as_ref()
                            .unwrap()
                            .create_image_view(&image_view_create_info, None)
                    }
                    .unwrap();
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
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0f32);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachement = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false);

        let color_blend_attachements = [color_blend_attachement.build()];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&color_blend_attachements);

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder();

        self.pipeline_layout = Some(
            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_pipeline_layout(&pipeline_layout_create_info, None)
            }
            .unwrap(),
        );

        let color_attachement = vk::AttachmentDescription::builder()
            .format(swapchain_create_info.image_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachement_reference = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_attachement_references = [color_attachement_reference.build()];
        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachement_references);

        let color_attachements = [color_attachement.build()];
        let subpasses = [subpass.build()];

        let dependencies = [vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .build()];

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&color_attachements)
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
            .subpass(0);

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
                    let attachements = vec![*image_view];
                    let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                        .render_pass(self.render_pass.unwrap())
                        .attachments(&attachements)
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
                }];

                unsafe {
                    self.device.as_ref().unwrap().cmd_begin_render_pass(
                        *command_buffer,
                        &vk::RenderPassBeginInfo::builder() //not sure how I feel about stacking all this together
                            .render_pass(self.render_pass.unwrap())
                            .framebuffer(self.swapchain_framebuffers.as_ref().unwrap()[i])
                            .render_area(vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: swapchain_create_info.image_extent,
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
                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .cmd_draw(*command_buffer, 3, 1, 0, 0)
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
        VulkanData::cleanup_swapchain(self);

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
}

fn main() {
    let path = env::current_dir().unwrap();
    println!("The current directory is {}", path.display());

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .build(&event_loop)
        .expect("Failed to build window");
    let mut vulkan_data = VulkanData::new();
    vulkan_data.init_vulkan(&window);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                unsafe { vulkan_data.device.as_ref().unwrap().device_wait_idle() }.unwrap();
                *control_flow = ControlFlow::Exit;
                vulkan_data.cleanup();
            }
            Event::MainEventsCleared => {
                //app update code
                window.request_redraw();
                vulkan_data.draw_frame();
            }
            _ => {}
        }
    })
}
