use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{WindowBuilder, Window};
use winit::event::{Event, WindowEvent};
use ash::{vk, Instance, Device};
use ash::version::{EntryV1_0, InstanceV1_0, DeviceV1_0};
use std::ffi::{CString, CStr};
use ash::vk::{PhysicalDevice, Framebuffer};
use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain};
use std::env;

struct VulkanData{
    instance: Option<Instance>,
    device: Option<Device>,
    main_queue: vk::Queue,
    surface_loader: Option<Surface>,
    surface: vk::SurfaceKHR,
    surface_capabilities: vk::SurfaceCapabilitiesKHR,
    surface_format: vk::SurfaceFormatKHR,
    swapchain_loader: Option<Swapchain>,
    swapchain: vk::SwapchainKHR,
    swapchain_image_views: Vec<vk::ImageView>,
    vert_shader_module: vk::ShaderModule,
    frag_shader_module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    graphics_pipeline: vk::Pipeline,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
}



impl VulkanData{
    fn new(window: &Window) -> Self{

        let mut vulkan_data: VulkanData = VulkanData{
            instance: None,
            device: None,
            main_queue: Default::default(),
            surface_capabilities: Default::default(),
            surface_format: Default::default(),
            surface_loader: None,
            surface: Default::default(),
            swapchain_loader: None,
            swapchain: Default::default(),
            swapchain_image_views: vec![],
            vert_shader_module: Default::default(),
            frag_shader_module: Default::default(),
            pipeline_layout: Default::default(),
            render_pass: Default::default(),
            graphics_pipeline: Default::default(),
            swapchain_framebuffers: vec![],
            command_pool: Default::default(),
            command_buffers: vec![],
            image_available_semaphore: Default::default(),
            render_finished_semaphore: Default::default(),
            in_flight_fence: Default::default(),
        };

        let validation_layer_names = [CString::new("VK_LAYER_KHRONOS_validation").expect("CString conversion failed")];
        let validation_layer_names_raw= validation_layer_names.iter().map(|c_string|c_string.as_ptr()).collect::<Vec<_>>();
        // validation_layers.remove(0);
        // println!("{:}", validation_layers[0]);

        let entry = unsafe { ash::EntryCustom::new()}.unwrap();

        let app_info = vk::ApplicationInfo::builder();
        let mut surface_extensions = ash_window::enumerate_required_extensions(window).unwrap();
        surface_extensions.push(DebugUtils::name());

        let raw_extensions = surface_extensions.iter().map(|extension| extension.to_owned().as_ptr()).collect::<Vec<_>>();

        let create_info = vk::InstanceCreateInfo::builder()
            .enabled_extension_names(&raw_extensions)
            .application_info(&app_info)
            .enabled_layer_names(&validation_layer_names_raw);
        vulkan_data.instance = Some(unsafe { entry.create_instance(&create_info, None) }.expect("Failed to create instance"));

        let instance = vulkan_data.instance.as_ref().unwrap();

        let physical_devices = unsafe {instance.enumerate_physical_devices()}.unwrap();
        let mut physical_device: PhysicalDevice = Default::default();



        if physical_devices.len() == 1{
            physical_device = physical_devices[0];
        }else{
            physical_devices.iter().find(|device|{

                //TODO: implement multiple gpu finding

                let properties = unsafe {instance.get_physical_device_properties(**device)};
                let device_name_cstring = unsafe {CStr::from_ptr(properties.device_name.as_ptr())}.to_owned();
                let device_name = device_name_cstring.to_str().unwrap();

                println!("Device Name: {}\nDriver: {}", device_name, &properties.driver_version);
                return true;
            });
        }
        let main_queue_index: u32 = unsafe { instance.get_physical_device_queue_family_properties(physical_device) }.iter()
            .position(|queue| queue.queue_flags
                .contains(vk::QueueFlags::GRAPHICS)).unwrap() as u32;

        let queue_create_info = vk::DeviceQueueCreateInfo::builder().queue_family_index(main_queue_index).queue_priorities(&[1.0f32]);
        let queue_create_infos = &[queue_create_info.build()];

        let device_extension_names_raw = vec![Swapchain::name().as_ptr()];
        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(queue_create_infos)
            .enabled_layer_names(&validation_layer_names_raw)
            .enabled_extension_names(&device_extension_names_raw);
        vulkan_data.device = Some(unsafe {instance.create_device(physical_device,&device_create_info,None)}.unwrap());
        let device = vulkan_data.device.as_ref().unwrap();

        vulkan_data.main_queue = unsafe {device.get_device_queue(main_queue_index,0)};

        vulkan_data.surface_loader = Some( Surface::new(&entry,instance));
        let surface_loader = vulkan_data.surface_loader.as_ref().unwrap();
        vulkan_data.surface = unsafe {ash_window::create_surface(&entry,instance,window,None)}.unwrap();
        let surface = &vulkan_data.surface;
        vulkan_data.surface_capabilities = unsafe {surface_loader.get_physical_device_surface_capabilities(physical_device,*surface)}.unwrap();
        let surface_formats = unsafe { surface_loader.get_physical_device_surface_formats(physical_device,*surface)}.unwrap();
        let surface_format = surface_formats.iter().find(|format| {
            return format.format == vk::Format::B8G8R8A8_SRGB && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR;
        }).unwrap();

        let _surface_support = unsafe {surface_loader.get_physical_device_surface_support(physical_device, main_queue_index, *surface)};

        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(main_queue_index);
        vulkan_data.command_pool = unsafe {device.create_command_pool(&command_pool_create_info, None)}.unwrap();

        VulkanData::recreate_swaphain(vulkan_data);
        let semaphore_create_info = vk::SemaphoreCreateInfo::builder();
        vulkan_data.image_available_semaphore = unsafe{device.create_semaphore(&semaphore_create_info,None)}.unwrap();
        vulkan_data.render_finished_semaphore = unsafe{device.create_semaphore(&semaphore_create_info,None)}.unwrap();
        let fence_create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        vulkan_data.in_flight_fence = unsafe{ device.create_fence(&fence_create_info, None)}.unwrap();

        return vulkan_data;

    }

    fn draw_frame(&self){
        let device = self.device.as_ref().unwrap();


        let fences = [self.in_flight_fence];
        unsafe { device.wait_for_fences(&fences,true,u64::MAX)}.unwrap();
        unsafe {device.reset_fences(&fences)}.unwrap();

        let (image_index,is_swapchain_suboptimal) =
            unsafe {self.swapchain_loader.as_ref().unwrap().acquire_next_image(self.swapchain,u64::MAX, self.image_available_semaphore, vk::Fence::null())}.unwrap();

        let wait_semaphores = [self.image_available_semaphore];
        let signal_semaphores = [self.render_finished_semaphore];
        let wait_stages = vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.command_buffers[image_index as usize]];
        let submits = [
            vk::SubmitInfo::builder().command_buffers(&command_buffers)
                .wait_semaphores(&wait_semaphores)
                .signal_semaphores(&signal_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .build()
        ];

        unsafe{device.queue_submit(self.main_queue,&submits, self.in_flight_fence)}.unwrap();

        let swapchains = [self.swapchain];
        let image_indices  = [image_index];
        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .wait_semaphores(&signal_semaphores);

        unsafe {self.swapchain_loader.as_ref().unwrap().queue_present(self.main_queue, &present_info)}.unwrap();
    }

    fn recreate_swaphain(vulkan_data: VulkanData){
        let device = vulkan_data.device.as_ref().unwrap();

        unsafe {device.device_wait_idle()}.unwrap();
        vulkan_data.cleanup_swapchain();


        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(vulkan_data.surface)
            .min_image_count(vulkan_data.surface_capabilities.min_image_count)
            .image_color_space(vulkan_data.surface_format.color_space)
            .image_format(vulkan_data.surface_format.format)
            .image_extent(vulkan_data.surface_capabilities.current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vulkan_data.surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::IMMEDIATE)
            .clipped(true);
        let swapchain_loader = Swapchain::new(vulkan_data.instance.as_ref().unwrap(), vulkan_data.device.as_ref().unwrap());
        let swapchain = unsafe {swapchain_loader.create_swapchain(&swapchain_create_info, None)}.unwrap();

        let swapchain_images = unsafe {swapchain_loader.get_swapchain_images(swapchain)}.unwrap();

        let swapchain_image_views = swapchain_images.iter().map(|image|{
            let image_view_create_info = vk::ImageViewCreateInfo::builder()
                .format(swapchain_create_info.image_format)
                .components(vk::ComponentMapping{
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
                .subresource_range(vk::ImageSubresourceRange{
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D);
            return unsafe{device.create_image_view(&image_view_create_info, None)}.unwrap();
        }).collect::<Vec<_>>();

        let mut vert_shader_file = std::fs::File::open("vert.spv").unwrap();
        let mut frag_shader_file = std::fs::File::open("frag.spv").unwrap();
        let vert_shader_code = ash::util::read_spv(&mut vert_shader_file).unwrap();
        let frag_shader_code = ash::util::read_spv(&mut frag_shader_file).unwrap();

        let vert_shader_module = VulkanData::create_shader_module(&device, vert_shader_code);
        let frag_shader_module = VulkanData::create_shader_module(&device, frag_shader_code);

        let vert_entry_string = CString::new("main").unwrap();
        let frag_entry_string = CString::new("main").unwrap();

        let vert_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(vert_entry_string.as_c_str());
        let frag_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(frag_entry_string.as_c_str());
        let shader_stages = vec![vert_shader_stage_create_info.build(),frag_shader_stage_create_info.build()];
        let vertex_input_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&[])
            .vertex_attribute_descriptions(&[]);

        let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport::builder()
            .x(0.0f32).y(0.0f32)
            .width(swapchain_create_info.image_extent.width as f32)
            .height(swapchain_create_info.image_extent.height as f32)
            .min_depth(0.0f32).max_depth(1.0f32);

        let scissor = vk::Rect2D::builder().offset(vk::Offset2D{ x: 0, y: 0 }).extent(swapchain_create_info.image_extent);

        let viewports = [viewport.build()];
        let scissors = [scissor.build()];
        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder().viewports(&viewports).scissors(&scissors);

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

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None)}.unwrap();

        let color_attachement = vk::AttachmentDescription::builder()
            .format(swapchain_create_info.image_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachement_reference = vk::AttachmentReference::builder().attachment(0).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
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
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE).build()];

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&color_attachements)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = unsafe {device.create_render_pass(&render_pass_info, None)}.unwrap();

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_create_info)
            .input_assembly_state(&input_assembly_create_info)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let graphics_pipeline = unsafe{device.create_graphics_pipelines(vk::PipelineCache::null(),&[pipeline_info.build()], None)}.unwrap()[0];

        let swapchain_framebuffers:Vec<Framebuffer> = swapchain_image_views.iter().map(|image_view|{
            let attachements = vec![*image_view];
            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachements)
                .width(swapchain_create_info.image_extent.width)
                .height(swapchain_create_info.image_extent.height)
                .layers(1);
            return unsafe {device.create_framebuffer(&framebuffer_create_info, None)}.unwrap();
        }).collect();


        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(vulkan_data.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(swapchain_framebuffers.len() as u32);

        let command_buffers = unsafe {device.allocate_command_buffers(&command_buffer_allocate_info)}.unwrap();

        command_buffers.iter().enumerate().for_each(|(i, command_buffer)|{
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder();
            unsafe{device.begin_command_buffer(*command_buffer, &command_buffer_begin_info)}.unwrap();

            let clear_colors = vec![vk::ClearValue{color:vk::ClearColorValue{float32: [0.0, 0.0, 0.0, 0.0]}}];

            unsafe{device.cmd_begin_render_pass(*command_buffer, &vk::RenderPassBeginInfo::builder() //not sure how I feel about stacking all this together
                .render_pass(render_pass)
                .framebuffer(swapchain_framebuffers[i])
                .render_area(vk::Rect2D{offset: vk::Offset2D{ x: 0, y: 0 }, extent: swapchain_create_info.image_extent})
                .clear_values(&clear_colors), vk::SubpassContents::INLINE)};

            unsafe{ device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::GRAPHICS, graphics_pipeline) };
            unsafe{ device.cmd_draw(*command_buffer, 3, 1, 0, 0)};
            unsafe{ device.cmd_end_render_pass(*command_buffer)};
            unsafe{ device.end_command_buffer(*command_buffer)}.unwrap();


        });

        vulkan_data.swapchain_loader = Some(swapchain_loader);
        vulkan_data.swapchain = swapchain;
        vulkan_data.swapchain_image_views = swapchain_image_views;
        vulkan_data.vert_shader_module = vert_shader_module;
        vulkan_data.frag_shader_module = frag_shader_module;
        vulkan_data.pipeline_layout = pipeline_layout;
        vulkan_data.render_pass = render_pass;
        vulkan_data.graphics_pipeline = graphics_pipeline;
        vulkan_data.swapchain_framebuffers = swapchain_framebuffers;
        vulkan_data.command_buffers = command_buffers;
    }

    fn create_shader_module(device: &Device, spv_code: Vec<u32>) -> vk::ShaderModule{
        let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(&spv_code);
        unsafe {device.create_shader_module(&shader_module_create_info, None)}.unwrap()
    }

    fn cleanup_swapchain(&self){
        self.swapchain_framebuffers.iter().for_each(|framebuffer| unsafe{self.device.as_ref().unwrap().destroy_framebuffer(*framebuffer, None)});
        unsafe {self.device.as_ref().unwrap().destroy_pipeline(self.graphics_pipeline,None)};
        unsafe {self.device.as_ref().unwrap().destroy_pipeline_layout(self.pipeline_layout,None)};
        unsafe {self.device.as_ref().unwrap().destroy_render_pass(self.render_pass, None)};
        unsafe {self.device.as_ref().unwrap().destroy_command_pool(self.command_pool, None)};
        self.swapchain_image_views.iter().for_each(|image_view| unsafe {self.device.as_ref().unwrap().destroy_image_view(*image_view, None)});
        unsafe {self.swapchain_loader.as_ref().unwrap().destroy_swapchain(self.swapchain, None)};

    }

    fn cleanup(&self){
        self.cleanup_swapchain();

        unsafe {
            self.device.as_ref().unwrap().destroy_semaphore(self.render_finished_semaphore, None);
            self.device.as_ref().unwrap().destroy_semaphore(self.image_available_semaphore, None);
            self.device.as_ref().unwrap().destroy_fence(self.in_flight_fence, None);
        }
        unsafe {self.device.as_ref().unwrap().destroy_shader_module(self.vert_shader_module,None)};
        unsafe {self.device.as_ref().unwrap().destroy_shader_module(self.frag_shader_module, None)};
        unsafe {self.surface_loader.as_ref().unwrap().destroy_surface(self.surface,None)};
        unsafe {self.device.as_ref().unwrap().destroy_device(None)};
        unsafe {self.instance.as_ref().unwrap().destroy_instance(None)};
    }
}

fn main(){
    let path = env::current_dir().unwrap();
    println!("The current directory is {}", path.display());

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).expect("Failed to build window");
    let vulkan_data = VulkanData::new(&window);

    event_loop.run(move |event, _, control_flow|{
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                unsafe{ vulkan_data.device.as_ref().unwrap().device_wait_idle()}.unwrap();
                *control_flow = ControlFlow::Exit;
                vulkan_data.cleanup();
            }
            Event::MainEventsCleared => {
                //app update code
                window.request_redraw();
                vulkan_data.draw_frame();
            },
            _ => {}
        }
    })
}