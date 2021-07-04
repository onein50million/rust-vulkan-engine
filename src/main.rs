use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{WindowBuilder, Window};
use winit::event::{Event, WindowEvent};
use ash::{vk, Instance, Device, Entry};
use ash::version::{EntryV1_0, InstanceV1_0, DeviceV1_0};
use std::ffi::{CString, CStr};
use ash::vk::{PhysicalDevice, Framebuffer, SwapchainKHR, ShaderModule, SurfaceFormatKHR, SurfaceKHR, Pipeline, PipelineLayout, RenderPass, ImageView, CommandBuffer, CommandPool};
use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain};
use std::env;

struct VulkanData{
    instance: Instance,
    entry: Entry,
    device: Device,
    physical_device: PhysicalDevice,
    main_queue: vk::Queue,
    main_queue_index: u32,
    surface_loader: Surface,
    surface: vk::SurfaceKHR,
    surface_format: vk::SurfaceFormatKHR,
    swapchain_loader: Swapchain,
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
        let instance = unsafe { entry.create_instance(&create_info, None) }.expect("Failed to create instance") ;

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
        let device = unsafe {instance.create_device(physical_device,&device_create_info,None)}.unwrap();

        let main_queue = unsafe {device.get_device_queue(main_queue_index,0)};

        let surface = unsafe {ash_window::create_surface(&entry,&instance,window,None)}.unwrap();
        let surface_loader = Surface::new(&entry,&instance);

        let (surface_format,
            swapchain_loader,
            swapchain,
            swapchain_image_views,
            vert_shader_module,
            frag_shader_module,
            pipeline_layout,
            render_pass,
            graphics_pipeline,
            swapchain_framebuffers,
            command_pool,
            command_buffers) = VulkanData::recreate_swaphain(&device, &instance, &surface, &surface_loader, &physical_device, main_queue_index);

        let _surface_support = unsafe {surface_loader.get_physical_device_surface_support(physical_device, main_queue_index, surface)};

        let semaphore_create_info = vk::SemaphoreCreateInfo::builder();
        let image_available_semaphore = unsafe{device.create_semaphore(&semaphore_create_info,None)}.unwrap();
        let render_finished_semaphore = unsafe{device.create_semaphore(&semaphore_create_info,None)}.unwrap();
        let fence_create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight_fence = unsafe{ device.create_fence(&fence_create_info, None)}.unwrap();

        return VulkanData{
            instance,
            entry,
            device,
            physical_device,
            main_queue,
            main_queue_index,
            surface_loader,
            surface,
            surface_format,
            swapchain_loader,
            swapchain,
            swapchain_image_views,
            vert_shader_module,
            frag_shader_module,
            pipeline_layout,
            render_pass,
            graphics_pipeline,
            swapchain_framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
        }
    }

    fn draw_frame(&mut self){

        let fences = [self.in_flight_fence];
        unsafe { self.device.wait_for_fences(&fences,true,u64::MAX)}.unwrap();
        unsafe {self.device.reset_fences(&fences)}.unwrap();

        let mut image_index: u32 = 0;

        match unsafe {self.swapchain_loader.acquire_next_image(self.swapchain,u64::MAX, self.image_available_semaphore, vk::Fence::null())}{
            Ok((index,_)) => {image_index = index}
            Err(e) => {if e == vk::Result::ERROR_OUT_OF_DATE_KHR || e == vk::Result::SUBOPTIMAL_KHR{
                unsafe {self.device.device_wait_idle()}.unwrap();
                self.cleanup_swapchain();
                let (surface_format, swapchain_loader, swapchain, image_views, vert_shader, frag_shader, pipeline_layout,render_pass,  graphics_pipeline, framebuffers, command_pool, command_buffers ) = VulkanData::recreate_swaphain(&self.device, &self.instance, &self.surface, &self.surface_loader, &self.physical_device, self.main_queue_index);
                self.surface_format = surface_format;
                self.swapchain_loader = swapchain_loader;
                self.swapchain = swapchain;
                self.swapchain_image_views = image_views;
                self.vert_shader_module = vert_shader;
                self.frag_shader_module = frag_shader;
                self.pipeline_layout = pipeline_layout;
                self.render_pass = render_pass;
                self.graphics_pipeline = graphics_pipeline;
                self.swapchain_framebuffers = framebuffers;
                self.command_pool = command_pool;
                self.command_buffers = command_buffers;                return;
            }else{
                panic!("acquire_next_image error");
            }}
        };


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

        unsafe{self.device.queue_submit(self.main_queue,&submits, self.in_flight_fence)}.unwrap();

        let swapchains = [self.swapchain];
        let image_indices  = [image_index];
        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .wait_semaphores(&signal_semaphores);

        match unsafe {self.swapchain_loader.queue_present(self.main_queue, &present_info)}{
            Ok(_) => {}
            Err(e) => {if e == vk::Result::ERROR_OUT_OF_DATE_KHR || e == vk::Result::SUBOPTIMAL_KHR{
                unsafe {self.device.device_wait_idle()}.unwrap();
                self.cleanup_swapchain();
                let (surface_format, swapchain_loader, swapchain, image_views, vert_shader, frag_shader, pipeline_layout,render_pass,  graphics_pipeline, framebuffers, command_pool, command_buffers ) = VulkanData::recreate_swaphain(&self.device, &self.instance, &self.surface, &self.surface_loader, &self.physical_device, self.main_queue_index);
                self.surface_format = surface_format;
                self.swapchain_loader = swapchain_loader;
                self.swapchain = swapchain;
                self.swapchain_image_views = image_views;
                self.vert_shader_module = vert_shader;
                self.frag_shader_module = frag_shader;
                self.pipeline_layout = pipeline_layout;
                self.render_pass = render_pass;
                self.graphics_pipeline = graphics_pipeline;
                self.swapchain_framebuffers = framebuffers;
                self.command_pool = command_pool;
                self.command_buffers = command_buffers;

            }}
        }




    }

    fn recreate_swaphain(device: &Device, instance: &Instance, surface: &SurfaceKHR, surface_loader: &Surface, physical_device: &PhysicalDevice, main_queue_index: u32) -> (SurfaceFormatKHR, Swapchain, SwapchainKHR, Vec<ImageView>, ShaderModule, ShaderModule, PipelineLayout, RenderPass, Pipeline, Vec<Framebuffer>, CommandPool, Vec<CommandBuffer>) {
        let surface_capabilities = unsafe {surface_loader.get_physical_device_surface_capabilities(*physical_device,*surface)}.unwrap();
        let surface_formats = unsafe { surface_loader.get_physical_device_surface_formats(*physical_device,*surface)}.unwrap();
        let surface_format = *surface_formats.iter().find(|format| {
            return format.format == vk::Format::B8G8R8A8_SRGB && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR;
        }).unwrap();

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(*surface)
            .min_image_count(surface_capabilities.min_image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(surface_capabilities.current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::IMMEDIATE)
            .clipped(true);

        let swapchain_loader = Swapchain::new(instance, device);
        let swapchain = unsafe {swapchain_loader.create_swapchain(&swapchain_create_info, None)}.unwrap();

        let swapchain_images = unsafe {swapchain_loader.get_swapchain_images(swapchain)}.unwrap();

        let swapchain_image_views: Vec<vk::ImageView> = swapchain_images.iter().map(|image|{
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

        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(main_queue_index);
        let command_pool = unsafe {device.create_command_pool(&command_pool_create_info, None)}.unwrap();

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
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
        return (surface_format,
                swapchain_loader,
                swapchain,
                swapchain_image_views,
                vert_shader_module,
                frag_shader_module,
                pipeline_layout,
                render_pass,
                graphics_pipeline,
                swapchain_framebuffers,
                command_pool,
                command_buffers)
    }


    fn create_shader_module(device: &Device, spv_code: Vec<u32>) -> vk::ShaderModule{
        let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(&spv_code);
        unsafe {device.create_shader_module(&shader_module_create_info, None)}.unwrap()
    }

    fn cleanup_swapchain(&mut self) {

                self.swapchain_framebuffers.iter().for_each(|framebuffer| unsafe{self.device.destroy_framebuffer(*framebuffer, None)});
        unsafe {self.device.destroy_pipeline(self.graphics_pipeline,None)};
        unsafe {self.device.destroy_pipeline_layout(self.pipeline_layout,None)};
        unsafe {self.device.destroy_render_pass(self.render_pass, None)};
        unsafe {self.device.destroy_command_pool(self.command_pool, None)};
        self.swapchain_image_views.iter().for_each(|image_view| unsafe {self.device.destroy_image_view(*image_view, None)});
        unsafe {self.swapchain_loader.destroy_swapchain(self.swapchain, None)};

    }

    fn cleanup(&mut self){
        VulkanData::cleanup_swapchain(self);

        unsafe {
            self.device.destroy_semaphore(self.render_finished_semaphore, None);
            self.device.destroy_semaphore(self.image_available_semaphore, None);
            self.device.destroy_fence(self.in_flight_fence, None);
        }
        unsafe {self.device.destroy_shader_module(self.vert_shader_module,None)};
        unsafe {self.device.destroy_shader_module(self.frag_shader_module, None)};
        unsafe {self.surface_loader.destroy_surface(self.surface,None)};
        unsafe {self.device.destroy_device(None)};
        unsafe {self.instance.destroy_instance(None)};
    }
}

fn main(){
    let path = env::current_dir().unwrap();
    println!("The current directory is {}", path.display());

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).expect("Failed to build window");
    let mut vulkan_data = VulkanData::new(&window);

    event_loop.run(move |event, _, control_flow|{
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                unsafe{ vulkan_data.device.device_wait_idle()}.unwrap();
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