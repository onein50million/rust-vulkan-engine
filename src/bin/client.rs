use egui::Color32;
use egui::plot::Value;
use egui::plot::Values;
use egui::FontDefinitions;
use egui::FontFamily;
use egui::Label;


use egui::Ui;
use egui_winit::winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use egui_winit::winit::event_loop::{ControlFlow, EventLoop};
use egui_winit::winit::window::WindowBuilder;


use erupt::vk;
use nalgebra::Translation3;
use nalgebra::{Matrix4, Vector2, Vector3};


use rand::Rng;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rust_vulkan_engine::game::client::Game;

use rust_vulkan_engine::network::{ClientState, Packet};
use rust_vulkan_engine::renderer::*;
use rust_vulkan_engine::renderer::buffers::UnmappedBuffer;
use rust_vulkan_engine::renderer::combination_types::CombinedDescriptor;
use rust_vulkan_engine::renderer::combination_types::CombinedSampledImage;
use rust_vulkan_engine::renderer::combination_types::DescriptorInfoData;
use rust_vulkan_engine::renderer::vulkan_data::VulkanData;
use rust_vulkan_engine::support;
use rust_vulkan_engine::support::*;
use rust_vulkan_engine::voxels::marching_cubes::World;
use winit::event::DeviceEvent;

use std::fs;
use std::net::UdpSocket;

use winit::event::{MouseButton, MouseScrollDelta};


struct Client {
    socket: UdpSocket,
    state: ClientState,
}

impl Client {
    fn process(&mut self, game: &mut Game) {
        let mut buffer = [0; 1024];
        while let Ok(num_bytes) = self.socket.recv(&mut buffer) {
            let unprocessed_datagram = &mut buffer[..num_bytes];
            match Packet::from_bytes(unprocessed_datagram) {
                None => {
                    println!("Invalid packet received from server")
                }
                Some(packet) => match (self.state, packet) {
                    (ClientState::ConnectionAwaiting, Packet::RequestAccepted) => {
                        self.state = ClientState::Connected;
                        println!("Client connected");
                    }
                    (_, _) => {
                        println!("Unknown state/packet combo")
                    }
                },
            }
        }

        self.socket
            .send(&Packet::Input(game.inputs).to_bytes())
            .unwrap();
    }
}

fn main() {
    let mut frametime: std::time::Instant = std::time::Instant::now();
    let mut time_since_last_frame: f64 = 0.0; //seconds

    let mut frametimes = [0.0; FRAME_SAMPLES];
    let mut frametimes_start = 0;
    let mut last_tick = std::time::Instant::now();

    let mut fonts = FontDefinitions::default();
    for (_, (_, font_size)) in &mut fonts.family_and_size {
        *font_size *= 1.3;
    }

    // Install my own font (maybe supporting non-latin characters):
    fonts.font_data.insert(
        "custom_font".to_owned(),
        std::borrow::Cow::Borrowed(include_bytes!("../../fonts/ShareTechMono-Regular.ttf")),
    );

    // Put my font first (highest priority):
    fonts
        .fonts_for_family
        .get_mut(&FontFamily::Proportional)
        .unwrap()
        .insert(0, "custom_font".to_owned());
    let mut ctx = egui::CtxRef::default();
    ctx.set_fonts(fonts);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .build(&event_loop)
        .expect("Failed to build window");
    window.set_cursor_grab(true).unwrap();
    // window.set_cursor
    window.set_cursor_visible(true);
    let mut state = egui_winit::State::new(&window);

    let mut vulkan_data = VulkanData::new();
    println!("Vulkan Data created");
    
    vulkan_data.init_vulkan(&window, |vulkan_data| {
        let mut rng = rand::thread_rng();
        // let voxel_image_size = (118,121,60);
        let voxel_image_size = (128, 64, 128);
    
        let (voxel_image_a, voxel_image_a_view) = {
            let image_info = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_3D)
            .extent(vk::Extent3D {
                width: voxel_image_size.0,
                height: voxel_image_size.1,
                depth: voxel_image_size.2,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R32G32B32A32_SINT)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC) 
            .samples(vk::SampleCountFlagBits::_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let allocation_info = vk_mem_erupt::AllocationCreateInfo {
                ..Default::default()
            };
    
            let (image, allocation, _) = vulkan_data
                .allocator
                .as_ref()
                .unwrap()
                .create_image(&image_info, &allocation_info)
                .unwrap();
            vulkan_data.lazy_transition_image_layout(image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, vk::ImageSubresourceRange{
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
            let view_info = vk::ImageViewCreateInfoBuilder::new()
            .image(image)
            .view_type(vk::ImageViewType::_3D)
            .format(vk::Format::R32G32B32A32_SINT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .flags(vk::ImageViewCreateFlags::empty());
    
            let image_view = unsafe {
                vulkan_data.device
                    .as_ref()
                    .unwrap()
                    .create_image_view(&view_info, None)
            }
            .unwrap();
    
            let mut data: Vec<u8> = vec![0; (voxel_image_size.0 * voxel_image_size.1 * voxel_image_size.2) as usize];
            let input_data = std::fs::read("test_room.danvox").unwrap();
            let input_size = (118,121,60);
            for (index, input) in  input_data.into_iter().enumerate(){
                let mut input_coord = (
                    index % input_size.0,
                    (index / input_size.0) % input_size.1,
                    index / (input_size.0 * input_size.1),
                );
                std::mem::swap(&mut input_coord.1, &mut input_coord.2);
                // let output_index = ((input_coord.0 * (voxel_image_size.1 as usize) + input_coord.1) * (voxel_image_size.2 as usize)) + input_coord.2;
                let output_index = input_coord.0 + input_coord.1 * (voxel_image_size.0 as usize) + input_coord.2 * ((voxel_image_size.0 as usize) * (voxel_image_size.1 as usize));

                data[output_index] = input;
            }

            // let mut data: Vec<u8> = vec![0; (voxel_image_size.0 * voxel_image_size.1 * voxel_image_size.2) as usize];
            // for (i, voxel) in data.iter_mut().enumerate(){
            //     if i < 128*64 && i % 30 == 0{
            //         *voxel = 1;
            //     }
            // }
            assert_eq!(data.len(), (voxel_image_size.0 * voxel_image_size.1 * voxel_image_size.2) as usize);

            let mut data: Vec<_> = data.into_iter().enumerate().map(|(index, value)|{
                if value > 0 {
                    [(index % (voxel_image_size.0 as usize)) as i32,
                    ((index / (voxel_image_size.0 as usize)) % (voxel_image_size.1 as usize)) as i32,
                    (index / ((voxel_image_size.0 as usize) * (voxel_image_size.1 as usize))) as i32,
                    0i32
                    ]
                }else{
                    [-1,-1,-1, 0]
                }
            }).collect();


            let seeds: Vec<_> = data.iter().copied().filter(|v| v[0] > -1).collect();
            for voxel in &mut data{
                if voxel[0] == -1{
                    *voxel = *seeds.choose(&mut rng).unwrap();
                }
            }


            let data: Vec<_> = data.into_iter().flat_map(|color|color.into_iter().map(|a|a.to_ne_bytes())).collect();


            let transfer_buffer = UnmappedBuffer::new(&vulkan_data, vk::BufferUsageFlags::TRANSFER_SRC, &data);
            let command_buffer = vulkan_data.begin_single_time_commands();
            unsafe {
                vulkan_data
                    .device
                    .as_ref()
                    .unwrap()
                    .cmd_copy_buffer_to_image(
                        command_buffer,
                        transfer_buffer.buffer,
                        image,
                        vk::ImageLayout::GENERAL,
                        &[
                            vk::BufferImageCopyBuilder::new()
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
                                    width: voxel_image_size.0,
                                    height: voxel_image_size.1,
                                    depth: voxel_image_size.2,
                                }),
                        ],
                    )
            };
            vulkan_data.end_single_time_commands(command_buffer);
            (image, image_view)
        };
        let (voxel_image_b, voxel_image_b_view) = {
            let image_info = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_3D)
            .extent(vk::Extent3D {
                width: voxel_image_size.0,
                height: voxel_image_size.1,
                depth: voxel_image_size.2,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R32G32B32A32_SINT)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC) 
            .samples(vk::SampleCountFlagBits::_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let allocation_info = vk_mem_erupt::AllocationCreateInfo {
                ..Default::default()
            };
    
            let (image, allocation, _) = vulkan_data
                .allocator
                .as_ref()
                .unwrap()
                .create_image(&image_info, &allocation_info)
                .unwrap();
            vulkan_data.lazy_transition_image_layout(image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, vk::ImageSubresourceRange{
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
            let view_info = vk::ImageViewCreateInfoBuilder::new()
            .image(image)
            .view_type(vk::ImageViewType::_3D)
            .format(vk::Format::R32G32B32A32_SINT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .flags(vk::ImageViewCreateFlags::empty());
    
            let image_view = unsafe {
                vulkan_data.device
                    .as_ref()
                    .unwrap()
                    .create_image_view(&view_info, None)
            }
            .unwrap();
            (image, image_view)
        };
        let command_buffer = vulkan_data.begin_single_time_commands();
        unsafe{
            vulkan_data.device.as_ref().unwrap().cmd_copy_image(command_buffer, voxel_image_a, vk::ImageLayout::GENERAL, voxel_image_b, vk::ImageLayout::GENERAL, &[
                vk::ImageCopyBuilder::new().extent(
                    vk::Extent3D{
                        width: voxel_image_size.0,
                        height: voxel_image_size.1,
                        depth: voxel_image_size.2,
                    }
                ).dst_subresource(vk::ImageSubresourceLayers{
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                }).src_subresource(vk::ImageSubresourceLayers{
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
            ]);
        }
        vulkan_data.end_single_time_commands(command_buffer);

        let (sdf_image, sdf_image_view, sdf_allocation) = {
            let image_info = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_3D)
            .extent(vk::Extent3D {
                width: voxel_image_size.0,
                height: voxel_image_size.1,
                depth: voxel_image_size.2,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8_UINT)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlagBits::_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let allocation_info = vk_mem_erupt::AllocationCreateInfo {
                ..Default::default()
            };
    
            let (image, allocation, _) = vulkan_data
                .allocator
                .as_ref()
                .unwrap()
                .create_image(&image_info, &allocation_info)
                .unwrap();

            vulkan_data.lazy_transition_image_layout(image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, vk::ImageSubresourceRange{
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
            let view_info = vk::ImageViewCreateInfoBuilder::new()
            .image(image)
            .view_type(vk::ImageViewType::_3D)
            .format(vk::Format::R8_UINT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .flags(vk::ImageViewCreateFlags::empty());
    
            let image_view = unsafe {
                vulkan_data.device
                    .as_ref()
                    .unwrap()
                    .create_image_view(&view_info, None)
            }
            .unwrap();
            (image, image_view, allocation)
        };
    
        let shader_module = VulkanData::create_shader_module(vulkan_data.device.as_ref().unwrap(), erupt::utils::decode_spv(&fs::read("shaders/3dSDF.spv").unwrap()).unwrap());

        let mut buffer_switch = true;
        println!("Running Voxel SDF Shader");
        let mut extra_rounds = 100;
        let mut divisor = 2;
        let max_length = voxel_image_size.0.max(voxel_image_size.1.max(voxel_image_size.2));
        while extra_rounds >= 0{
            println!("Current divisor: {divisor}");

            let combined_descriptors = if buffer_switch {
                [CombinedDescriptor {
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    descriptor_info: DescriptorInfoData::Image {
                        image_view: voxel_image_a_view,
                        sampler: None,
                        layout: vk::ImageLayout::GENERAL,
                    },
                },
                CombinedDescriptor {
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    descriptor_info: DescriptorInfoData::Image {
                        image_view: voxel_image_b_view,
                        sampler: None,
                        layout: vk::ImageLayout::GENERAL,
                    },
                },]
            }else{
                [CombinedDescriptor {
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    descriptor_info: DescriptorInfoData::Image {
                        image_view: voxel_image_b_view,
                        sampler: None,
                        layout: vk::ImageLayout::GENERAL,
                    },
                },
                CombinedDescriptor {
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    descriptor_info: DescriptorInfoData::Image {
                        image_view: voxel_image_a_view,
                        sampler: None,
                        layout: vk::ImageLayout::GENERAL,
                    },
                },]
            };

            #[repr(C)]
            struct SdfPushConstants{
                current_divisor: i32,
                direction: i32,
                // random_seed: u32,
            }
            let group_size = (voxel_image_size.0/4, voxel_image_size.1/4, voxel_image_size.2/4);
            for direction in 0..(3*3*3){
                vulkan_data.run_arbitrary_compute_shader(shader_module, SdfPushConstants{
                    current_divisor: divisor as i32,
                    direction,
                    // random_seed: rng.gen(),
                }, &combined_descriptors, group_size);
            }
            if max_length/(divisor * 2) >= 1{
                divisor *= 2;
            }else{
                extra_rounds -= 1;
            }
            let command_buffer = vulkan_data.begin_single_time_commands();

            let (source_image, dest_image) = if buffer_switch{
                (voxel_image_b, voxel_image_a)
            }else{
                (voxel_image_a, voxel_image_b)
            };
            unsafe{
                vulkan_data.device.as_ref().unwrap().cmd_copy_image(command_buffer, source_image, vk::ImageLayout::GENERAL, dest_image, vk::ImageLayout::GENERAL, &[
                    vk::ImageCopyBuilder::new().extent(
                        vk::Extent3D{
                            width: voxel_image_size.0,
                            height: voxel_image_size.1,
                            depth: voxel_image_size.2,
                        }
                    ).dst_subresource(vk::ImageSubresourceLayers{
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    }).src_subresource(vk::ImageSubresourceLayers{
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                ]);
            }
            vulkan_data.end_single_time_commands(command_buffer);
            buffer_switch = !buffer_switch;
        }

        {
            let shader_module = VulkanData::create_shader_module(vulkan_data.device.as_ref().unwrap(), erupt::utils::decode_spv(&fs::read("shaders/bakeVoronoi.spv").unwrap()).unwrap());
            let combined_descriptors = [CombinedDescriptor {
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                descriptor_info: DescriptorInfoData::Image {
                    image_view: if buffer_switch{ voxel_image_a_view}else { voxel_image_b_view},
                    sampler: None,
                    layout: vk::ImageLayout::GENERAL,
                },
            },
            CombinedDescriptor {
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                descriptor_info: DescriptorInfoData::Image {
                    image_view: sdf_image_view,
                    sampler: None,
                    layout: vk::ImageLayout::GENERAL,
                },
            },];
            let group_size = (voxel_image_size.0/4, voxel_image_size.1/4, voxel_image_size.2/4);
            vulkan_data.run_arbitrary_compute_shader(shader_module, 0, &combined_descriptors, group_size);

        }
        
        vulkan_data.lazy_transition_image_layout(sdf_image, vk::ImageLayout::GENERAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageSubresourceRange{
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

        let sdf_sampler = {
            let sampler_info = vk::SamplerCreateInfoBuilder::new()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
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
    
            unsafe {
                vulkan_data
                    .device
                    .as_ref()
                    .unwrap()
                    .create_sampler(&sampler_info, None)
                    .unwrap()
            }
        };
        vulkan_data.voxel_sdf = Some(CombinedSampledImage{
            image: sdf_image,
            image_view: sdf_image_view,
            sampler: sdf_sampler,
            allocation: sdf_allocation,
            width: voxel_image_size.0,
            height: voxel_image_size.1,
        });

    });
    println!("Vulkan Data initialized");

 



    // vulkan_data.load_folder("models/test_ball".into());

    let world = World::new_random();
    {
        let vertices = world.generate_mesh();
        let indices = (0..vertices.len()).collect();
        vulkan_data.load_vertices_and_indices(vertices, indices , false);
        vulkan_data.update_vertex_and_index_buffers();
    }
    let mut game = Game::new();
    println!("Game created");
    let socket = UdpSocket::bind("0.0.0.0:0").unwrap();
    socket.connect("127.0.0.1:2022").unwrap();
    socket.set_nonblocking(true).unwrap();
    let mut client = Client {
        socket,
        state: ClientState::Disconnected,
    };
    client
        .socket
        .send(
            &Packet::RequestConnect {
                username: "daniel".to_string(),
            }
            .to_bytes(),
        )
        .unwrap();
    client.state = ClientState::ConnectionAwaiting;

    let mut texture_version = 0;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        let raw_input = state.take_egui_input(&window);
        ctx.begin_frame(raw_input);

        if ctx.texture().version != texture_version {
            texture_version = ctx.texture().version;
            vulkan_data.update_ui_texture(ctx.texture());
        }

        let (output, shapes) = ctx.end_frame();
        let clipped_meshes = ctx.tessellate(shapes); // create triangles to paint

        let mut vertices = vec![];
        let mut indices = vec![];

        let mut current_index_offset = 0;
        for mesh in clipped_meshes {
            vertices.extend_from_slice(&mesh.1.vertices);
            indices.extend(
                mesh.1
                    .indices
                    .iter()
                    .map(|index| index + current_index_offset as u32),
            );
            current_index_offset += mesh.1.vertices.len();
        }

        // println!("vertex length: {:}", vertices.len());
        // println!("index length: {:}", indices.len());
        vulkan_data.ui_data.update_buffers(&vertices, &indices);
        state.handle_output(&window, &ctx, output);
        match event {
            Event::DeviceEvent { device_id, event }=>{
                match event{
                    DeviceEvent::MouseMotion { delta }=>{
                        game.mouse_buffer.x += delta.0;
                        game.mouse_buffer.y += delta.1;
                    }
                    _ =>{}
                }
            },
            Event::WindowEvent { event, .. } => {
                let egui_handling = state.on_event(&ctx, &event);
                if !egui_handling {
                    match event {
                        WindowEvent::CloseRequested => {
                            unsafe { vulkan_data.device.as_ref().unwrap().device_wait_idle() }
                                .unwrap();
                            close_app(&mut vulkan_data, control_flow);
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            // game.mouse_position.x = position.x;
                            // game.mouse_position.y = position.y;
                            // let old_mouse_position = game.mouse_position;
                            match vulkan_data.surface_capabilities {
                                Some(surface_capabilities) => {
                                    game.mouse_position.x = -((position.x
                                        / surface_capabilities.current_extent.width as f64)
                                        * 2.0
                                        - 1.0);
                                    game.mouse_position.y = -((position.y
                                        / surface_capabilities.current_extent.height as f64)
                                        * 2.0
                                        - 1.0);
                                }
                                None => {}
                            }
                            // game.mouse_buffer += old_mouse_position - game.mouse_position;
                        }
                        WindowEvent::MouseInput { button, state, .. } => {
                            match button {
                                MouseButton::Left => {
                                    if state == ElementState::Released {
                                        game.inputs.left_click = true;
                                    }
                                }
                                MouseButton::Middle => {
                                    game.inputs.panning = match state {
                                        ElementState::Pressed => true,
                                        ElementState::Released => false,
                                    }
                                }
                                MouseButton::Right => {
                                    if state == ElementState::Released {
                                        game.inputs.right_click = true;
                                    }
                                }
                                _ => {}
                            };
                        }
                        WindowEvent::MouseWheel { delta, .. } => match delta {
                            MouseScrollDelta::LineDelta(_, vertical_lines) => {
                                game.inputs.zoom = (game.inputs.zoom
                                    * 1.1f64.powf(-vertical_lines as f64))
                                .clamp(0.00001, 2.0);
                            }
                            _ => {}
                        },
                        WindowEvent::KeyboardInput { input, .. } => {
                            // if input.virtual_keycode == Some(VirtualKeyCode::Escape)
                            //     && input.state == ElementState::Released
                            // {
                            //     close_app(&mut vulkan_data, control_flow);
                            // }
                            if input.virtual_keycode == Some(VirtualKeyCode::W) {
                                game.inputs.up = match input.state {
                                    ElementState::Released => 0.0,
                                    ElementState::Pressed => 1.0,
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::S) {
                                game.inputs.down = match input.state {
                                    ElementState::Released => 0.0,
                                    ElementState::Pressed => 1.0,
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::A) {
                                game.inputs.left = match input.state {
                                    ElementState::Released => 0.0,
                                    ElementState::Pressed => 1.0,
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::D) {
                                game.inputs.right = match input.state {
                                    ElementState::Released => 0.0,
                                    ElementState::Pressed => 1.0,
                                };
                            }

                            if input.virtual_keycode == Some(VirtualKeyCode::Up) {
                                game.inputs.exposure *= match input.state {
                                    ElementState::Released => 1.1,
                                    ElementState::Pressed => 1.0,
                                };
                                dbg!(game.inputs.exposure);
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Down) {
                                game.inputs.exposure *= match input.state {
                                    ElementState::Released => 0.9,
                                    ElementState::Pressed => 1.0,
                                };
                                dbg!(game.inputs.exposure);
                            }

                            if input.virtual_keycode == Some(VirtualKeyCode::Left) {
                                game.inputs.angle += match input.state {
                                    ElementState::Released => -0.05,
                                    ElementState::Pressed => 0.0,
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Right) {
                                game.inputs.angle += match input.state {
                                    ElementState::Released => 0.05,
                                    ElementState::Pressed => 0.0,
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Key1) {
                                match input.state {
                                    ElementState::Released => {
                                        game.inputs.map_mode = support::map_modes::SATELITE
                                    }
                                    _ => {}
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Key2) {
                                match input.state {
                                    ElementState::Released => {
                                        game.inputs.map_mode = support::map_modes::PAPER
                                    }
                                    _ => {}
                                };
                            }
                        }
                        _ => {}
                    }
                }
            }
            Event::MainEventsCleared => {
                //app update code
                // window.request_redraw();
                let elapsed_time = frametime.elapsed().as_secs_f64();
                frametime = std::time::Instant::now();

                //Network loop
                // client.process(&mut game);

                if time_since_last_frame > 1.0 / FRAMERATE_TARGET {
                    let delta_time = last_tick.elapsed().as_secs_f64();
                    frametimes[frametimes_start] = delta_time;
                    last_tick = std::time::Instant::now();

                    frametimes_start = (frametimes_start + 1) % FRAME_SAMPLES;
                    let mut average_frametime = 0.0;
                    for i in 0..frametimes.len() {
                        average_frametime += frametimes[i];
                    }
                    average_frametime /= FRAME_SAMPLES as f64;

                    window.set_title(format!("Fps: {:.0} Frametimes: {:}", 1.0/average_frametime, average_frametime).as_str());

                    game.process(
                        delta_time,
                    );

                    update_renderer(&game, &mut vulkan_data);
                    vulkan_data.transfer_data_to_gpu();

                    let _draw_frame_result = vulkan_data.draw_frame();
                    time_since_last_frame -= 1.0 / FRAMERATE_TARGET;
                }
                time_since_last_frame += elapsed_time;
            }
            _ => {}
        }
    })
}

fn update_renderer(game: &Game, vulkan_data: &mut VulkanData) {
    let clip = Matrix4::<f64>::identity();

    let projection = vulkan_data.get_projection(game.inputs.zoom);

    let projection_matrix = clip * projection.to_homogeneous();
    vulkan_data.post_process_ubo.as_mut().unwrap().get_mut().proj = projection_matrix.cast();

    // let view_matrix = Translation3::new(0.0, 0.0, 10.0).inverse().to_homogeneous();
    let view_matrix = game.player.get_view_matrix();
    vulkan_data.uniform_buffer_object.view = view_matrix.cast();
    vulkan_data.uniform_buffer_object.proj = projection_matrix.cast();

    vulkan_data.uniform_buffer_object.time = game.start_time.elapsed().as_secs_f32();
    vulkan_data.uniform_buffer_object.player_position = Vector3::zeros();
    vulkan_data.uniform_buffer_object.exposure = game.inputs.exposure as f32;

    vulkan_data.uniform_buffer_object.mouse_position = Vector2::new(
        (game.mouse_position.x) as f32,
        (game.mouse_position.y) as f32,
    );
}

fn close_app(vulkan_data: &mut VulkanData, control_flow: &mut ControlFlow) {
    *control_flow = ControlFlow::Exit;
}
