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


use nalgebra::Translation3;
use nalgebra::{Matrix4, Vector2, Vector3};


use rust_vulkan_engine::game::client::Game;

use rust_vulkan_engine::network::{ClientState, Packet};
use rust_vulkan_engine::renderer::*;
use rust_vulkan_engine::renderer::vulkan_data::VulkanData;
use rust_vulkan_engine::support;
use rust_vulkan_engine::support::*;
use rust_vulkan_engine::voxels::marching_cubes::World;

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
    window.set_cursor_grab(false).unwrap();
    window.set_cursor_visible(true);
    let mut state = egui_winit::State::new(&window);

    let mut vulkan_data = VulkanData::new();
    println!("Vulkan Data created");
    vulkan_data.init_vulkan(&window, |vulkan_data| {
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

                    window.set_title(format!("Frametimes: {:}", average_frametime).as_str());

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

    let view_matrix = Translation3::new(0.0, 0.0, 10.0).inverse().to_homogeneous();
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
