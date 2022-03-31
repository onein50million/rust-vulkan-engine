use rust_vulkan_engine::game::*;
use rust_vulkan_engine::renderer::*;
use rust_vulkan_engine::support::*;
use egui_winit::winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use egui_winit::winit::event_loop::{ControlFlow, EventLoop};
use egui_winit::winit::window::WindowBuilder;
use std::env;
use std::net::UdpSocket;
use nalgebra::{Matrix4, Point3, Rotation3, Translation3, UnitQuaternion, Vector2, Vector3};
use slotmap::SlotMap;
use winit::event::{MouseButton, MouseScrollDelta};
use rust_vulkan_engine::game::client::{Game, ObjectType};
use rust_vulkan_engine::network::{ClientState, Packet};
//Coordinate system for future reference:
//from starting location
//up: negative y
//down: positive y
//forward: negative z
//backward: positive z
//left: negative x
//right: positive x


struct Client{
    socket: UdpSocket,
    state: ClientState
}

impl Client{
    fn process(&mut self, game: &mut Game, renderer: &mut VulkanData, inputs: &Inputs){
        let mut buffer = [0; 1024];
        while let Ok(num_bytes) = self.socket.recv(&mut buffer){
            let unprocessed_datagram = &mut buffer[..num_bytes];
            match Packet::from_bytes(unprocessed_datagram){
                None => {println!("Invalid packet received from server")}
                Some(packet) => {
                    match (self.state,packet){
                        (ClientState::ConnectionAwaiting, Packet::RequestAccepted) => {
                            self.state = ClientState::Connected;
                            println!("Client connected");
                        },
                        (ClientState::Connected, Packet::GameObject{key, object}) => {
                            game.game_objects[key].position = object.position;
                            //TODO rest of object transfer
                        },
                        (_, _) => {
                            println!("Unknown state/packet combo")
                        }
                    }
                }
            }
        }

        self.socket.send(&Packet::Input(*inputs).to_bytes()).unwrap();

    }
}



fn main() {

    let mut frametime: std::time::Instant = std::time::Instant::now();
    let mut time_since_last_frame: f64 = 0.0; //seconds

    let mut frametimes = [0.0; FRAME_SAMPLES];
    let mut frametimes_start = 0;
    let mut last_tick = std::time::Instant::now();

    let path = env::current_dir().unwrap();
    println!("The current directory is {}", path.display());
    let mut ctx = egui::CtxRef::default();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .build(&event_loop)
        .expect("Failed to build window");
    window.set_cursor_grab(false).unwrap();
    window.set_cursor_visible(true);
    let mut state = egui_winit::State::new(&window);


    let mut vulkan_data = VulkanData::new();
    println!("Vulkan Data created");
    vulkan_data.init_vulkan(&window);
    println!("Vulkan Data initialized");

    let mut game = Game::new();
    println!("Game created");
    let socket = UdpSocket::bind("0.0.0.0:0").unwrap();
    socket.connect("127.0.0.1:2022").unwrap();
    socket.set_nonblocking(true);
    let mut client = Client{
        socket,
        state: ClientState::Disconnected
    };

    client.socket.send(&Packet::RequestConnect{ username: "daniel".to_string()}.to_bytes()).unwrap();
    client.state = ClientState::ConnectionAwaiting;

    let mut texture_version = 0;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        let raw_input = state.take_egui_input(&window);
        ctx.begin_frame(raw_input);

        if ctx.texture().version != texture_version {
            println!("UI Texture changed");
            texture_version = ctx.texture().version;
            vulkan_data.update_ui_texture(ctx.texture());
        }

        // egui::SidePanel::left("my_left_panel").show(&ctx, |ui| {
        // });
        // println!("egui texture version: {:?}", ctx.texture().version);

        let (output, shapes) = ctx.end_frame();
        let clipped_meshes = ctx.tessellate(shapes); // create triangles to paint

        let mut vertices = vec![];
        let mut indices = vec![];
        for mesh in clipped_meshes {
            vertices.extend_from_slice(&mesh.1.vertices);
            indices.extend_from_slice(&mesh.1.indices);
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
                            game.mouse_position.x = position.x;
                            game.mouse_position.y = position.y;
                            // println!("Position: {:?}", position);
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
                                _ => {}
                            };
                        }
                        WindowEvent::MouseWheel { delta, .. } => match delta {
                            MouseScrollDelta::LineDelta(_, vertical_lines) => {
                                game.inputs.zoom = (game.inputs.zoom
                                    * 1.1f64.powf(vertical_lines as f64))
                                    .max(1.0);
                            }
                            _ => {}
                        },
                        WindowEvent::KeyboardInput { input, .. } => {
                            if input.virtual_keycode == Some(VirtualKeyCode::Escape)
                                && input.state == ElementState::Released
                            {
                                close_app(&mut vulkan_data, control_flow);
                            }
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
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Down) {
                                game.inputs.exposure *= match input.state {
                                    ElementState::Released => 0.9,
                                    ElementState::Pressed => 1.0,
                                };
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
                client.process();

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
                    game.process(delta_time);
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

    let view_matrix: Matrix4<f64> = (Rotation3::from_euler_angles(0.0, game.inputs.angle, 0.0)
        .to_homogeneous()
        * (Translation3::new(0.0,-10.0,0.0).to_homogeneous()
        * UnitQuaternion::face_towards(
        &Vector3::new(1.0, 1.0, 1.0),
        &Vector3::new(0.0, -1.0, 0.0),
    ).to_homogeneous()))
        .try_inverse()
        .unwrap();


    for game_object in game.game_objects.values(){
        let render_object = &mut vulkan_data.objects[game_object.render_object_index];
        let model_matrix = (Matrix4::from(Translation3::from(game_object.position))
            * Matrix4::from(Rotation3::from(game_object.rotation)))
            .cast();
        render_object.model = model_matrix;

        match &game_object.animation_handler {
            None => {}
            Some(animation_handler) => {
                render_object.set_animation(animation_handler.index, animation_handler.progress, animation_handler.previous_frame, animation_handler.next_frame)
            }
        }

    }

    vulkan_data.uniform_buffer_object.view = view_matrix.cast();
    vulkan_data.uniform_buffer_object.proj = projection_matrix.cast();

    vulkan_data.uniform_buffer_object.time = f32::NAN;
    vulkan_data.uniform_buffer_object.player_position = Vector3::zeros();
    vulkan_data.uniform_buffer_object.exposure = game.inputs.exposure as f32;

    vulkan_data.uniform_buffer_object.mouse_position = Vector2::new(
        (game.mouse_position.x) as f32,
        (game.mouse_position.y) as f32,
    );
}



fn close_app(vulkan_data: &mut VulkanData, control_flow: &mut ControlFlow) {
    *control_flow = ControlFlow::Exit;
    vulkan_data.cleanup();
}
