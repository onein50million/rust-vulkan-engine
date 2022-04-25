use egui::Button;
use egui::Color32;
use egui::Frame;
use egui::Grid;
use egui::Label;
use egui::Rgba;
use egui::ScrollArea;
use egui::Style;
use egui::TopBottomPanel;
use egui::Ui;
use egui::Window;
use egui::plot::Value;
use egui::plot::Values;
use nalgebra::Point3;
use rust_vulkan_engine::game::client::GameObject;
use rust_vulkan_engine::renderer::*;
use rust_vulkan_engine::support::*;
use egui_winit::winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use egui_winit::winit::event_loop::{ControlFlow, EventLoop};
use egui_winit::winit::window::WindowBuilder;
use rust_vulkan_engine::world::*;
use std::env;
use std::net::UdpSocket;
use std::time::Instant;
use nalgebra::{Matrix4, Rotation3, Translation3, UnitQuaternion, Vector2, Vector3};
use winit::event::{MouseButton, MouseScrollDelta};
use rust_vulkan_engine::game::client::{Game};
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
    fn process(&mut self, game: &mut Game){
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
                        (_, _) => {
                            println!("Unknown state/packet combo")
                        }
                    }
                }
            }
        }

        self.socket.send(&Packet::Input(game.inputs).to_bytes()).unwrap();

    }
}

// fn get_unique_id() -> u64{
//     fastrand::u64(..)
// }

fn add_plot(ui: &mut Ui, values: &[Value], heading: &str, num_samples: usize, start: usize){
    let values = &values[start..];
    let num_samples = num_samples.min(values.len());
    let chunk_size = values.len()/num_samples;
    let line = egui::plot::Line::new(Values::from_values_iter(values.chunks(chunk_size).map(|values|{
        let mut sum = 0.0;
        let mut month_sum = 0.0;
        for value in values{
            sum += value.y;
            month_sum += value.x;
        }
        Value::new(month_sum/values.len() as f64, sum/values.len() as f64)
    })));
    ui.add(Label::new(heading).heading());
    ui.add(egui::plot::Plot::new(1).line(line).view_aspect(1.0).allow_zoom(false).include_y(0.0).include_x(values[0].x).allow_drag(false));
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


    let world = World::new();

    let planet_mesh = rust_vulkan_engine::planet_gen::get_planet(World::RADIUS as f32);
    let planet_render_object_index = VulkanData::load_vertices_and_indices(
        &mut vulkan_data,
        planet_mesh.vertices,
        planet_mesh.indices,
        true,
    );

    /* Dummy objects to fill offsets */
    vulkan_data.load_folder("models/planet/deep_water".parse().unwrap()); 
    vulkan_data.load_folder("models/planet/shallow_water".parse().unwrap());
    vulkan_data.load_folder("models/planet/foliage".parse().unwrap());
    vulkan_data.load_folder("models/planet/desert".parse().unwrap());
    vulkan_data.load_folder("models/planet/mountain".parse().unwrap());
    vulkan_data.load_folder("models/planet/snow".parse().unwrap());


    // let planet_render_object_index = VulkanData::load_vertices_and_indices(
    //     &mut vulkan_data,
    //     world.get_vertices(),
    //     world.get_indices()
    // );
    vulkan_data.update_vertex_and_index_buffers();
    
    let mut game = Game::new(
        planet_render_object_index,
        world
    );
    println!("Game created");
    let socket = UdpSocket::bind("0.0.0.0:0").unwrap();
    socket.connect("127.0.0.1:2022").unwrap();
    socket.set_nonblocking(true).unwrap();
    let mut client = Client{
        socket,
        state: ClientState::Disconnected
    };

    //Do some simulation to see if state stabilizes
    let months = 12*10000;
    let mut population_history = Vec::with_capacity(months);
    let mut price_histories = vec![Vec::with_capacity(months); Good::VARIANT_COUNT];
    for month in 0..months{
        population_history.push(
            Value::new(month as f64,
            game.world.provinces[0].pops.population()));
        

        for good_index in 0..Good::VARIANT_COUNT{
            price_histories[good_index].push(Value::new(
                month as f64,
                game.world.provinces[0].market.price[good_index]
            ));
        }
        game.world.process(1.0/12.0);
    }

    client.socket.send(&Packet::RequestConnect{ username: "daniel".to_string()}.to_bytes()).unwrap();
    client.state = ClientState::ConnectionAwaiting;


    let mut current_month = months;
    let mut texture_version = 0;


    let mut population_graph_window_open = false;
    let mut price_graph_window_open = false;
    let mut pop_table_open = false;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        let raw_input = state.take_egui_input(&window);
        ctx.begin_frame(raw_input);

        if ctx.texture().version != texture_version {
            texture_version = ctx.texture().version;
            vulkan_data.update_ui_texture(ctx.texture());
        }

        // egui::SidePanel::left("my_left_panel").show(&ctx, |ui| {
        //     ScrollArea::vertical().show(ui, |ui|{
        //         ui.add(egui::Label::new(format!("{:}", game.world.provinces[0])))
        //     })
        // });


        TopBottomPanel::top("get_unique_id()").show(&ctx, |ui|{
            ui.label("hello world!");
            ui.horizontal(|ui|{
                ui.checkbox(&mut population_graph_window_open, "Population Plot");
                ui.checkbox(&mut pop_table_open, "Pops Table");
                ui.checkbox(&mut price_graph_window_open, "Price Plots");
            });
        });

        if population_graph_window_open{
            Window::new("Population Plot").show(&ctx, |ui|{
                ui.label("hello!");
                add_plot(ui, &population_history, "Total Population", 1000, 0);
            });
        }
        if price_graph_window_open{
            Window::new("Price Plots").show(&ctx, |ui|{
                ScrollArea::vertical().show(ui, |ui|{
                    for good_index in 0..Good::VARIANT_COUNT{
                        let good = Good::try_from(good_index).unwrap();
                            add_plot(ui, &price_histories[good_index], &format!("{:?} Price", good), 1000, 0);    
                    }
                });

            });

        }
        if pop_table_open{
            Window::new("Pop Table").show(&ctx, |ui|{
                Grid::new(0).striped(true).show(ui, |ui|{
                    ui.label("Culture");       
                    ui.label("Industry");       
                    ui.label("Population");
                    ui.label("Money");
                    for good_index in 0..Good::VARIANT_COUNT{
                        ui.label(format!("Owned {:?}", Good::try_from(good_index).unwrap()));
                    }
                    ui.end_row();     
                    for (i, slice) in game.world.provinces[0].pops.pop_slices.iter().enumerate(){
                        let culture = Culture::try_from(i / Industry::VARIANT_COUNT).unwrap();
                        let industry = Industry::try_from(i % Industry::VARIANT_COUNT).unwrap();
                        ui.label(format!("{:?}", culture));       
                        ui.label(format!("{:?}", industry));       
                        ui.label(format!("{:.0}",slice.population));
                        ui.label(format!("{:.2}",slice.money));
                        for good_index in 0..Good::VARIANT_COUNT{
                            ui.label(format!("{:.2}",slice.owned_goods[good_index]));
                        }
                        ui.end_row();
                    }
                })
            });
        }
        // println!("egui texture version: {:?}", ctx.texture().version);

        let (output, shapes) = ctx.end_frame();
        let clipped_meshes = ctx.tessellate(shapes); // create triangles to paint

        let mut vertices = vec![];
        let mut indices = vec![];

        let mut current_index_offset = 0;
        for mesh in clipped_meshes {
            vertices.extend_from_slice(&mesh.1.vertices);
            indices.extend(mesh.1.indices.iter().map(|index| index + current_index_offset as u32));
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
                client.process(&mut game);

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
                    population_history.push(
                        Value::new(current_month as f64,
                        game.world.provinces[0].pops.population()));
                    game.process(1.0/12.0);
                    current_month += 1;
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

    // let view_matrix: Matrix4<f64> = (Rotation3::from_euler_angles(0.0, game.inputs.angle, 0.0)
    //     .to_homogeneous()
    //     * (Translation3::new(0.0,-10.0,0.0).to_homogeneous()
    //     * UnitQuaternion::face_towards(
    //     &Vector3::new(1.0, 1.0, 1.0),
    //     &Vector3::new(0.0, -1.0, 0.0),
    // ).to_homogeneous()))
    //     .try_inverse()
    //     .unwrap();

    // let time = game.start_time.elapsed().as_secs_f64();
    let view_matrix = game.camera.get_view_matrix();

    let render_object = &mut vulkan_data.objects[game.planet.render_object_index];
    let model_matrix = (Matrix4::from(Translation3::from(game.planet.position))
        * Matrix4::from(Rotation3::from(game.planet.rotation)))
        .cast();
    render_object.model = model_matrix;

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
