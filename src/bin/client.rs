use egui_winit::winit::event::{Event, WindowEvent};
use egui_winit::winit::event_loop::{ControlFlow, EventLoop};
use egui_winit::winit::window::WindowBuilder;

use gilrs::Gilrs;
use nalgebra::{Matrix4, UnitQuaternion, Vector2, Vector3};

use rust_vulkan_engine::game::{
    AnimationHandler, ClientGame, GameObject, GameTick, PlayerKey, PlayerMap, ClientPlayer, NetworkBuffer,
};

use rust_vulkan_engine::network::{ClientState, ClientToServerPacket, ServerToClientPacket, NETWORK_TICKRATE};

use rust_vulkan_engine::renderer::*;
use rust_vulkan_engine::support::*;

use std::net::UdpSocket;
use std::ops::Sub;
use std::path::PathBuf;
use std::time::{Instant, Duration};

use winit::event::{MouseButton, MouseScrollDelta};
//Coordinate system for future reference:
//from starting location
//up: negative y
//down: positive y
//forward: negative z
//backward: positive z
//left: negative x
//right: positive x

const POS_X_FACE: usize = 0;
const NEG_X_FACE: usize = 1;
const POS_Y_FACE: usize = 3; //These two are swapped because I had them backwards when I was doing the conversion and now it's a bunch of work to fix it
const NEG_Y_FACE: usize = 2; // ^^^^^
const POS_Z_FACE: usize = 4;
const NEG_Z_FACE: usize = 5;

//This is really confusing to figure out but hopefully it should give neighbouring pixels on a cubemap
fn get_neighbour_pixels(index: usize) -> Box<[usize]> {
    let (x, y) = index_to_pixel(index);

    let current_face = index / (CUBEMAP_WIDTH * CUBEMAP_WIDTH);

    const MAX_COORD: usize = CUBEMAP_WIDTH - 1;

    //tuple is (neighbouring face, whether it needs to be flipped)
    let left_face;
    let right_face;
    let up_face;
    let down_face;
    match current_face {
        POS_X_FACE => {
            //positive x
            left_face = POS_Z_FACE;
            right_face = NEG_Z_FACE;
            up_face = NEG_Y_FACE;
            down_face = POS_Y_FACE;
        }
        NEG_X_FACE => {
            //negative x
            left_face = NEG_Z_FACE;
            right_face = POS_Z_FACE;
            up_face = NEG_Y_FACE;
            down_face = POS_Y_FACE;
        }
        POS_Y_FACE => {
            //positive y
            left_face = NEG_X_FACE;
            right_face = POS_X_FACE;
            up_face = POS_Z_FACE;
            down_face = NEG_Z_FACE;
        }
        NEG_Y_FACE => {
            //negative y
            left_face = NEG_X_FACE;
            right_face = POS_X_FACE;
            up_face = NEG_Z_FACE;
            down_face = POS_Z_FACE;
        }
        POS_Z_FACE => {
            //positive z
            left_face = NEG_X_FACE;
            right_face = POS_X_FACE;
            up_face = NEG_Y_FACE;
            down_face = POS_Y_FACE;
        }
        NEG_Z_FACE => {
            //negative z
            left_face = POS_X_FACE;
            right_face = NEG_X_FACE;
            up_face = NEG_Y_FACE;
            down_face = POS_Y_FACE;
        }
        _ => {
            panic!("Too many faces in cubemap");
        }
    }
    let mut out = Vec::with_capacity(4);

    if x == 0 {
        match (current_face, left_face) {
            (POS_X_FACE, POS_Z_FACE) => out.push(pixel_to_index(MAX_COORD, y, left_face)),
            (NEG_X_FACE, NEG_Z_FACE) => out.push(pixel_to_index(MAX_COORD, y, left_face)),
            (POS_Y_FACE, NEG_X_FACE) => {
                out.push(pixel_to_index(MAX_COORD - y, MAX_COORD, left_face))
            }
            (NEG_Y_FACE, NEG_X_FACE) => out.push(pixel_to_index(y, 0, left_face)),
            (POS_Z_FACE, NEG_X_FACE) => out.push(pixel_to_index(MAX_COORD, y, left_face)),
            (NEG_Z_FACE, POS_X_FACE) => out.push(pixel_to_index(MAX_COORD, y, left_face)),
            _ => panic!("invalid case"),
        }
    } else {
        out.push(pixel_to_index(x - 1, y, current_face));
    }

    if x == MAX_COORD {
        match (current_face, right_face) {
            (POS_X_FACE, NEG_Z_FACE) => out.push(pixel_to_index(0, y, right_face)),
            (NEG_X_FACE, POS_Z_FACE) => out.push(pixel_to_index(0, y, right_face)),
            (POS_Y_FACE, POS_X_FACE) => out.push(pixel_to_index(y, MAX_COORD, right_face)),
            (NEG_Y_FACE, POS_X_FACE) => out.push(pixel_to_index(MAX_COORD - y, 0, right_face)),
            (POS_Z_FACE, POS_X_FACE) => out.push(pixel_to_index(0, y, right_face)),
            (NEG_Z_FACE, NEG_X_FACE) => out.push(pixel_to_index(0, y, right_face)),
            _ => panic!("invalid case"),
        }
    } else {
        out.push(pixel_to_index(x + 1, y, current_face));
    }

    if y == 0 {
        match (current_face, up_face) {
            (POS_X_FACE, NEG_Y_FACE) => out.push(pixel_to_index(MAX_COORD, MAX_COORD - x, up_face)),
            (NEG_X_FACE, NEG_Y_FACE) => out.push(pixel_to_index(0, x, up_face)),
            (POS_Y_FACE, POS_Z_FACE) => out.push(pixel_to_index(x, MAX_COORD, up_face)),
            (NEG_Y_FACE, NEG_Z_FACE) => out.push(pixel_to_index(MAX_COORD - x, 0, up_face)),
            (POS_Z_FACE, NEG_Y_FACE) => out.push(pixel_to_index(x, MAX_COORD, up_face)),
            (NEG_Z_FACE, NEG_Y_FACE) => out.push(pixel_to_index(MAX_COORD - x, 0, up_face)),
            _ => panic!("Invalid case"),
        }
    } else {
        out.push(pixel_to_index(x, y - 1, current_face));
    }

    if y == MAX_COORD {
        match (current_face, down_face) {
            (POS_X_FACE, POS_Y_FACE) => out.push(pixel_to_index(MAX_COORD, x, down_face)),
            (NEG_X_FACE, POS_Y_FACE) => out.push(pixel_to_index(0, MAX_COORD - x, down_face)),
            (POS_Y_FACE, NEG_Z_FACE) => {
                out.push(pixel_to_index(MAX_COORD - x, MAX_COORD, down_face))
            }
            (NEG_Y_FACE, POS_Z_FACE) => out.push(pixel_to_index(x, 0, down_face)),
            (POS_Z_FACE, POS_Y_FACE) => out.push(pixel_to_index(x, 0, down_face)),
            (NEG_Z_FACE, POS_Y_FACE) => {
                out.push(pixel_to_index(MAX_COORD - x, MAX_COORD, down_face))
            }
            _ => panic!("Invalid case"),
        }
    } else {
        out.push(pixel_to_index(x, y + 1, current_face));
    }

    out.into_boxed_slice()
}

//includes diagonals
fn get_full_neighbour_pixels(index: usize) -> Box<[usize]> {
    let neighbours = get_neighbour_pixels(index);
    let out_neighbours = neighbours.to_vec();
    // let left = neighbours[0];
    // out_neighbours.push(get_neighbour_pixels(left)[2]);
    // out_neighbours.push(get_neighbour_pixels(left)[3]);

    // let right = neighbours[1];
    // out_neighbours.push(get_neighbour_pixels(right)[2]);
    // out_neighbours.push(get_neighbour_pixels(right)[3]);

    out_neighbours.into_boxed_slice()
}

struct LocalClient {
    socket: UdpSocket,
    state: ClientState,
    current_tick: GameTick,
    player_key: Option<PlayerKey>,
    player_model_index: usize,
    start_time: Instant,
}

impl LocalClient {
    fn process_packets(&mut self, game: &mut ClientGame, vulkan_data: &mut VulkanData) {
        let mut buffer = [0; 1024];
        while let Ok(num_bytes) = self.socket.recv(&mut buffer) {
            let unprocessed_datagram = &mut buffer[..num_bytes];

            match ServerToClientPacket::from_bytes(unprocessed_datagram) {
                None => {
                    println!("Invalid packet received from server")
                }
                Some(packet) => match (self.state, packet) {
                    (
                        ClientState::ConnectionAwaiting,
                        ServerToClientPacket::RequestAccepted(game_tick, player_key, num_players),
                    ) => {
                        self.state = ClientState::Connected;
                        self.current_tick = game_tick;
                        self.player_key = Some(player_key);
                        game.camera.targeted_player = Some(player_key);
                        println!("Client connected");

                        // let render_object_index = vulkan_data.objects.len();
                        // vulkan_data
                        //     .objects
                        //     .push(vulkan_data.objects[self.player_model_index].clone());
                        // let mut local_player = ClientPlayer::new(render_object_index, &vulkan_data);
                        // local_player.input_buffer = 
                        // game.players.push(local_player);
                        let num_new_players = (num_players as i64 - game.players.len() as i64).max(0) as usize;
                        for _ in 0..num_new_players {
                            let render_object_index = vulkan_data.objects.len();
                            vulkan_data
                                .objects
                                .push(vulkan_data.objects[self.player_model_index].clone());
                                game.players.push(ClientPlayer::new(render_object_index, &vulkan_data));
                        }
                        game.players[player_key].input_buffer = Some(NetworkBuffer::new(||Inputs::new()));
                    }
                    (
                        ClientState::Connected,
                        ServerToClientPacket::PlayerUpdate {
                            key,
                            player_state,
                            last_client_input,
                        },
                    ) => {
                        // dbg!(key);
                        let server_player_count = key.get() + 1;
                        if server_player_count > game.players.len(){
                            let num_new_players = (server_player_count as i64 - game.players.len() as i64).max(0) as usize;
                            for _ in 0..num_new_players {
                                let render_object_index = vulkan_data.objects.len();
                                vulkan_data
                                    .objects
                                    .push(vulkan_data.objects[self.player_model_index].clone());
                                    game.players.push(ClientPlayer::new(render_object_index, &vulkan_data));
                            }
                        }
                        let player = game.players.get_mut(key).unwrap();
                        player.last_server_player_state = player_state;
                        player.last_client_input_from_server = last_client_input;

                    }
                    (ClientState::Connected, ServerToClientPacket::RequestAccepted(..)) => {
                        println!("Duplicate Accept packet")
                    }
                    (ClientState::Connected, ServerToClientPacket::RequestDenied) => {
                        println!("Client already connect and received denial")
                    }
                    (ClientState::ConnectionAwaiting | ClientState::Disconnected | ClientState::TimedOut, _) => {
                        println!("Unknown state/packet combo")
                    }                    
                },
            }
        }
    }

    fn send_packets(&self, game: &ClientGame){
        match self.state {
            ClientState::Disconnected => {}
            ClientState::ConnectionAwaiting => {}
            ClientState::Connected => {
                self.socket
                    .send(
                        &ClientToServerPacket::Input {
                            inputs: game.inputs,
                            tick_sent: self.start_time.elapsed().as_secs_f64(),
                        }
                        .to_bytes(),
                    )
                    .unwrap();
            }
            ClientState::TimedOut => {},
            
        }
    }
}

fn main() {
    // let mut frametime: std::time::Instant = std::time::Instant::now();
    // let mut time_since_last_frame: f64 = 0.0; //seconds

    // let mut network_frametime: std::time::Instant = std::time::Instant::now();
    // let mut time_since_last_network_tick: f64 = 0.0; //seconds
    let mut frame_start = std::time::Instant::now();
    let mut network_start = std::time::Instant::now();

    let mut frametimes = [0.0; FRAME_SAMPLES];
    let mut frametimes_start = 0;
    let mut last_tick = std::time::Instant::now();

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
    vulkan_data.init_vulkan(&window, |_ | {});
    println!("Vulkan Data initialized");

    let mut game = ClientGame::new();
    println!("Game created");
    let socket = UdpSocket::bind("0.0.0.0:0").unwrap();
    socket.connect("127.0.0.1:2022").unwrap();
    socket.set_nonblocking(true).unwrap();
    let mut client = LocalClient {
        socket,
        state: ClientState::Disconnected,
        current_tick: GameTick::new(0),
        player_key: None,
        player_model_index: vulkan_data.load_folder(PathBuf::from("models/person")),
        start_time: Instant::now(),
    };
    let mut sword_index = vulkan_data.load_folder(PathBuf::from("models/sword"));
    game.players.push(ClientPlayer::new(client.player_model_index, &vulkan_data));
    vulkan_data.load_folder(PathBuf::from("models/test_ball"));
    client
        .socket
        .send(
            &ClientToServerPacket::RequestConnect {
                username: "daniel".to_string(),
            }
            .to_bytes(),
        )
        .unwrap();
    client.state = ClientState::ConnectionAwaiting;
    println!("attempting to connect to server");
    vulkan_data.update_vertex_and_index_buffers();

    let mut texture_version = 0;

    let mut gilrs = Gilrs::new().unwrap();

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

        // println!("egui texture version: {:?}", ctx.texture().version);

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
        while let Some(gilrs::Event { id, event, time }) = gilrs.next_event() {
            match event {
                gilrs::EventType::AxisChanged(axis, value, _) => {
                    let value = value as f64;
                    match axis {
                        gilrs::Axis::LeftStickX => {
                            game.inputs.left_stick.x = value;
                        }
                        gilrs::Axis::LeftStickY => {
                            game.inputs.left_stick.y = value;
                        }
                        gilrs::Axis::RightStickX => {
                            game.inputs.right_stick.x = value;
                        }
                        gilrs::Axis::RightStickY => {
                            game.inputs.right_stick.y = value;
                        }
                        _ => {}
                    }
                }
                gilrs::EventType::ButtonReleased(button, _) => {
                    match button{
                        gilrs::Button::South => game.inputs.jump = false,
                        _ => {},
                    }
                }
                gilrs::EventType::ButtonPressed(button, _) => {
                    match button{
                        gilrs::Button::South => game.inputs.jump = true,
                        _ => {},
                    }
                }
                _ => {}
            }
        }
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
                            // println!("Position: {:?}", position);
                        }
                        WindowEvent::MouseInput { .. } => {}
                        WindowEvent::MouseWheel { .. } => {}
                        WindowEvent::KeyboardInput { .. } => {}
                        _ => {}
                    }
                }
            }
            Event::MainEventsCleared => {
                //app update code
                // let elapsed_time = frametime.elapsed().as_secs_f64();
                // let network_elapsed_time = network_frametime.elapsed().as_secs_f64();
                // frametime = std::time::Instant::now();
                // network_frametime = std::time::Instant::now();

                while network_start.elapsed().as_secs_f64() > 1.0 / NETWORK_TICKRATE {
                    network_start += Duration::from_secs_f64(1.0 / NETWORK_TICKRATE);
                    client.send_packets(&mut game);
                    client.current_tick += 1usize;
                    // dbg!(client.current_tick);
                }

                if frame_start.elapsed().as_secs_f64() > 1.0 / FRAMERATE_TARGET {
                    frame_start += Duration::from_secs_f64(1.0 / FRAMERATE_TARGET);
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
                    
                    client.process_packets(&mut game, &mut vulkan_data);
                    if let Some(player_key) = client.player_key{
                        game.inputs.camera_direction = game.camera.get_direction(game.players[player_key].game_object.position);
                        game.players[player_key].input_buffer.as_mut().unwrap().add_value(game.inputs);
                    }
                    game.camera.longitude += game.inputs.right_stick.x * delta_time;
                    game.camera.latitude += game.inputs.right_stick.y * delta_time;

                    if let Some(player_key) = client.player_key{
                        game.players[player_key].game_object.animation_handler.as_mut().unwrap().process(delta_time * game.inputs.left_stick.y.abs())
                    }
                    for player in &mut game.players {
                        // if let Some(animation_handler) = &mut player.game_object.animation_handler {
                        //     animation_handler.process(delta_time * game.inputs.left_stick.y.abs());
                        // }
                        player.update_gameobject(&client.start_time);
                    }

                    update_renderer(&game, &mut vulkan_data,sword_index);
                    vulkan_data.transfer_data_to_gpu();

                    let _draw_frame_result = vulkan_data.draw_frame();
                }
            }
            _ => {}
        }
    })
}

fn update_renderer(game: &ClientGame, vulkan_data: &mut VulkanData, sword: usize) {
    let clip = Matrix4::<f64>::identity();

    let projection = vulkan_data.get_projection(1.0);
    let projection_matrix = clip * projection;

    
    let target_position = match game.camera.targeted_player {
        Some(player_key) => game.players[player_key].game_object.position,
        None => Vector3::zeros(),
    };
    let view_matrix = game.camera.get_view_matrix(target_position);



    for player in &game.players {
        // dbg!(player.position);
        vulkan_data.objects[player.game_object.render_object_index].model = player.game_object.get_transform().cast();
        vulkan_data.objects[sword].model = vulkan_data.get_bone_matrix(player.game_object.render_object_index, 15284, 12);
        
        if let Some(animation_handler) = &player.game_object.animation_handler {
            vulkan_data.objects[player.game_object.render_object_index].set_animation(
                animation_handler.index,
                animation_handler.progress,
                animation_handler.previous_frame,
                animation_handler.next_frame,
            )
        }
    }
    
    vulkan_data.uniform_buffer_object.view = view_matrix.cast();
    vulkan_data.uniform_buffer_object.proj = projection_matrix.cast();

    vulkan_data.uniform_buffer_object.player_position = Vector3::zeros();

    vulkan_data.uniform_buffer_object.mouse_position = Vector2::new(
        (game.mouse_position.x) as f32,
        (game.mouse_position.y) as f32,
    );
}

fn close_app(vulkan_data: &mut VulkanData, control_flow: &mut ControlFlow) {
    *control_flow = ControlFlow::Exit;
    vulkan_data.cleanup();
}
