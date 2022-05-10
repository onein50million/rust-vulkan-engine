use egui::plot::Value;
use egui::plot::Values;

use egui::Grid;
use egui::Label;

use egui::ScrollArea;

use egui::TopBottomPanel;
use egui::Ui;
use egui::Window;
use egui_winit::winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use egui_winit::winit::event_loop::{ControlFlow, EventLoop};
use egui_winit::winit::window::WindowBuilder;

use nalgebra::Matrix3;

use nalgebra::Point3;
use nalgebra::{Matrix4, Rotation3, Translation3, Vector2, Vector3};

use noise::NoiseFn;
use rust_vulkan_engine::game::client::Game;

use rust_vulkan_engine::network::{ClientState, Packet};
use rust_vulkan_engine::renderer::*;
use rust_vulkan_engine::support;
use rust_vulkan_engine::support::*;
use rust_vulkan_engine::world::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use std::net::UdpSocket;
use std::time::Instant;

use winit::event::{MouseButton, MouseScrollDelta};
//Coordinate system for future reference:
//from starting location
//up: negative y
//down: positive y
//forward: negative z
//backward: positive z
//left: negative x
//right: positive x

const CUBEMAP_WIDTH: usize = CubemapRender::CUBEMAP_WIDTH as usize;

fn province_shift(perlin: &noise::Perlin, point: &Vector3<f64>, w: f64) -> f64 {
    (perlin.get([point.x as f64, point.y as f64, point.z as f64, w]) * 2.0 - 1.0) * 0.01
}

fn index_to_coordinate(index: usize) -> Vector3<f32> {
    const CORRECTION_MATRIX: Matrix3<f64> = Matrix3::new(
        1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, -1.0000000,
        0.0000000,
    );

    let (x, y) = index_to_pixel(index);
    let face = index / (CUBEMAP_WIDTH * CUBEMAP_WIDTH);

    let x = ((x as f64) / CUBEMAP_WIDTH as f64) * 2.0 - 1.0;
    let y = ((y as f64) / CUBEMAP_WIDTH as f64) * 2.0 - 1.0;

    let normal;
    match face {
        0 => {
            normal = Vector3::new(1.0, -y, -x);
        }
        1 => {
            normal = Vector3::new(-1.0, -y, x);
        }
        2 => {
            normal = Vector3::new(x, 1.0, y);
        }
        3 => {
            normal = Vector3::new(x, -1.0, -y);
        }
        4 => {
            normal = Vector3::new(x, -y, 1.0);
        }
        5 => {
            normal = Vector3::new(-x, -y, -1.0);
        }
        _ => {
            panic!("Too many faces in cubemap");
        }
    }

    let secondary_correction_matrix: Matrix4<f64> =
        Matrix4::from(Rotation3::from_euler_angles(-90f64.to_radians(), 0.0, 0.0));
    let normal = CORRECTION_MATRIX * normal.normalize();
    let normal = secondary_correction_matrix
        .transform_point(&Point3::from(normal))
        .coords;
    let normal = normal.component_mul(&Vector3::new(1.0, 1.0, 1.0));
    let point = normal * World::RADIUS;

    point.cast()
}

fn index_to_pixel(index: usize) -> (usize, usize) {
    (
        (index % (CUBEMAP_WIDTH * CUBEMAP_WIDTH)) % CUBEMAP_WIDTH,
        (index % (CUBEMAP_WIDTH * CUBEMAP_WIDTH)) / CUBEMAP_WIDTH,
    )
}

fn pixel_to_index(x: usize, y: usize, face: usize) -> usize {
    let face_offset = face * CUBEMAP_WIDTH * CUBEMAP_WIDTH;
    let y_offset = y * CUBEMAP_WIDTH;
    x + y_offset + face_offset
}

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

#[derive(Debug, Clone)]
struct CategorizedElevation {
    position: Vector3<f32>,
    elevation: f32,
    province_id: Option<usize>,
    is_coastal: bool,
}

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

// fn get_unique_id() -> u64{
//     fastrand::u64(..)
// }

fn add_plot(ui: &mut Ui, values: &[Value], heading: &str, num_samples: usize, start: usize) {
    let values = &values[start..];
    let num_samples = num_samples.min(values.len());
    let chunk_size = values.len() / num_samples;
    let line = egui::plot::Line::new(Values::from_values_iter(values.chunks(chunk_size).map(
        |values| {
            let mut sum = 0.0;
            let mut day_sum = 0.0;
            for value in values {
                sum += value.y;
                day_sum += value.x;
            }
            Value::new(day_sum / values.len() as f64, sum / values.len() as f64)
        },
    )));
    ui.add(Label::new(heading).heading());
    ui.add(
        egui::plot::Plot::new(1)
            .line(line)
            .view_aspect(1.0)
            .allow_zoom(false)
            .include_y(0.0)
            .include_x(values[0].x)
            .allow_drag(false),
    );
}

struct Histories {
    population: Box<[VecDeque<Value>]>,
    prices: Box<[Box<[VecDeque<Value>]>]>,
    last_time_check: Instant,
}
impl Histories {
    fn add_new_tick(&mut self, world: &World, month: usize, print_progress: bool) {
        for (province_index, province) in world.provinces.iter().enumerate() {
            if print_progress && self.last_time_check.elapsed().as_secs_f64() > 1.0 {
                self.last_time_check = std::time::Instant::now();
                println!("Day: {month}");
            }
            if self.population[province_index].len() > 10000 {
                self.population[province_index].pop_back();
            }
            self.population[province_index]
                .push_front(Value::new(month as f64, province.pops.population()));

            for good_index in 0..Good::VARIANT_COUNT {
                if self.prices[province_index][good_index].len() > 10000 {
                    self.prices[province_index][good_index].pop_back();
                }
                self.prices[province_index][good_index]
                    .push_front(Value::new(month as f64, province.market.price[good_index]));
            }
        }
    }
}

fn main() {
    let mut frametime: std::time::Instant = std::time::Instant::now();
    let mut time_since_last_frame: f64 = 0.0; //seconds

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
    vulkan_data.init_vulkan(&window);
    println!("Vulkan Data initialized");

    let planet_mesh = rust_vulkan_engine::planet_gen::get_planet(World::RADIUS as f32);

    let elevations = {
        let elevation_vertices: Box<[ElevationVertex]> = planet_mesh
            .vertices
            .iter()
            .map(|vertex| ElevationVertex {
                position: vertex.position,
                elevation: vertex.elevation,
            })
            .collect();
        let elevation_indices: Box<[u32]> = planet_mesh
            .indices
            .iter()
            .map(|&index| index as u32)
            .collect();
        let mut elevation_cubemap =
            CubemapRender::new(&vulkan_data, &elevation_vertices, &elevation_indices);
        elevation_cubemap.render(&vulkan_data);
        vulkan_data.planet_normal_map = Some(elevation_cubemap.get_normal(&mut vulkan_data));
        vulkan_data.update_descriptor_sets();
        elevation_cubemap.into_image(&vulkan_data)
    };

    let rng = fastrand::Rng::new();
    let mut elevations: Vec<_> = elevations
        .iter()
        .enumerate()
        .map(|(index, &elevation)| CategorizedElevation {
            elevation,
            province_id: if elevation > 0.0 { None } else { Some(0) },
            position: index_to_coordinate(index),
            is_coastal: false,
        })
        .collect();

    println!("Starting to separate provinces");

    let mut used_ids = HashSet::new();
    used_ids.insert(0);
    loop {
        let current_id = rng.usize(1..);
        assert_eq!(used_ids.contains(&current_id), false);
        used_ids.insert(current_id);
        let mut num_set = 0;
        match elevations.iter().position(|a| a.province_id.is_none()) {
            Some(index) => {
                let mut queue = VecDeque::new();
                queue.push_back(index);
                while queue.len() > 0 && num_set < (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) / 10000 {
                    let n = queue.pop_front().unwrap();
                    if elevations[n].province_id.is_none() && elevations[n].elevation > 0.0 {
                        elevations[n].province_id = Some(current_id);
                        num_set += 1;
                        queue.extend(get_neighbour_pixels(n).iter());
                    }
                }
            }
            None => break,
        }
    }

    println!("Finished separating provinces");

    let mut bad_provinces = HashSet::new();
    for id in used_ids {
        let mut num_pixels = 0;
        let province = elevations.iter().enumerate().filter_map(|(i, a)| {
            if a.province_id.unwrap() == id {
                num_pixels += 1;
                Some(i)
            } else {
                None
            }
        });
        let neighbours: Box<[_]> = province.map(|a| get_neighbour_pixels(a)).collect();
        let num_neighbours = neighbours
            .into_iter()
            .map(|a| a.into_iter())
            .flatten()
            .filter(|&&a| elevations[a].province_id.unwrap() != id)
            .count();
        let neighbour_ratio = (num_neighbours as f64) / (num_pixels as f64);
        if !(num_pixels > 5 || neighbour_ratio < 10.0) {
            bad_provinces.insert(id);
        }
    }

    let mut found_borders = HashSet::new();
    for i in 0..elevations.len() {
        if elevations[i].elevation < 0.0
            || bad_provinces.contains(&elevations[i].province_id.unwrap())
        {
            continue;
        }
        let neighbours = get_neighbour_pixels(i);

        for &neighbour in neighbours.iter() {
            if elevations[neighbour].elevation > 0.0
                && elevations[i].province_id != elevations[neighbour].province_id
            {
                found_borders.insert(if i > neighbour {
                    (i, neighbour)
                } else {
                    (neighbour, i)
                });
            }
        }

        if neighbours
            .iter()
            .any(|&neighbour_index| elevations[neighbour_index].elevation < 0.0)
        {
            elevations[i].is_coastal = true;
        }
    }

    let mut province_map = HashMap::new();
    let mut province_points = vec![];
    let perlin = noise::Perlin::new();

    for &pair in found_borders.iter() {
        let midpoint = elevations[pair.0]
            .position
            .lerp(&elevations[pair.1].position, 0.5);
        let noise_point = midpoint.normalize().cast() * 30.0;
        let noise_shift = if true || elevations[pair.0].is_coastal || elevations[pair.1].is_coastal
        {
            Vector3::zeros()
        } else {
            Vector3::new(
                province_shift(&perlin, &noise_point, 0.0),
                province_shift(&perlin, &noise_point, 100.0),
                province_shift(&perlin, &noise_point, 1000.0),
            )
        };
        let shifted_position: Vector3<f32> =
            ((midpoint.normalize().cast() + noise_shift).normalize() * World::RADIUS).cast()
                * 1.001;

        //point a
        let entry = province_map
            .entry(elevations[pair.0].province_id.unwrap())
            .or_insert(vec![]);
        entry.push(province_points.len());

        //point b
        let entry = province_map
            .entry(elevations[pair.1].province_id.unwrap())
            .or_insert(vec![]);
        entry.push(province_points.len());
        province_points.push(shifted_position);
    }

    let province_indices: Box<[_]> = province_map.into_values().collect();
    let world = World::new(&province_points, &province_indices);

    if let Some(line_data) = &mut vulkan_data.line_data {
        for point in &world.points {
            line_data.add_point(*point);
        }
        for province in world.provinces.iter() {
            for chunk in province.point_indices.chunks(2) {
                if chunk.len() < 2 {
                    continue;
                }
                let first_point = chunk[0];
                let second_point = chunk[1];
                line_data.connect_points(first_point, second_point);
            }
        }
    }

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
    vulkan_data.load_folder("models/planet/data".parse().unwrap());

    // let planet_render_object_index = VulkanData::load_vertices_and_indices(
    //     &mut vulkan_data,
    //     world.get_vertices(),
    //     world.get_indices()
    // );
    vulkan_data.update_vertex_and_index_buffers();

    let mut game = Game::new(planet_render_object_index, world);
    println!("Game created");
    let socket = UdpSocket::bind("0.0.0.0:0").unwrap();
    socket.connect("127.0.0.1:2022").unwrap();
    socket.set_nonblocking(true).unwrap();
    let mut client = Client {
        socket,
        state: ClientState::Disconnected,
    };

    let last_time_check = std::time::Instant::now();
    //Do some simulation to see if state stabilizes
    let days = 12 * 100;
    let mut histories = Histories {
        population: vec![VecDeque::with_capacity(days); game.world.provinces.len()]
            .into_boxed_slice(),
        prices: vec![
            vec![VecDeque::with_capacity(days); Good::VARIANT_COUNT].into_boxed_slice();
            game.world.provinces.len()
        ]
        .into_boxed_slice(),
        last_time_check,
    };
    for day in 0..days {
        histories.add_new_tick(&game.world, day, true);
        game.world.process(1.0 / 365.0);
    }

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

    let mut current_day = days;
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

        TopBottomPanel::top("get_unique_id()").show(&ctx, |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut population_graph_window_open, "Population Plot");
                ui.checkbox(&mut pop_table_open, "Pops Table");
                ui.checkbox(&mut price_graph_window_open, "Price Plots");
            });
        });

        if population_graph_window_open {
            Window::new("Population Plot").show(&ctx, |ui| match game.selected_province {
                Some(selected_province) => {
                    let values = histories.population[selected_province].as_slices();
                    add_plot(
                        ui,
                        &[values.0, values.1].concat(),
                        "Total Population",
                        1000,
                        0,
                    );
                }
                None => {}
            });
        }
        if price_graph_window_open {
            Window::new("Price Plots").show(&ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| match game.selected_province {
                    Some(selected_province) => {
                        for good_index in 0..Good::VARIANT_COUNT {
                            let good = Good::try_from(good_index).unwrap();

                            let values =
                                histories.prices[selected_province][good_index].as_slices();
                            add_plot(
                                ui,
                                &[values.0, values.1].concat(),
                                &format!("{:?} Price", good),
                                1000,
                                0,
                            );
                        }
                    }
                    None => {}
                });
            });
        }
        if pop_table_open {
            Window::new("Pop Table").show(&ctx, |ui| {
                Grid::new(0).striped(true).show(ui, |ui| {
                    ui.label("Culture");
                    ui.label("Industry");
                    ui.label("Population");
                    ui.label("Money");
                    for good_index in 0..Good::VARIANT_COUNT {
                        ui.label(format!("Owned {:?}", Good::try_from(good_index).unwrap()));
                    }
                    ui.end_row();
                    for (i, slice) in game.world.provinces[0].pops.pop_slices.iter().enumerate() {
                        let culture = Culture::try_from(i / Industry::VARIANT_COUNT).unwrap();
                        let industry = Industry::try_from(i % Industry::VARIANT_COUNT).unwrap();
                        ui.label(format!("{:?}", culture));
                        ui.label(format!("{:?}", industry));
                        ui.label(format!("{:.0}", slice.population));
                        ui.label(format!("{:.2}", slice.money));
                        for good_index in 0..Good::VARIANT_COUNT {
                            ui.label(format!("{:.2}", slice.owned_goods[good_index]));
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
                            game.mouse_position.x =
                                -((position.x / window.inner_size().width as f64) * 2.0 - 1.0);
                            game.mouse_position.y =
                                -((position.y / window.inner_size().height as f64) * 2.0 - 1.0);
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
                                    * 1.1f64.powf(-vertical_lines as f64))
                                .clamp(0.00001, 2.0);
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

                    histories.add_new_tick(&game.world, current_day, false);
                    game.process(1.0 / 365.0, &vulkan_data.get_projection(game.inputs.zoom));

                    if let Some(selected_province) = game.selected_province {
                        let province_indices =
                            &game.world.provinces[selected_province].point_indices;
                        if let Some(line_data) = &mut vulkan_data.line_data {
                            line_data.select_points(&province_indices);
                        }
                    }

                    current_day += 1;

                    update_renderer(&game, &mut vulkan_data, current_day as f32 / 365.0);
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

fn update_renderer(game: &Game, vulkan_data: &mut VulkanData, current_year: f32) {
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

    vulkan_data.uniform_buffer_object.map_mode = game.inputs.map_mode as u32;

    match game.inputs.map_mode {
        map_modes::SATELITE => {
            vulkan_data.uniform_buffer_object.lights[0].color.w = 1.0;
            vulkan_data.uniform_buffer_object.lights[1].color.w = 0.0;
            vulkan_data.uniform_buffer_object.cubemap_index = 0;
        }
        map_modes::PAPER => {
            vulkan_data.uniform_buffer_object.lights[0].color.w = 0.0;
            vulkan_data.uniform_buffer_object.lights[1].color.w = 1.0;
            vulkan_data.uniform_buffer_object.cubemap_index = 1;
        }
        _ => {}
    }
    vulkan_data.uniform_buffer_object.time = current_year;
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
