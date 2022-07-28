use egui::plot::Value;
use egui::plot::Values;
use egui::ComboBox;
use egui::Id;
use egui::Slider;

use egui::Grid;
use egui::Label;

use egui::ScrollArea;

use egui::TopBottomPanel;
use egui::Ui;
use egui::Window;
use egui_winit::winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use egui_winit::winit::event_loop::{ControlFlow, EventLoop};
use egui_winit::winit::window::WindowBuilder;

use nalgebra::ComplexField;
use nalgebra::{Matrix4, Rotation3, Translation3, Vector2, Vector3};

use noise::NoiseFn;
use rust_vulkan_engine::game::client::Game;

use rust_vulkan_engine::network::{ClientState, Packet};
use rust_vulkan_engine::planet_gen::get_aridity;
use rust_vulkan_engine::planet_gen::get_countries;
use rust_vulkan_engine::planet_gen::get_elevations;
use rust_vulkan_engine::planet_gen::get_feb_temps;
use rust_vulkan_engine::planet_gen::get_july_temps;
use rust_vulkan_engine::planet_gen::get_languages;
use rust_vulkan_engine::planet_gen::get_populations;
use rust_vulkan_engine::planet_gen::get_provinces;
use rust_vulkan_engine::planet_gen::get_water;
use rust_vulkan_engine::renderer::*;
use rust_vulkan_engine::support;
use rust_vulkan_engine::support::*;
use rust_vulkan_engine::world::ideology::Beliefs;
use rust_vulkan_engine::world::organization::OrganizationKey;
use rust_vulkan_engine::world::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use std::fmt::Debug;
use std::net::UdpSocket;
use std::ops::Add;

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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ProvinceKeyOptions {
    None,
    Sea,
    Some(ProvinceKey),
}

#[derive(Debug, Clone)]
struct CategorizedElevation {
    position: Vector3<f32>,
    elevation: f32,
    province_key: ProvinceKeyOptions,
    is_coastal: bool,
    aridity: f64,
    population: f64,
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

fn add_plot(ui: &mut Ui, values: &[(f64, f64)], heading: &str, num_samples: usize, start: usize) {
    let values = &values[start..];
    let num_samples = num_samples.min(values.len());
    let chunk_size = values.len() / num_samples;
    let line = egui::plot::Line::new(Values::from_values_iter(values.chunks(chunk_size).map(
        |values| {
            let mut sum = 0.0;
            let mut day_sum = 0.0;
            for value in values {
                sum += value.1;
                day_sum += value.0;
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
            .include_x(values[0].0)
            .allow_drag(false),
    );
}

fn double_line_plot(
    ui: &mut Ui,
    values_a: &[(f64, f64)],
    values_b: &[(f64, f64)],
    heading: &str,
    num_samples: usize,
    start: usize,
) {
    let line_a = {
        let values = &values_a[start..];
        let num_samples = num_samples.min(values.len());
        let chunk_size = values.len() / num_samples;
        egui::plot::Line::new(Values::from_values_iter(values.chunks(chunk_size).map(
            |values| {
                let mut sum = 0.0;
                let mut day_sum = 0.0;
                for value in values {
                    sum += value.1;
                    day_sum += value.0;
                }
                Value::new(day_sum / values.len() as f64, sum / values.len() as f64)
            },
        )))
    };
    let line_b = {
        let values = &values_b[start..];
        let num_samples = num_samples.min(values.len());
        let chunk_size = values.len() / num_samples;
        egui::plot::Line::new(Values::from_values_iter(values.chunks(chunk_size).map(
            |values| {
                let mut sum = 0.0;
                let mut day_sum = 0.0;
                for value in values {
                    sum += value.1;
                    day_sum += value.0;
                }
                Value::new(day_sum / values.len() as f64, sum / values.len() as f64)
            },
        )))
    };
    ui.add(Label::new(heading).heading());
    ui.add(
        egui::plot::Plot::new(1)
            .line(line_a)
            .line(line_b)
            .view_aspect(1.0)
            .allow_zoom(false)
            .include_y(0.0)
            .include_x(values_a.last().unwrap().0.max(values_b.last().unwrap().0))
            .allow_drag(false),
    );
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

    let elevations = get_elevations();
    let aridity = get_aridity();
    let feb_temps = get_feb_temps();
    let july_temps = get_july_temps();
    let population = get_populations();
    let (nations_map, nation_names_and_defintions) = get_countries();
    // for nation in nations_map.iter(){
    //     if nation.is_some(){
    //         dbg!(nation);
    //     }
    // }
    let (provinces_map, province_vertices, province_indices, province_names) = get_provinces();
    println!("provinces gotten");
    let water_map = get_water();
    let (language_map, color_to_language_name) = get_languages();

    let pops_underwater = population
        .iter()
        .enumerate()
        .filter(|&(i, _)| water_map[i] < -1000.0)
        .map(|(_, &pop)| pop)
        .sum::<f32>();

    dbg!(pops_underwater);
    let planet_mesh = rust_vulkan_engine::planet_gen::get_planet(World::RADIUS as f32);

    let mut categorized_elevations: Vec<_> = elevations
        .iter()
        .enumerate()
        .map(|(index, &elevation)| CategorizedElevation {
            elevation,
            province_key: if elevation > 0.0 {
                ProvinceKeyOptions::None
            } else {
                ProvinceKeyOptions::Sea
            },
            position: index_to_coordinate(index),
            is_coastal: false,
            population: population[index] as f64,
            aridity: aridity[index] as f64,
        })
        .collect();

    let world_file = "world_file.json";
    let world = match &mut std::fs::File::open(world_file) {
        Ok(file) => {
            println!("Loading world file");
            World::load(file)
        }
        Err(_) => {
            let mut num_provinces = 0;
            for (i, elevation_value) in categorized_elevations.iter_mut().enumerate() {
                elevation_value.province_key = match provinces_map[i] {
                    None => ProvinceKeyOptions::Sea,
                    Some(key) => {
                        num_provinces = num_provinces.max(key);
                        ProvinceKeyOptions::Some(ProvinceKey(key as usize))
                    }
                }
            }
            let num_provinces = (num_provinces + 1) as usize;

            let province_data = {
                let mut province_data =
                    ProvinceMap(vec![ProvinceData::new(); num_provinces].into_boxed_slice());

                let perlin = noise::Perlin::new();
                for (i, categorized_elevation) in categorized_elevations.iter().enumerate() {
                    let aridity = categorized_elevation.aridity;
                    let feb_temp = feb_temps[i] as f64;
                    let july_temp = july_temps[i] as f64;
                    let ore_sample_point = [
                        categorized_elevation.position.normalize().cast().x,
                        categorized_elevation.position.normalize().cast().y,
                        categorized_elevation.position.normalize().cast().z,
                    ];
                    let ore = perlin
                        .get(ore_sample_point)
                        .add(0.5)
                        .clamp(0.0, 1.0)
                        .powf(16.0)
                        * 1.0;
                    let population = categorized_elevation.population;
                    if let ProvinceKeyOptions::Some(key) = categorized_elevation.province_key {
                        province_data[key].add_sample(
                            &province_names[key],
                            categorized_elevation.elevation as f64,
                            aridity,
                            feb_temp,
                            july_temp,
                            ore,
                            population,
                            nations_map[i].map(|n| n as usize),
                            language_map[i],
                        );
                    }
                }
                province_data
            };

            let mut province_indices_vec = vec![vec![]; num_provinces];
            for i in 0..num_provinces {
                province_indices_vec[i].extend(province_indices[ProvinceKey(i)].iter());
            }

            let world = World::new(
                &province_vertices,
                &ProvinceMap(province_indices_vec.into_boxed_slice()),
                &province_data,
                nation_names_and_defintions,
                color_to_language_name,
            );

            world.save(world_file);
            world
        }
    };
    let mut vulkan_data = VulkanData::new();
    println!("Vulkan Data created");
    vulkan_data.init_vulkan(&window, |vulkan_data| {
        //Planet data maps
        let elevations_cubemap = vulkan_data.get_cubemap_from_slice(&elevations);
        vulkan_data
            .planet_textures
            .push(vulkan_data.get_normal(&elevations_cubemap));
        vulkan_data.planet_textures.push(elevations_cubemap);
        vulkan_data
            .planet_textures
            .push(vulkan_data.get_cubemap_from_slice(&aridity));
        vulkan_data
            .planet_textures
            .push(vulkan_data.get_cubemap_from_slice(&feb_temps));
        vulkan_data
            .planet_textures
            .push(vulkan_data.get_cubemap_from_slice(&july_temps));
        vulkan_data
            .planet_textures
            .push(vulkan_data.get_cubemap_from_slice(&water_map));
        vulkan_data.create_province_data(province_vertices.iter().map(|&v|v), province_indices.0.iter().flat_map(|p|p.iter().map(|&i|i)))
    });
    println!("Vulkan Data initialized");


    // if let Some(line_data) = &mut vulkan_data.line_data {
    //     for point in &world.points {
    //         line_data.add_point(*point);
    //     }
    //     for province in world.provinces.0.iter() {
    //         for chunk in province.point_indices.chunks(2) {
    //             if chunk.len() < 2 {
    //                 continue;
    //             }
    //             let first_point = chunk[0];
    //             let second_point = chunk[1];
    //             line_data.connect_points(first_point, second_point);
    //         }
    //     }
    // }

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
    vulkan_data.load_folder("models/planet/desert3".parse().unwrap());
    vulkan_data.load_folder("models/planet/mountain3".parse().unwrap());
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

    let mut population_graph_window_open = false;
    let mut price_graph_window_open = false;
    let mut supply_demand_window_open = false;
    let mut pop_table_open = false;
    let mut pop_ideology_info_open = false;
    let mut selected_slice = 0;
    let mut province_info_open = false;
    let mut global_market_open = false;
    let mut global_supply_demand_open = false;
    let mut organizations_list_open = false;
    let mut organization_filter = String::new();
    let mut organization_info_open = false;
    let mut console_open = false;
    let mut console_input = String::new();
    let mut console_output = String::new();
    let mut game_speed = 0u16;

    let mut slider_value = 0.0;
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

        const DAYS_IN_MONTHS: &[usize] = &[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

        TopBottomPanel::top("top_panel").show(&ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("Speed: {:}", game_speed));
                let year = game.world.current_year.floor() as usize;
                let month = ((game.world.current_year % 1.0) * 12.0).floor() as usize;
                let day = ((game.world.current_year % 1.0) * 365.0).floor() as usize
                    % DAYS_IN_MONTHS[month];
                ui.label(format!("Date: {:}/{:}/{:}", year + 1, month + 1, day + 1));

                let hour = (game.world.current_year * 365.0 * 24.0) % 24.0;
                let minute = (hour * 60.0) % 60.0;
                ui.label(format!("Time: {:}:{:}", hour as usize, minute as usize));
                ui.checkbox(&mut population_graph_window_open, "Population Plot");
                ui.checkbox(&mut pop_table_open, "Pops Table");
                ui.checkbox(&mut pop_ideology_info_open, "Pops Ideology");
                ui.checkbox(&mut price_graph_window_open, "Price Plots");
                ui.checkbox(&mut supply_demand_window_open, "Supply/Demand");
                ui.checkbox(&mut province_info_open, "Province Info");
                ui.checkbox(&mut global_market_open, "Global Market");
                ui.checkbox(&mut global_supply_demand_open, "Global Supply/Demand");
                ui.checkbox(&mut organizations_list_open, "Organizations");
                ui.checkbox(&mut organization_info_open, "Org Info");
                ui.checkbox(&mut console_open, "Console");
            });
        });

        TopBottomPanel::bottom("bottom_panel").show(&ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&game.world.organizations[game.world.player_organization].name);
                ui.label(format!(
                    "Money: {:}",
                    big_number_format(
                        game.world.organizations[game.world.player_organization].money
                    )
                ));
                for good_index in 0..Good::VARIANT_COUNT {
                    let good = Good::try_from(good_index).unwrap();
                    ui.label(format!(
                        "{:?}: {:.1}",
                        good,
                        &game.world.organizations[game.world.player_organization].owned_goods
                            [good_index]
                    ));
                }
            });
            ui.horizontal(|ui| {
                match (&game.world.selected_province, &game.world.targeted_province) {
                    (Some(source), Some(dest)) => {
                        ui.add(Slider::new(&mut slider_value, (0.0)..=(1.0)));
                        if ui.button("Transfer troops").clicked() {
                            game.world.organizations[game.world.player_organization]
                                .transfer_troops(*source, *dest, slider_value);
                        }
                    }
                    (_, _) => {}
                }
                if let Some(selected_org) = game.world.selected_organization {
                    let relations = game
                        .world
                        .relations
                        .get_relations_mut(game.world.player_organization, selected_org);
                    let button_string = if relations.at_war {
                        "Make peace with"
                    } else {
                        "Declare war on"
                    };
                    if ui
                        .button(format!(
                            "{:} {:}",
                            button_string, &game.world.organizations[selected_org].name
                        ))
                        .clicked()
                    {
                        relations.at_war = !relations.at_war;
                    }
                }
            })
        });

        if population_graph_window_open {
            Window::new("Population Plot").show(&ctx, |ui| match game.world.selected_province {
                Some(selected_province) => {
                    let values = game.world.histories.population[selected_province].as_slices();
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
                ScrollArea::vertical().show(ui, |ui| match game.world.selected_province {
                    Some(selected_province) => {
                        for good_index in 0..Good::VARIANT_COUNT {
                            let good = Good::try_from(good_index).unwrap();

                            let values = game.world.histories.prices[selected_province][good_index]
                                .as_slices();
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
        if supply_demand_window_open {
            Window::new("Supply/Demand Plots").show(&ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| match game.world.selected_province {
                    Some(selected_province) => {
                        for good_index in 0..Good::VARIANT_COUNT {
                            let good = Good::try_from(good_index).unwrap();

                            let values_a = game.world.histories.supply[selected_province]
                                [good_index]
                                .as_slices();
                            let values_b = game.world.histories.demand[selected_province]
                                [good_index]
                                .as_slices();
                            double_line_plot(
                                ui,
                                &[values_a.0, values_a.1].concat(),
                                &[values_b.0, values_b.1].concat(),
                                &format!("{:?}", good),
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
                    ui.label("min met need");
                    ui.label("trickleback");
                    for good_index in 0..Good::VARIANT_COUNT {
                        ui.label(format!("Owned {:?}", Good::try_from(good_index).unwrap()));
                    }
                    ui.end_row();
                    match game.world.selected_province {
                        Some(selected_province) => {
                            for (i, slice) in game.world.provinces[selected_province]
                                .pops
                                .pop_slices
                                .iter()
                                .enumerate()
                            {
                                let culture =
                                    Culture::try_from(i / Industry::VARIANT_COUNT).unwrap();
                                let industry =
                                    Industry::try_from(i % Industry::VARIANT_COUNT).unwrap();
                                ui.label(format!("{:?}", culture));
                                ui.label(format!("{:?}", industry));
                                ui.label(format!("{:}", big_number_format(slice.population)));
                                ui.label(format!("{:}", big_number_format(slice.money)));
                                ui.label(format!("{:.2}", slice.minimum_met_needs));
                                ui.label(format!("{:.2}", slice.trickleback));
                                for good_index in 0..Good::VARIANT_COUNT {
                                    ui.label(format!("{:.2}", slice.owned_goods[good_index]));
                                }
                                ui.end_row();
                            }
                        }
                        None => {}
                    }
                })
            });
        }
        if pop_ideology_info_open {
            Window::new("Pop Table").show(&ctx, |ui| {
                ComboBox::from_label("Select Slice").show_ui(ui, |ui| {
                    match game.world.selected_province {
                        Some(selected_province) => {
                            for (i, _) in game.world.provinces[selected_province]
                                .pops
                                .pop_slices
                                .iter()
                                .enumerate()
                            {
                                ui.selectable_value(
                                    &mut selected_slice,
                                    i,
                                    format!("{:?}", Industry::try_from(i).unwrap()),
                                );
                            }
                        }
                        None => {}
                    }
                });
                match game.world.selected_province {
                    Some(selected_province) => {
                        let beliefs = &game.world.provinces[selected_province].pops.pop_slices
                            [selected_slice]
                            .beliefs;
                        for (question, response) in beliefs.responses.iter().enumerate() {
                            let question = Beliefs::get_question(question);
                            let question_text = question.question;
                            let answer = if response.get_value() > 0.0 {
                                question.upper_bound_meaning
                            } else {
                                question.lower_bound_meaning
                            };
                            ui.label(format!(
                                "{}: {:.2} towards {} with importance of {:.2}",
                                question_text,
                                response.get_value().abs(),
                                answer,
                                response.get_importance()
                            ));
                        }
                    }
                    None => {}
                }
            });
        }
        if province_info_open {
            Window::new("Province Info").show(&ctx, |ui| {
                Grid::new("province_info_grid")
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("Name");
                        ui.label("Id");
                        ui.label("Position");
                        ui.label("Population");
                        ui.label("Money");
                        ui.label("Travel Cost");
                        ui.label("Tax Rate");
                        // ui.label("Recruit Ratio");
                        for industry_index in 0..Industry::VARIANT_COUNT {
                            ui.label(format!(
                                "{:?} Industry Size",
                                Industry::try_from(industry_index).unwrap()
                            ));
                            ui.label("Productivity");
                        }

                        ui.end_row();
                        match game.world.selected_province {
                            Some(province_key) => {
                                let province = &game.world.provinces[province_key];
                                let province_population: f64 = province
                                    .pops
                                    .pop_slices
                                    .iter()
                                    .map(|a| (a.population))
                                    .sum();
                                let province_money: f64 =
                                    province.pops.pop_slices.iter().map(|a| (a.money)).sum();
                                ui.label(format!("{:}", province.name));
                                ui.label(format!("{:?}", province_key));
                                ui.label(format!("{:?}", province.position));
                                ui.label(format!("{:}", big_number_format(province_population)));
                                ui.label(format!("{:}", big_number_format(province_money)));
                                ui.label(format!("{:.2}", province.trader_cost_multiplier));
                                ui.label(format!("{:.2}", province.tax_rate));
                                // ui.label(format!("{:.2}", province.recruit_limiter));
                                for industry_index in 0..Industry::VARIANT_COUNT {
                                    ui.label(format!(
                                        "{:.1}",
                                        province.industry_data[industry_index].size
                                    ));
                                    ui.label(format!(
                                        "{:.1}",
                                        province.industry_data[industry_index].productivity
                                    ));
                                }
                                ui.end_row();
                            }
                            None => {}
                        }
                    });
                if let Some(selected_province_key) = game.world.selected_province {
                    Grid::new("Languages Table").striped(true).show(ui, |ui| {
                        ui.label("Language Name");
                        // ui.label("Military type");
                        ui.label("Presence");
                        ui.end_row();
                        for lang in game
                            .world
                            .language_manager
                            .languages
                            .values()
                            .filter(|l| l.province_presence[selected_province_key] > 0.01)
                        {
                            ui.label(&lang.name);

                            ui.label(format!(
                                "{:.2}",
                                lang.province_presence[selected_province_key]
                            ));
                            ui.end_row();
                        }
                    });

                    Grid::new("Military Table").striped(true).show(ui, |ui| {
                        ui.label("Org name");
                        // ui.label("Military type");
                        ui.label("Deployed troops");
                        ui.end_row();
                        for org in game
                            .world
                            .organizations
                            .values()
                            .filter(|o| o.military.deployed_forces[selected_province_key] > 0.5)
                        {
                            ui.label(&org.name);

                            ui.label(format!(
                                "{:}",
                                big_number_format(
                                    org.military.deployed_forces[selected_province_key]
                                )
                            ));
                            ui.end_row();
                        }
                    });
                    Grid::new("Control Table").striped(true).show(ui, |ui| {
                        ui.label("Org name");
                        ui.label("Control");
                        ui.end_row();
                        for org in game
                            .world
                            .organizations
                            .values()
                            .filter(|o| o.province_control[selected_province_key] > 0.01)
                        {
                            ui.label(&org.name);

                            ui.label(format!(
                                "{:.4}",
                                org.province_control[selected_province_key]
                            ));
                            ui.end_row();
                        }
                        ui.label("Sum");
                        ui.label(format!(
                            "{:.4}",
                            game.world
                                .organizations
                                .values()
                                .map(|o| o.province_control[selected_province_key])
                                .sum::<f64>()
                        ))
                    });
                }
            });
        }
        if global_market_open {
            Window::new("Global Price Plots").show(&ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    for good_index in 0..Good::VARIANT_COUNT {
                        let good = Good::try_from(good_index).unwrap();

                        let values = game.world.histories.global_prices[good_index].as_slices();
                        add_plot(
                            ui,
                            &[values.0, values.1].concat(),
                            &format!("{:?} Price", good),
                            1000,
                            0,
                        );
                    }
                });
            });
        }

        if global_supply_demand_open {
            Window::new("Global Supply Demand Plots").show(&ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    for good_index in 0..Good::VARIANT_COUNT {
                        let good = Good::try_from(good_index).unwrap();

                        let values_a = game.world.histories.global_supply[good_index].as_slices();
                        let values_b = game.world.histories.global_demand[good_index].as_slices();
                        double_line_plot(
                            ui,
                            &[values_a.0, values_a.1].concat(),
                            &[values_b.0, values_b.1].concat(),
                            &format!("{:?}", good),
                            1000,
                            0,
                        );
                    }
                });
            });
        }

        if organizations_list_open {
            Window::new("Organizations").show(&ctx, |ui| {
                ui.text_edit_singleline(&mut organization_filter);
                ScrollArea::vertical().show(ui, |ui| {
                    let mut sorted_orgs: Box<[_]> = game.world.organizations.iter().collect();
                    sorted_orgs.sort_by_key(|(k, v)| v.name.as_str());
                    for (id, organization) in sorted_orgs.iter().filter(|(_, o)| {
                        organization_filter.is_empty()
                            || o.name
                                .to_lowercase()
                                .contains(&organization_filter.to_lowercase())
                    }) {
                        ui.radio_value(
                            &mut game.world.selected_organization,
                            Some(OrganizationKey(id.0)),
                            &organization.name,
                        );
                    }
                });
            });
        }
        if organization_info_open {
            match game.world.selected_organization {
                Some(org_key) => {
                    let org = &game.world.organizations[org_key];
                    Window::new(format!("Organization: {:}", org.name))
                        .id(Id::new("OrganizationInfo"))
                        .show(&ctx, |ui| {
                            Grid::new(0).striped(true).show(ui, |ui| {
                                ui.label("money");
                                ui.label("army size");
                                ui.end_row();
                                ui.label(format!("{:}", big_number_format(org.money)));
                                ui.label(format!(
                                    "{:}",
                                    big_number_format(
                                        org.military.deployed_forces.0.iter().sum::<f64>()
                                    )
                                ));
                            });
                            ui.label("Branches");
                            for branch in &org.branches {
                                let branch = &game.world.branches[*branch];
                                let controlling_party =
                                    &game.world.political_parties[branch.controlling_party];
                                ui.label(format!(
                                    "{:} Party: {:}\n Ideology: {:?}",
                                    branch.name, controlling_party.name, controlling_party.ideology
                                ));
                            }
                        });
                }
                None => {
                    Window::new("No Organization Selected").show(&ctx, |_| {});
                }
            }
        }
        if console_open {
            Window::new("Console").show(&ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    for line in console_output.lines() {
                        ui.label(line);
                    }
                });
                let response = ui.text_edit_singleline(&mut console_input);
                if response.lost_focus() && ui.input().key_pressed(egui::Key::Enter) {
                    console_output = format!(
                        "{console_output}{:}",
                        game.world.process_command(&console_input)
                    );
                    console_input.clear()
                }
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
                            match vulkan_data.surface_capabilities{
                                Some(surface_capabilities) => {
                                    game.mouse_position.x =
                                    -((position.x / surface_capabilities.current_extent.width as f64) * 2.0 - 1.0);
                                    game.mouse_position.y =
                                        -((position.y / surface_capabilities.current_extent.height as f64) * 2.0 - 1.0);
    
                                },
                                None => {},
                            }
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
                            if input.virtual_keycode == Some(VirtualKeyCode::NumpadAdd) {
                                game_speed += match input.state {
                                    ElementState::Released => 1,
                                    ElementState::Pressed => 0,
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::NumpadSubtract) {
                                game_speed -= match input.state {
                                    ElementState::Released => {
                                        if game_speed > 0 {
                                            1
                                        } else {
                                            0
                                        }
                                    }
                                    ElementState::Pressed => 0,
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
                        &vulkan_data.get_projection(game.inputs.zoom),
                        game_speed,
                    );

                    // if let Some(selected_province) = game.selected_province {
                    //     let province_indices =
                    //         &game.world.provinces[selected_province].point_indices;
                    //     if let Some(line_data) = &mut vulkan_data.line_data {
                    //         line_data.set_color(&province_indices, Vector4::new(1.0,1.0,1.0,1.0));
                    //     }
                    // }

                    // let selected_points =
                    //     if let Some(selected_province) = game.world.selected_province {
                    //         game.world.provinces[selected_province].point_indices.iter()
                    //     } else {
                    //         [].iter()
                    //     };
                    // let targeted_points =
                    //     if let Some(targeted_province) = game.world.targeted_province {
                    //         game.world.provinces[targeted_province].point_indices.iter()
                    //     } else {
                    //         [].iter()
                    //     };
                    // let mut highlighted_points: HashSet<usize> =
                    //     HashSet::with_capacity(game.world.provinces.0.len());
                    // if let Some(selected_organization) = game.world.selected_organization {
                    //     for (province_key, &control_factor) in game.world.organizations
                    //         [selected_organization]
                    //         .province_control
                    //         .iter()
                    //     {
                    //         if control_factor > 0.5 {
                    //             highlighted_points
                    //                 .extend(game.world.provinces[province_key].point_indices.iter())
                    //         }
                    //     }
                    // }

                    // let mut player_country_points: HashSet<usize> =
                    //     HashSet::with_capacity(game.world.provinces.0.len());
                    // for (province_key, province) in game.world.provinces.0.iter().enumerate() {
                    //     let province_key = ProvinceKey(province_key);
                    //     if game.world.organizations[game.world.player_organization].province_control
                    //         [province_key]
                    //         > 0.01
                    //     {
                    //         player_country_points.extend(province.point_indices.iter())
                    //     }
                    // }
                    // if let Some(line_data) = &mut vulkan_data.line_data {
                    //     line_data.update_selection(
                    //         selected_points,
                    //         highlighted_points.iter(),
                    //         targeted_points,
                    //         player_country_points.iter(),
                    //     )
                    // }
                    if let Some(province_data) = &mut vulkan_data.province_data{
                        for (org_key, org) in game.world.organizations.iter(){
                            for (province, &control) in org.province_control.iter(){
                                if control > 0.5{
                                    for &index in &game.world.provinces[province].point_indices{
                                        province_data.vertex_data[index].nation_index = org_key.0 as u32;
                                        province_data.vertex_data[index].flags = 0;
                                    }
                                }
                            }
                        }
                        if let Some(selected_province) = game.world.selected_province{
                            for &index in &game.world.provinces[selected_province].point_indices{
                                province_data.vertex_data[index].flags |= provinceflags::SELECTED;
                            }
                        }
                        if let Some(targeted_province) = game.world.targeted_province{
                            for &index in &game.world.provinces[targeted_province].point_indices{
                                province_data.vertex_data[index].flags |= provinceflags::TARGETED;
                            }
                        }
                    } 

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
    let view_matrix = game.camera.get_view_matrix(game.planet.get_transform());

    let render_object = &mut vulkan_data.objects[game.planet.render_object_index];
    let model_matrix = (Matrix4::from(Translation3::from(game.planet.position))
        * Matrix4::from(Rotation3::from(game.planet.rotation)))
    .cast();
    render_object.model = model_matrix;

    vulkan_data.uniform_buffer_object.view = view_matrix.cast();
    vulkan_data.uniform_buffer_object.proj = projection_matrix.cast();

    if let Some(line_data) = &mut vulkan_data.province_data {
        line_data.model_view_projection =
            (projection_matrix * view_matrix * game.planet.get_transform()).cast();
    }

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
    vulkan_data.uniform_buffer_object.time = game.world.current_year as f32;
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
