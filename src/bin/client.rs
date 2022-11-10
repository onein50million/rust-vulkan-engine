use egui::Color32;
use egui::plot::Value;
use egui::plot::Values;
use egui::ComboBox;
use egui::FontDefinitions;
use egui::FontFamily;
use egui::Grid;
use egui::Label;
use egui::SidePanel;
use egui::TextStyle;
use egui::Visuals;

use egui::ScrollArea;

use egui::TopBottomPanel;
use egui::Ui;
use egui::Window;
use egui_winit::winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use egui_winit::winit::event_loop::{ControlFlow, EventLoop};
use egui_winit::winit::window::WindowBuilder;

use float_ord::FloatOrd;
use nalgebra::Orthographic3;
use nalgebra::Point3;
use nalgebra::{Matrix4, Rotation3, Translation3, Vector2, Vector3};

use noise::NoiseFn;
use num_enum::IntoPrimitive;
use num_enum::TryFromPrimitive;
use rust_vulkan_engine::game::client::Game;

use rust_vulkan_engine::game::directions;
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
use rust_vulkan_engine::world::agent::AgentAction;
use rust_vulkan_engine::world::organization::DiplomaticAction;
use rust_vulkan_engine::world::organization::OrganizationKey;
use rust_vulkan_engine::world::*;
use rusttype::Font;
use variant_count::VariantCount;

use std::convert::TryFrom;
use std::f64::consts::PI;
use std::fmt::Debug;
use std::fs;
use std::net::UdpSocket;
use std::ops::Add;

use winit::event::{MouseButton, MouseScrollDelta};

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
    let num_samples = num_samples.min(values.len()).max(1);
    let chunk_size = (values.len() / num_samples).max(1);
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
            .include_x(values.get(0).unwrap_or(&(0.0, 0.0)).0)
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
        let num_samples = num_samples.min(values.len()).max(1);
        let chunk_size = (values.len() / num_samples).max(1);
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
        let num_samples = num_samples.min(values.len()).max(1);
        let chunk_size = (values.len() / num_samples).max(1);
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
            .include_x(
                values_a
                    .last()
                    .unwrap_or(&(0.0, 0.0))
                    .0
                    .max(values_b.last().unwrap_or(&(0.0, 0.0)).0),
            )
            .allow_drag(false),
    );
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

    let mut visuals = ctx.style().visuals.clone();
    visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(0x60, 0x78, 0x8b);
    visuals.faint_bg_color = Color32::from_rgb(0x8e, 0xac, 0xbb);
    visuals.extreme_bg_color = Color32::from_rgb(0x34, 0x51, 0x5e);
    visuals.widgets.noninteractive.fg_stroke.color = Color32::from_rgb(0, 0, 0);
    visuals.widgets.active.fg_stroke.color = Color32::from_rgb(0,0,0);
    visuals.widgets.inactive.fg_stroke.color = Color32::from_rgb(0x4e, 0x01, 0x17);

    // visuals.widgets.noninteractive.bg_stroke.color = Color32::from_rgb(0x4e, 0x01, 0x17);
    ctx.set_visuals(visuals);

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
    let nation_names_and_defintions = get_countries();
    // for nation in nations_map.iter(){
    //     if nation.is_some(){
    //         dbg!(nation);
    //     }
    // }
    let (provinces_map, province_vertices, province_indices, province_names, province_owners) =
        get_provinces();
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
                            Some(
                                *province_owners
                                    .0
                                    .get(key.0)
                                    .expect("Missing province owner in province"),
                            ),
                            language_map[i],
                        );
                    }
                }
                province_data
            };

            let mut province_indices_vec = vec![vec![]; num_provinces];
            dbg!(num_provinces);
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

        const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let font = Font::try_from_vec(fs::read("fonts/Aboreto-Regular.ttf").unwrap()).unwrap();

        let width = 64;
        let height = 128;
        let font = font.layout(
            LETTERS,
            rusttype::Scale {
                x: width as f32,
                y: height as f32,
            },
            rusttype::Point { x: 0.0, y: 0.0 },
        );

        vulkan_data.create_province_data(
            province_vertices.iter().map(|&v| v),
            province_indices.0.iter().flat_map(|p| p.iter().map(|&i| i)),
            font,
            width,
            height,
            LETTERS.len(),
        );
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

    #[repr(usize)]
    #[derive(VariantCount, PartialEq, Debug, IntoPrimitive, TryFromPrimitive)]
    enum SortOrder {
        Name,
        Population,
        ArmySize,
    }
    let mut organization_filter = String::new();

    let mut player_organization_detals_open = false;
    let mut market_window_open = false;
    let mut organizations_window_open = false;
    let mut organizations_sort = SortOrder::Name;

    enum SelectedTab {
        Pops,
        Market,
        Industry,
        Military,
        Details,
    }
    let mut province_selected_tab = SelectedTab::Pops;
    let mut selected_good_index = 0;

    let mut console_open = false;
    let mut console_input = String::new();
    let mut console_output = String::new();
    let mut game_speed = 0u16;

    // let mut actions_window_open = false;

    // let mut slider_value = 0.0;
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
            });
            ui.horizontal(|ui| {
                if let Some(player_org) = game.player_agent.controlling_organization {
                    ui.checkbox(
                        &mut player_organization_detals_open,
                        format!("{}", game.world.organizations[player_org].name),
                    );
                }
                ui.checkbox(&mut market_window_open, "Global Market");
                ui.checkbox(&mut organizations_window_open, "Organizations");
                ui.checkbox(&mut console_open, "Console");
            });
        });

        TopBottomPanel::bottom("bottom_panel").show(&ctx, |ui| {
            ui.label(format!(
                "Political Power ⚖: {:}",
                game.player_agent.political_power
            ));
            if let Some(player_org) = game.player_agent.controlling_organization {
                ui.horizontal(|ui| {
                    ui.label(&game.world.organizations[player_org].name);
                    ui.label(format!(
                        "Money: {:}",
                        big_number_format(game.world.organizations[player_org].money)
                    ));

                    Grid::new("owned_goods").show(ui, |ui| {
                        for good_index in 0..Good::VARIANT_COUNT {
                            let good = Good::try_from(good_index).unwrap();
                            ui.label(format!(
                                "{:?}: {:.1}",
                                good, &game.world.organizations[player_org].owned_goods[good_index]
                            ));
                            if good_index % 5 == 0 {
                                ui.end_row();
                            }
                        }
                    });
                });
            }
        });

        if let Some(player_org) = game.player_agent.controlling_organization {
            if player_organization_detals_open {
                Window::new(&game.world.organizations[player_org].name).show(&ctx, |ui| {
                    ui.vertical(|ui| {
                        ui.label("Diplo Offers");
                        let mut decision = None;
                        let mut decision_index = None;
                        for (i, offer) in game.world.organizations[player_org]
                            .diplomatic_offers
                            .iter()
                            .enumerate()
                        {
                            ui.horizontal(|ui| {
                                if ui.button("Accept").clicked() {
                                    decision = Some(true);
                                    decision_index = Some(i);
                                }
                                if ui.button("Decline").clicked() {
                                    decision = Some(false);
                                    decision_index = Some(i);
                                }
                                ui.label(format!(
                                    "From: {}, Offer: {:?}",
                                    game.world.organizations[offer.from].name, offer.offer_type
                                ))
                            });
                        }
                        if let (Some(decision), Some(index)) = (decision, decision_index) {
                            if let Err(err) = game.player_agent.attempt_action(
                                &mut game.world,
                                AgentAction::RespondToDiplomaticOffer(index, decision),
                            ) {
                                dbg!(err);
                            }
                        }
                    });
                });
            }
        }

        if let Some(selected_province) = game.world.selected_province {
            let controls_province =
                if let Some(player_org) = game.player_agent.controlling_organization {
                    game.world.organizations[player_org].province_control[selected_province] > 0.5
                } else {
                    false
                };
            Window::new("Province Details").show(&ctx, |ui| {
                TopBottomPanel::new(egui::panel::TopBottomSide::Top, "province_tabs").show_inside(
                    ui,
                    |ui| {
                        ui.horizontal(|ui| {
                            if ui.button("Pops").clicked() {
                                province_selected_tab = SelectedTab::Pops
                            } else if ui.button("Industry").clicked() {
                                province_selected_tab = SelectedTab::Industry
                            } else if ui.button("Market").clicked() {
                                province_selected_tab = SelectedTab::Market
                            } else if ui.button("Military").clicked() {
                                province_selected_tab = SelectedTab::Military
                            } else if ui.button("Details").clicked() {
                                province_selected_tab = SelectedTab::Details
                            }
                        });
                    },
                );
                match province_selected_tab {
                    SelectedTab::Pops => {
                        Grid::new(0).striped(true).show(ui, |ui| {
                            ui.label("Culture");
                            ui.label("Industry");
                            ui.label("Population");
                            ui.label("Money");
                            ui.label("min met need");
                            ui.label("Need Bottleneck");
                            ui.label("trickleback");
                            // for good_index in 0..Good::VARIANT_COUNT {
                            //     ui.label(format!("Owned {:?}", Good::try_from(good_index).unwrap()));
                            // }
                            ui.end_row();
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
                                ui.label(format!("{:?}", slice.need_bottleneck));
                                ui.label(format!("{:.2}", slice.trickleback));
                                // for good_index in 0..Good::VARIANT_COUNT {
                                //     ui.label(format!(
                                //         "{:}",
                                //         big_number_format(slice.owned_goods[good_index])
                                //     ));
                                // }
                                ui.end_row();
                            }
                        });
                    }
                    SelectedTab::Market => {
                        SidePanel::left("market_left_panel").show_inside(ui, |ui| {
                            ScrollArea::vertical().show(ui, |ui| {
                                Grid::new(0).striped(true).show(ui, |ui| {
                                    ui.label("");
                                    ui.label("Name");
                                    ui.label("Price");
                                    ui.label("Supply");
                                    ui.label("Demand");
                                    ui.end_row();
                                    for good_index in 0..Good::VARIANT_COUNT {
                                        let good = Good::try_from(good_index).unwrap();
                                        ui.radio_value(&mut selected_good_index, good_index, "");
                                        ui.label(format!("{:?}", good));
                                        ui.label(format!(
                                            "{:.2}",
                                            game.world.provinces[selected_province].market.price
                                                [good_index]
                                        ));
                                        ui.label(format!(
                                            "{:}",
                                            big_number_format(
                                                game.world.provinces[selected_province]
                                                    .market
                                                    .supply[good_index]
                                            )
                                        ));
                                        ui.label(format!(
                                            "{:}",
                                            big_number_format(
                                                game.world.provinces[selected_province]
                                                    .market
                                                    .demand[good_index]
                                            )
                                        ));
                                        ui.end_row()
                                    }
                                });
                            })
                        });
                        ScrollArea::vertical().show(ui, |ui| {
                            {
                                let values = game.world.histories.prices[selected_province]
                                    [selected_good_index]
                                    .as_slices();
                                add_plot(ui, &[values.0, values.1].concat(), "Price", 1000, 0);
                            }
                            {
                                let values = game.world.histories.supply[selected_province]
                                    [selected_good_index]
                                    .as_slices();
                                add_plot(ui, &[values.0, values.1].concat(), "Supply", 1000, 0);
                            }
                            {
                                let values = game.world.histories.demand[selected_province]
                                    [selected_good_index]
                                    .as_slices();
                                add_plot(ui, &[values.0, values.1].concat(), "Demand", 1000, 0);
                            }
                        });
                    }
                    SelectedTab::Industry => {
                        ScrollArea::vertical().show(ui, |ui| {
                            Grid::new("Industry_Grid").striped(true).show(ui, |ui| {
                                ui.label("Building");
                                ui.label("Size");
                                if controls_province {
                                    ui.label("Increase");
                                    ui.label("Decrease");
                                }

                                ui.end_row();
                                for building_index in 0..Building::VARIANT_COUNT {
                                    let building = Building::try_from(building_index).unwrap();
                                    let province = &game.world.provinces[selected_province];
                                    let building_size = province.industry_data
                                        [building.get_industry() as usize]
                                        .size
                                        * province.building_ratios[building_index];
                                    if building_size < 1.0 {
                                        continue;
                                    }
                                    ui.label(format!("{:?}", building));
                                    ui.label(format!("{:.0}", building_size));
                                    if controls_province {
                                        if ui.button("-").clicked() {
                                            let action = AgentAction::ModifyBuilding(
                                                selected_province,
                                                building,
                                                building_size * -0.1,
                                            );
                                            if let Err(err) = game
                                                .player_agent
                                                .attempt_action(&mut game.world, action)
                                            {
                                                dbg!(err);
                                            }
                                        }
                                        if ui.button("+").clicked() {
                                            let action = AgentAction::ModifyBuilding(
                                                selected_province,
                                                building,
                                                building_size * 0.1,
                                            );
                                            if let Err(err) = game
                                                .player_agent
                                                .attempt_action(&mut game.world, action)
                                            {
                                                dbg!(err);
                                            }
                                        }
                                    }

                                    ui.end_row();
                                }
                            });
                        });
                    }
                    SelectedTab::Military => {
                        Grid::new("Military Table").striped(true).show(ui, |ui| {
                            ui.label("Org name");
                            // ui.label("Military type");
                            ui.label("Deployed troops");
                            ui.label("Survival Needs");
                            ui.label("Ammo Needs");
                            ui.end_row();
                            for org in game.world.organizations.values().filter(|o| {
                                o.military.deployed_forces[selected_province].num_troops > 0.5
                            }) {
                                ui.label(&org.name);

                                ui.label(format!(
                                    "{:}",
                                    big_number_format(
                                        org.military.deployed_forces[selected_province].num_troops
                                    )
                                ));
                                ui.label(
                                    org.military.deployed_forces[selected_province]
                                        .survival_needs_met,
                                );
                                ui.label(
                                    org.military.deployed_forces[selected_province].ammo_needs_met,
                                );

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
                                .filter(|o| o.province_control[selected_province] > 0.01)
                            {
                                ui.label(&org.name);

                                ui.label(format!("{:.4}", org.province_control[selected_province]));
                                ui.end_row();
                            }
                            ui.label("Sum");
                            ui.label(format!(
                                "{:.4}",
                                game.world
                                    .organizations
                                    .values()
                                    .map(|o| o.province_control[selected_province])
                                    .sum::<f64>()
                            ))
                        });
                        if let Some(player_org) = game.player_agent.controlling_organization {
                            let troop_weight = game.world.organizations[player_org]
                                .military
                                .province_weights[selected_province];
                            if troop_weight >= 1.0 && ui.button("-").clicked() {
                                if let Err(err) = game.player_agent.attempt_action(
                                    &mut game.world,
                                    AgentAction::SetTroopWeight {
                                        province: selected_province,
                                        weight: troop_weight - 1.0,
                                    },
                                ) {
                                    dbg!(err);
                                }
                            }
                            ui.label(format!("Troop Weight: {:.2}", troop_weight));
                            if ui.button("+").clicked() {
                                if let Err(err) = game.player_agent.attempt_action(
                                    &mut game.world,
                                    AgentAction::SetTroopWeight {
                                        province: selected_province,
                                        weight: troop_weight + 1.0,
                                    },
                                ) {
                                    dbg!(err);
                                }
                            }
                        }
                    }
                    SelectedTab::Details => {
                        Grid::new("province_info_grid")
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("Name");
                                ui.label("Id");
                                ui.label("Position");
                                ui.label("Population");
                                ui.label("Money");
                                // ui.label("Travel Cost");
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
                                let province = &game.world.provinces[selected_province];
                                let province_population: f64 = province
                                    .pops
                                    .pop_slices
                                    .iter()
                                    .map(|a| (a.population))
                                    .sum();
                                let province_money: f64 =
                                    province.pops.pop_slices.iter().map(|a| (a.money)).sum();
                                ui.label(format!("{:}", province.name));
                                ui.label(format!("{:?}", selected_province));
                                ui.label(format!("{:.0?}", province.position));
                                ui.label(format!("{:}", big_number_format(province_population)));
                                ui.label(format!("{:}", big_number_format(province_money)));
                                ui.label(format!("{:.2}", province.tax_rate));
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
                            });
                    }
                }
            });
        }

        if organizations_window_open {
            Window::new("Organizations").show(&ctx, |ui| {
                SidePanel::left("org_left_panel").show_inside(ui, |ui| {
                    ComboBox::from_label("Sort Order").show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut organizations_sort,
                            SortOrder::Population,
                            "Population",
                        );
                        ui.selectable_value(
                            &mut organizations_sort,
                            SortOrder::ArmySize,
                            "Army Size",
                        );
                        ui.selectable_value(&mut organizations_sort, SortOrder::Name, "Name");
                    });
                    ui.text_edit_singleline(&mut organization_filter);
                    ScrollArea::vertical().show(ui, |ui| {
                        let mut sorted_orgs: Box<[_]> = game.world.organizations.iter().collect();
                        match organizations_sort {
                            SortOrder::Name => sorted_orgs.sort_by_key(|(_, v)| v.name.as_str()),
                            SortOrder::Population => sorted_orgs
                                .sort_by_key(|(k, _)| FloatOrd(game.world.get_org_population(*k))),
                            SortOrder::ArmySize => sorted_orgs.sort_by_key(|(_, v)| {
                                FloatOrd(
                                    v.military
                                        .deployed_forces
                                        .0
                                        .iter()
                                        .map(|p| p.num_troops)
                                        .sum::<f64>(),
                                )
                            }),
                        }
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
                if let Some(selected_org) = game.world.selected_organization {
                    let org = &game.world.organizations[selected_org];
                    Grid::new(0).striped(true).show(ui, |ui| {
                        ui.label("money");
                        ui.label("org wealth");
                        ui.label("army size");
                        ui.label("population");
                        ui.label("pop wealth");
                        ui.label("wealth per capita");
                        ui.end_row();
                        ui.label(format!("{:}", big_number_format(org.money)));
                        ui.label(format!(
                            "{:}",
                            big_number_format(game.world.get_org_wealth(selected_org))
                        ));
                        ui.label(format!(
                            "{:}",
                            big_number_format(
                                org.military
                                    .deployed_forces
                                    .0
                                    .iter()
                                    .map(|f| f.num_troops)
                                    .sum::<f64>()
                            )
                        ));
                        let population = game.world.get_org_population(selected_org);
                        let wealth = game.world.get_org_pop_wealth(selected_org);
                        let wealth_per_capita = wealth / population;
                        ui.label(format!("{:}", big_number_format(population)));
                        ui.label(format!("{:}", big_number_format(wealth)));
                        ui.label(format!("{:}", big_number_format(wealth_per_capita)));
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
                    if let Some(controlling_agent) = game
                        .ai_agents
                        .iter_mut()
                        .find(|a| a.controlling_organization == Some(selected_org))
                    {
                        if ui.button("Take Control").clicked() {
                            std::mem::swap(&mut game.player_agent, controlling_agent)
                        }
                    }
                    if let Some(controlling_org) = game.player_agent.controlling_organization {
                        if controlling_org != selected_org {
                            let relations = game
                                .world
                                .relations
                                .get_relations_mut(controlling_org, selected_org);
                            if !relations.at_war {
                                if ui.button("Declare War").clicked() {
                                    if let Err(err) = game.player_agent.attempt_action(
                                        &mut game.world,
                                        AgentAction::DiplomaticAction(
                                            selected_org,
                                            DiplomaticAction::DeclareWar,
                                        ),
                                    ) {
                                        dbg!(err);
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }

        if market_window_open {
            Window::new("Global Markets").show(&ctx, |ui| {
                SidePanel::left("global_market_left_panel").show_inside(ui, |ui| {
                    ScrollArea::vertical().show(ui, |ui| {
                        Grid::new(0).striped(true).show(ui, |ui| {
                            ui.label("");
                            ui.label("Name");
                            ui.label("Price");
                            ui.label("Supply");
                            ui.label("Demand");
                            ui.end_row();
                            for good_index in 0..Good::VARIANT_COUNT {
                                let good = Good::try_from(good_index).unwrap();
                                ui.radio_value(&mut selected_good_index, good_index, "");
                                ui.label(format!("{:?}", good));
                                ui.label(format!(
                                    "{:.2}",
                                    game.world.global_market.price[good_index]
                                ));
                                ui.label(format!(
                                    "{:}",
                                    big_number_format(game.world.global_market.supply[good_index])
                                ));
                                ui.label(format!(
                                    "{:}",
                                    big_number_format(game.world.global_market.demand[good_index])
                                ));
                                ui.end_row()
                            }
                        });
                    })
                });
                ScrollArea::vertical().show(ui, |ui| {
                    {
                        let values =
                            game.world.histories.global_prices[selected_good_index].as_slices();
                        add_plot(ui, &[values.0, values.1].concat(), "Price", 1000, 0);
                    }
                    {
                        let values =
                            game.world.histories.global_supply[selected_good_index].as_slices();
                        add_plot(ui, &[values.0, values.1].concat(), "Supply", 1000, 0);
                    }
                    {
                        let values =
                            game.world.histories.global_demand[selected_good_index].as_slices();
                        add_plot(ui, &[values.0, values.1].concat(), "Demand", 1000, 0);
                    }
                });
            });
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
                    if let Some(province_data) = &mut vulkan_data.province_data {
                        province_data.drawing = game.inputs.map_mode == map_modes::SATELITE;
                        for (org_key, org) in game.world.organizations.iter() {
                            //Deciding how to place the nation text label, it's pretty slow (~0.03ms) so it should either be optimized or done less frequently, or even replaced entirely
                            if let Some((province, _)) = org
                                .province_control
                                .iter()
                                .filter(|&(_, &c)| c > 0.5)
                                .max_by_key(|&(p, _)| {
                                    let province = &game.world.provinces[p];
                                    let largest_circle_radius = province
                                        .point_indices
                                        .iter()
                                        .map(|&index| {
                                            let vertex = &game.world.points[index];
                                            FloatOrd(nalgebra::distance(
                                                &Point3::from(vertex.cast()),
                                                &Point3::from(province.position),
                                            ))
                                        })
                                        .max()
                                        .unwrap()
                                        .0;
                                    let province_approx_area =
                                        largest_circle_radius * largest_circle_radius * PI;
                                    FloatOrd(province_approx_area)
                                })
                            {
                                let province = &game.world.provinces[province];
                                let province_position = province.position;

                                // let width = (World::RADIUS * 0.01) * org.name.len() as f64;
                                let width = province
                                    .point_indices
                                    .iter()
                                    .map(|&index| {
                                        let vertex = &game.world.points[index];
                                        FloatOrd(nalgebra::distance(
                                            &Point3::from(vertex.cast()),
                                            &Point3::from(province_position),
                                        ))
                                    })
                                    .max()
                                    .unwrap()
                                    .0
                                    * 0.5;
                                let height = (width / org.name.len() as f64) * 2.0;
                                let ortho = Orthographic3::new(
                                    -width * 0.5,
                                    width * 0.5,
                                    -height * 0.5,
                                    height * 0.5,
                                    -World::RADIUS,
                                    World::RADIUS,
                                )
                                .to_homogeneous()
                                .cast::<f32>();

                                province_data.nation_data[org_key.0].name_matrix = ortho
                                    * Matrix4::face_towards(
                                        &Point3::origin(),
                                        &Point3::from(province_position.normalize()),
                                        &directions::UP,
                                    )
                                    .try_inverse()
                                    .unwrap()
                                    .cast();
                                let mut name_string: Vec<_> = org
                                    .name
                                    .to_uppercase()
                                    .as_bytes()
                                    .iter()
                                    .map(|b| if b == &b' ' { 255 } else { b - b'A' })
                                    .collect();
                                for _ in 0..(4 - name_string.len().rem_euclid(4)) {
                                    name_string.push(255);
                                }
                                for (i, chunk) in name_string.chunks(4).enumerate() {
                                    if i < province_data.nation_data[org_key.0].name_string.len() {
                                        province_data.nation_data[org_key.0].name_string[i] =
                                            u32::from_le_bytes(chunk.try_into().unwrap());
                                    }
                                }
                                province_data.nation_data[org_key.0].name_length =
                                    org.name.len() as u32;
                            }
                            for (province, &control) in org.province_control.iter() {
                                if control > 0.5 {
                                    // position += game.world.provinces[province].position.cast();
                                    for &index in &game.world.provinces[province].point_indices {
                                        province_data.vertex_data[index].nation_index =
                                            org_key.0 as u32;
                                        province_data.vertex_data[index].flags = 0;
                                    }
                                }
                            }

                            // province_data.nation_data[org_key.0].name_matrix = vulkan_data.objects[planet_render_object_index].model * ortho * Matrix4::face_towards(&Point3::origin(), &Point3::from(position), &directions::UP).try_inverse().unwrap().cast();
                        }
                        if let Some(selected_province) = game.world.selected_province {
                            for &index in &game.world.provinces[selected_province].point_indices {
                                province_data.vertex_data[index].flags |= provinceflags::SELECTED;
                            }
                        }
                        if let Some(targeted_province) = game.world.targeted_province {
                            for &index in &game.world.provinces[targeted_province].point_indices {
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
    vulkan_data.post_process_ubo.as_mut().unwrap().get_mut().proj = projection_matrix.cast();

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
