mod cube;
mod game;
mod renderer;
mod support;
mod pop;
mod organization;
mod market;

use crate::game::*;
use crate::renderer::*;
use crate::support::*;
use egui_winit::winit::event::{DeviceEvent, ElementState, Event, VirtualKeyCode, WindowEvent};
use egui_winit::winit::event_loop::{ControlFlow, EventLoop};
use egui_winit::winit::window::WindowBuilder;
use std::env;
use winit::event::{MouseButton, MouseScrollDelta, ScanCode};
use crate::market::Good;

//Coordinate system for future reference:
//from starting location
//up: negative y
//down: positive y
//forward: negative z
//backward: positive z
//left: negative x
//right: positive x

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

    let mut game = Game::new(&window);

    let mut texture_version = 0;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        let raw_input = state.take_egui_input(&window);
        ctx.begin_frame(raw_input);

        if ctx.texture().version != texture_version {
            println!("UI Texture changed");
            texture_version = ctx.texture().version;
            game.vulkan_data.update_ui_texture(ctx.texture());
        }

        egui::SidePanel::left("my_left_panel").show(&ctx, |ui| {
            match game.selected_province{
                None => {}
                Some(province_id) => {
                    ui.add(egui::Label::new(format!("Population: {:.0}", game.provinces[province_id].get_population())));
                    let food = game.provinces[province_id].market[Good::Food];
                    ui.add(egui::Label::new(format!("Food:\nSupply: {:.2}\nDemand: {:.2}\nPrice: {:.2}",food.supply, food.demand, food.price )));
                }
            }
        });
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
        game.vulkan_data.ui_data.update_buffers(&vertices, &indices);

        state.handle_output(&window, &ctx, output);
        match event {
            Event::WindowEvent { event, .. } => {
                let egui_handling = state.on_event(&ctx, &event);
                if !egui_handling {
                    match event {
                        WindowEvent::Focused(is_focused) => game.focused = is_focused,
                        WindowEvent::CloseRequested => {
                            unsafe { game.vulkan_data.device.as_ref().unwrap().device_wait_idle() }
                                .unwrap();
                            close_app(&mut game.vulkan_data, control_flow);
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            game.mouse_position.x = position.x;
                            game.mouse_position.y = position.y;
                            // println!("Position: {:?}", position);
                        }
                        WindowEvent::MouseInput {
                            button, state, ..
                        } => {
                            match button {
                                MouseButton::Left => {
                                    if state == ElementState::Released {
                                        game.inputs.left_click = true;
                                    }
                                }
                                MouseButton::Middle => {
                                    game.inputs.panning = match state {
                                        ElementState::Pressed => { true }
                                        ElementState::Released => { false }
                                    }
                                }
                                _ => {}
                            };
                        }
                        WindowEvent::MouseWheel {
                            delta, ..
                        } => {
                            match delta {
                                MouseScrollDelta::LineDelta(_, vertical_lines) => {
                                    game.inputs.zoom = (game.inputs.zoom * 1.1f64.powf(vertical_lines as f64)).max(1.0);
                                }
                                _ => {}
                            }
                        }
                        WindowEvent::KeyboardInput { input, .. } => {
                            if input.virtual_keycode == Some(VirtualKeyCode::Escape)
                                && input.state == ElementState::Released
                            {
                                close_app(&mut game.vulkan_data, control_flow);
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Up) {
                                game.inputs.up = match input.state {
                                    ElementState::Released => 0.0,
                                    ElementState::Pressed => 1.0,
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Down) {
                                game.inputs.down = match input.state {
                                    ElementState::Released => 0.0,
                                    ElementState::Pressed => 1.0,
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Left) {
                                game.inputs.left = match input.state {
                                    ElementState::Released => 0.0,
                                    ElementState::Pressed => 1.0,
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Right) {
                                game.inputs.right = match input.state {
                                    ElementState::Released => 0.0,
                                    ElementState::Pressed => 1.0,
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
                                        game.inputs.map_mode = support::map_modes::ELEVATION
                                    }
                                    _ => {}
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Key3) {
                                match input.state {
                                    ElementState::Released => {
                                        game.inputs.map_mode = support::map_modes::ARIDITY
                                    }
                                    _ => {}
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Key4) {
                                match input.state {
                                    ElementState::Released => {
                                        game.inputs.map_mode = support::map_modes::POPULATION
                                    }
                                    _ => {}
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Key5) {
                                match input.state {
                                    ElementState::Released => game.inputs.map_mode = 4,
                                    _ => {}
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Key6) {
                                match input.state {
                                    ElementState::Released => game.inputs.map_mode = 5,
                                    _ => {}
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Key7) {
                                match input.state {
                                    ElementState::Released => game.inputs.map_mode = 6,
                                    _ => {}
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Key8) {
                                match input.state {
                                    ElementState::Released => game.inputs.map_mode = 7,
                                    _ => {}
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Key9) {
                                match input.state {
                                    ElementState::Released => game.inputs.map_mode = 8,
                                    _ => {}
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Key0) {
                                match input.state {
                                    ElementState::Released => game.inputs.map_mode = 9,
                                    _ => {}
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Equals) {
                                match input.state {
                                    ElementState::Released => game.inputs.zoom *= 1.1,
                                    _ => {}
                                };
                            }
                            if input.virtual_keycode == Some(VirtualKeyCode::Minus) {
                                match input.state {
                                    ElementState::Released => game.inputs.zoom *= 0.9,
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

                if time_since_last_frame > 1.0 / FRAMERATE_TARGET {
                    frametimes[frametimes_start] = last_tick.elapsed().as_secs_f64();
                    last_tick = std::time::Instant::now();

                    frametimes_start = (frametimes_start + 1) % FRAME_SAMPLES;
                    let mut average_frametime = 0.0;
                    for i in 0..frametimes.len() {
                        average_frametime += frametimes[i];
                    }
                    average_frametime /= FRAME_SAMPLES as f64;

                    window.set_title(format!("Frametimes: {:}", average_frametime).as_str());
                    game.process();
                    time_since_last_frame -= 1.0 / FRAMERATE_TARGET;
                }
                time_since_last_frame += elapsed_time;
            },
            _ => {}
        }
    })
}

fn close_app(vulkan_data: &mut VulkanData, control_flow: &mut ControlFlow) {
    *control_flow = ControlFlow::Exit;
    vulkan_data.cleanup();
}
