mod support;
mod cube;
mod game;
mod renderer;
mod nft;

use std::env;
use winit::event::{DeviceEvent, Event, VirtualKeyCode, WindowEvent, ElementState};
use winit::event_loop::{ControlFlow, EventLoop};
use crate::game::*;
use winit::window::WindowBuilder;
use crate::support::*;
use crate::renderer::*;


//Coordinate system for future reference:
//from starting location
//up: negative y
//down: positive y
//forward: negative z
//backward: positive z
//left: negative x
//right: positive x

fn main() {
    nft::getState();

    let mut frametime: std::time::Instant = std::time::Instant::now();
    let mut time_since_last_frame: f64 = 0.0; //seconds

    let mut frametimes = [0.0; FRAME_SAMPLES];
    let mut frametimes_start = 0;
    let mut last_tick = std::time::Instant::now();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .build(&event_loop)
        .expect("Failed to build window");
    window.set_cursor_grab(true).unwrap();
    window.set_cursor_visible(false);

    let mut game = Game::new(&window);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

           match event {
            Event::WindowEvent {
                event,
                ..
            } => {
                match event {
                    WindowEvent::Focused(is_focused) => game.focused = is_focused,
                    WindowEvent::CloseRequested => {
                        unsafe { game.vulkan_data.device.as_ref().unwrap().device_wait_idle() }.unwrap();
                        close_app(&mut game.vulkan_data, control_flow);
                    }
                    _ => {}
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
                    for i in 0..frametimes.len(){
                        average_frametime += frametimes[i];
                    }
                    average_frametime /= FRAME_SAMPLES as f64;

                    window.set_title(format!("Frametimes: {:}", average_frametime).as_str());
                    game.process();
                    time_since_last_frame -= 1.0 / FRAMERATE_TARGET;
                }
                time_since_last_frame += elapsed_time;
            }
            Event::DeviceEvent {
                device_id: _,
                event: device_event,
            } => match device_event {
                DeviceEvent::MouseMotion { delta } => {
                    game.mouse_buffer.x += delta.0;
                    game.mouse_buffer.y += delta.1;
                    game.mouse_buffer.x = game.mouse_buffer.x.min(game.vulkan_data.surface_capabilities.unwrap().current_extent.width as f64).max(0.0);
                    game.mouse_buffer.y = game.mouse_buffer.y.min(game.vulkan_data.surface_capabilities.unwrap().current_extent.height as f64).max(0.0);

                }
                DeviceEvent::Key(key) => {
                    match key.virtual_keycode {
                        Some(keycode) => {
                            if game.focused{
                                if keycode == VirtualKeyCode::Escape && key.state == ElementState::Released {
                                    close_app(&mut game.vulkan_data, control_flow);
                                }
                                if keycode == VirtualKeyCode::W{
                                    game.inputs.forward = match key.state{
                                        ElementState::Released => 0.0,
                                        ElementState::Pressed => 1.0
                                    };
                                }
                                if keycode == VirtualKeyCode::S{
                                    game.inputs.backward = match key.state{
                                        ElementState::Released => 0.0,
                                        ElementState::Pressed => 1.0
                                    };
                                }
                                if keycode == VirtualKeyCode::A{
                                    game.inputs.left = match key.state{
                                        ElementState::Released => 0.0,
                                        ElementState::Pressed => 1.0
                                    };
                                }
                                if keycode == VirtualKeyCode::D{
                                    game.inputs.right = match key.state{
                                        ElementState::Released => 0.0,
                                        ElementState::Pressed => 1.0
                                    };
                                }
                                if keycode == VirtualKeyCode::Space{
                                    game.inputs.up = match key.state{
                                        ElementState::Released => 0.0,
                                        ElementState::Pressed => 1.0
                                    };
                                }
                                if keycode == VirtualKeyCode::LControl{
                                    game.inputs.down = match key.state{
                                        ElementState::Released => 0.0,
                                        ElementState::Pressed => 1.0
                                    };
                                }
                                if keycode == VirtualKeyCode::LShift{
                                    game.inputs.sprint = match key.state{
                                        ElementState::Released => 0.0,
                                        ElementState::Pressed => 1.0
                                    };
                                }
                                if keycode == VirtualKeyCode::Up{
                                    game.inputs.camera_y = match key.state{
                                        ElementState::Released => 0.0,
                                        ElementState::Pressed => 1.0
                                    };
                                }
                                if keycode == VirtualKeyCode::Down{
                                    game.inputs.camera_y = match key.state{
                                        ElementState::Released => 0.0,
                                        ElementState::Pressed => -1.0
                                    };
                                }
                                if keycode == VirtualKeyCode::Left{
                                    game.inputs.camera_x = match key.state{
                                        ElementState::Released => 0.0,
                                        ElementState::Pressed => -1.0
                                    };
                                }
                                if keycode == VirtualKeyCode::Right{
                                    game.inputs.camera_x = match key.state{
                                        ElementState::Released => 0.0,
                                        ElementState::Pressed => 1.0
                                    };
                                }

                            }


                        }
                        _ => {}
                    }
                },
                _ => {}
            },
            _ => {}
        }
    })
}

fn close_app(vulkan_data: &mut VulkanData, control_flow: &mut ControlFlow) {
    *control_flow = ControlFlow::Exit;
    vulkan_data.cleanup();
}