pub mod animation;
pub mod ui;
pub mod buffers;
pub mod combination_types;
pub mod cpu_image;
pub mod drawables;
pub mod vulkan_data;

//TODO: look into this: https://docs.rs/crevice/latest/crevice/index.html

//TODO: Generate JSON descriptor set info and parse that instead of having to manually keep everything up to date
//https://www.reddit.com/r/vulkan/comments/s7e9wn/reflection_on_shaders_to_determine_uniforms/

//BIG TODO: Split this monstrosity of a file into multiple easier to manage chunks

//BEHOLD: MY STUFF

#[derive(Debug, Copy, Clone)]
pub enum DanielError {
    Minimized,
    SwapchainNotCreated,
}

