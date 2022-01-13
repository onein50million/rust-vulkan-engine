use std::f64::consts::PI;
use crate::renderer::*;
use nalgebra::{Matrix4, Point3, Rotation3, Translation3, UnitQuaternion, Vector2, Vector3};
use std::path::Path;
use std::time::Instant;
use winit::window::Window;
use crate::marching_cubes::World;
use crate::Vertex;



pub(crate) struct GameObject {
    position: Vector3<f64>,
    rotation: UnitQuaternion<f64>,
    render_object_index: usize,
}
impl GameObject {
    fn new(render_object_index: usize) -> Self {
        return Self {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.0),
            render_object_index,
        };
    }
    fn process(&mut self, _delta_time: f64) {
        //Do stuff
    }
}

pub(crate) struct Inputs {
    pub(crate) up: f64,
    pub(crate) down: f64,
    pub(crate) left: f64,
    pub(crate) right: f64,
    pub(crate) map_mode: u8,
    pub(crate) zoom: f64,
    pub(crate) panning: bool,
    pub(crate) left_click: bool,
}
impl Inputs {
    pub(crate) fn new() -> Self {
        return Inputs {
            left: 0.0,
            right: 0.0,
            up: 0.0,
            down: 0.0,
            map_mode: 0,
            zoom: 1.0,
            panning: false,
            left_click: false
        };
    }
}
struct Camera {
    position: Vector3<f64>,
    rotation: UnitQuaternion<f64>,
}
impl Camera {
    fn new() -> Self {

        Self { position: Vector3::new(5.0,-10.0, 5.0), rotation: UnitQuaternion::face_towards(&Vector3::new(0.0,-1.0,0.0),&Vector3::new(1.0,0.0,0.0)) }
    }
    fn get_rotation(&self) -> UnitQuaternion<f64>{
        return self.rotation
    }
    fn get_position(&self) -> Vector3<f64>{
        return self.position;
    }
}

pub(crate) struct Game {
    game_start: Instant,
    objects: Vec<GameObject>,
    pub(crate) mouse_position: Vector2<f64>,
    last_mouse_position: Vector2<f64>,
    pub(crate) inputs: Inputs,
    pub(crate) focused: bool,
    last_frame_instant: Instant,
    pub(crate) vulkan_data: VulkanData,
    camera: Camera,
}

impl Game {
    pub(crate) fn new(window: &Window) -> Self {
        let mut objects = vec![];

        let world =  World::new_random();


        let mut vulkan_data = VulkanData::new();
        vulkan_data.init_vulkan(&window);

        objects.push(GameObject::new(vulkan_data.objects.len()));
        let voxel_render_object = RenderObject::new(
            &mut vulkan_data,
            world.generate_mesh(),
            vec![],
            TextureSet::new_empty(),
            false
        );

        vulkan_data.objects.push(
            voxel_render_object
        );

        vulkan_data.update_vertex_and_index_buffers();

        let game = Game {
            game_start: std::time::Instant::now(),
            objects,
            mouse_position: Vector2::new(0.0, 0.0),
            last_mouse_position: Vector2::new(0.0,0.0),
            inputs: Inputs::new(),
            focused: false,
            last_frame_instant: Instant::now(),
            vulkan_data,
            camera: Camera::new(),
        };
        return game;
    }

    pub(crate) fn process(&mut self) {
        let delta_time = self.last_frame_instant.elapsed().as_secs_f64();
        self.last_frame_instant = std::time::Instant::now();
        let delta_mouse = self.last_mouse_position - self.mouse_position;
        self.last_mouse_position = self.mouse_position;

        for i in 0..self.objects.len() {
            self.objects[i].process(delta_time);
        }

        // self.camera.rotation = UnitQuaternion::face_towards(&(self.camera.position - self.objects[0].position), &Vector3::new(0.0,-1.0,0.0));


        if self.inputs.panning{
            self.camera.position.x += delta_mouse.y * delta_time * 5.0;
            self.camera.position.z -= delta_mouse.x * delta_time * 5.0;
        }


        self.update_renderer();
        self.vulkan_data.transfer_data_to_gpu();
        self.vulkan_data.draw_frame();

    }

    fn update_renderer(&mut self) {
        // let clip = Matrix4::new(
        //                         1.0,  0.0, 0.0, 0.0,
        //                         0.0, -1.0, 0.0, 0.0,
        //                         0.0,  0.0, 0.5, 0.0,
        //                         0.0,  0.0, 0.5, 1.0);
        let clip = Matrix4::<f64>::identity();

        let projection = self.vulkan_data.get_projection(self.inputs.zoom);
        let projection_matrix = clip * projection.to_homogeneous();

        let cubemap_projection = self
            .vulkan_data
            .get_cubemap_projection(self.inputs.zoom);

        let cubemap_projection_matrix = clip
            * cubemap_projection.to_homogeneous();

        let view_matrix = (Matrix4::from(Translation3::from(self.camera.get_position()))
            * self.camera.get_rotation().to_homogeneous())
        .try_inverse()
        .unwrap();
        let view_matrix_no_translation =
            ((self.camera.get_rotation().to_homogeneous())
            .try_inverse()
            .unwrap());

        self.vulkan_data
            .cubemap
            .as_mut()
            .unwrap()
            .process(view_matrix_no_translation.cast(), cubemap_projection_matrix.cast());

        for i in 0..self.objects.len() {
            let model_matrix = (Matrix4::from(Translation3::from(self.objects[i].position))
                * Matrix4::from(Rotation3::from(self.objects[i].rotation)))
            .cast();
            self.vulkan_data.objects[self.objects[i].render_object_index].model = model_matrix;
            self.vulkan_data.objects[self.objects[i].render_object_index].view = view_matrix.cast();
            self.vulkan_data.objects[self.objects[i].render_object_index].proj = projection_matrix.cast();
        }

        self.vulkan_data.uniform_buffer_object.time = 0.0; //TODO: use actual time
        self.vulkan_data.uniform_buffer_object.mouse_position = Vector2::new(
            (self.mouse_position.x) as f32,
            (self.mouse_position.y) as f32,
        );
    }
}
