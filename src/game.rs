use std::path::Path;
use crate::renderer::*;
use nalgebra::{Matrix4, Rotation3, Translation3, UnitQuaternion, Vector2, Vector3};
use std::time::{Instant};
use winit::window::Window;

const RADIUS: f64 = 6_371_000.0;

// const GOLDEN_RATIO: f64 = (1.0 + 5.0.sqrt())/2.0;
// struct GlobePoint{
//     elevation: f64,
// }
//
//
// struct Globe{
//     points: Vec<GlobePoint>,
// }
// impl Globe{
//     fn new(num_points: usize)-> Self{
//
//         let points = vec![GlobePoint{elevation:0.0}; num_points];
//         let output = Self{
//             points
//         };
//
//         return output;
//     }
//     fn get_position(&self, point_index: usize) -> Vector3<f64>{
//         let theta =  2.0 * f64::PI() * point_index as f64 / GOLDEN_RATIO;
//         let phi = (1.0 - 2.0 * (point_index as f64 + 0.5)/self.points.len() as f64).acos();
//         return Vector3::new(
//             theta.cos() * phi.sin(),
//             theta.sin() * phi.sin(),
//             phi.cos(),
//         )
//     }
// }

pub(crate) struct GameObject {
    position: Vector3<f64>,
    rotation: UnitQuaternion<f64>,
    render_object_index: usize,
}
impl GameObject {
    fn new(render_object_index: usize) -> Self{
        return Self{
            position: Vector3::new(0.0,0.0,0.0,),
            rotation: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.0),
            render_object_index
        }
    }
    fn process(&mut self, _delta_time: f64) {
        //Do stuff
    }
}


pub(crate) struct Inputs {
    pub(crate) forward: f64,
    pub(crate) backward: f64,
    pub(crate) left: f64,
    pub(crate) right: f64,
    pub(crate) camera_y: f64,
    pub(crate) camera_x: f64,
    pub(crate) up: f64,
    pub(crate) down: f64,
    pub(crate) sprint: f64,
}
impl Inputs {
    pub(crate) fn new() -> Self {
        return Inputs {
            forward: 0.0,
            backward: 0.0,
            left: 0.0,
            right: 0.0,
            camera_y: 0.0,
            camera_x: 0.0,
            up: 0.0,
            down: 0.0,
            sprint: 0.0,
        };
    }
}
struct Camera{
    position: Vector3<f64>,
    rotation: UnitQuaternion<f64>,
}
impl Camera{
    fn new()->Self{
        let position = Vector3::new(10_000_000.0,0.0,0.0);
        let rotation = UnitQuaternion::<f64>::identity();

        Self{
            position,
            rotation
        }
    }
}

pub(crate) struct Game {
    objects: Vec<GameObject>,
    pub(crate) mouse_buffer: Vector2<f64>,
    pub(crate) inputs: Inputs,
    pub(crate) focused: bool,
    last_frame_instant: Instant,
    pub(crate) vulkan_data: VulkanData,
    camera: Camera,
    planet_index:usize,
}

impl Game {
    pub(crate) fn new(window: &Window) -> Self {
        let mut objects = vec![];

        let mut vulkan_data = VulkanData::new();
        vulkan_data.init_vulkan(&window);

        let planet_index = objects.len();
        objects.push(GameObject::new(vulkan_data.load_folder("models/planet".parse().unwrap())));
        objects[planet_index].position = Vector3::new(0.0,0.0,0.0);
        vulkan_data.objects[objects[planet_index].render_object_index].is_globe = true;

        vulkan_data.update_vertex_buffer();

        let mut game = Game {
            objects,
            mouse_buffer: Vector2::new(0.0, 0.0),
            inputs: Inputs::new(),
            focused: false,
            last_frame_instant: Instant::now(),
            vulkan_data,
            camera: Camera::new(),
            planet_index
        };
        game.load_elevation_data();
        return game;
    }

    fn load_elevation_data(&mut self){
        let paths = std::fs::read_dir("../GSG/elevation").unwrap();

        #[derive(Copy, Clone, bincode::Encode, bincode::Decode)]
        struct VertexElevation {
            sum: f64,
            sample_count: usize,
        }
        let vertex_count = self.vulkan_data.objects[self.objects[self.planet_index].render_object_index].vertex_count as usize;
        let vertex_start = self.vulkan_data.objects[self.objects[self.planet_index].render_object_index].vertex_start as usize;
        let _index_count = self.vulkan_data.objects[self.objects[self.planet_index].render_object_index].index_count as usize;
        let _index_start = self.vulkan_data.objects[self.objects[self.planet_index].render_object_index].index_start as usize;

        let mut vertices_elevation = std::iter::repeat(
            VertexElevation { sum: 0.0, sample_count: 0 })
            .take(vertex_count)
            .collect::<Vec<_>>();


        let bincode_config = bincode::config::Configuration::standard();
        match std::fs::read(Path::new("vertex_elevations.bin")){
            Ok(bytes) => {
                vertices_elevation = bincode::decode_from_slice(&bytes, bincode_config).unwrap();
            }
            Err(_) => {
                for folder in paths{
                    let mut images = std::fs::read_dir(folder.unwrap().path()).unwrap();
                    let path = images.find(|file|{file.as_ref().unwrap().file_name().to_str().unwrap().to_lowercase().contains("mea")}).unwrap().unwrap().path();
                    println!("Path: {:?}", path);
                    let dataset = gdal::Dataset::open(path).unwrap();
                    let mut transform = dataset.geo_transform().unwrap();
                    let mut inv_transform = [0.0f64; 6];

                    unsafe {
                        assert_eq!(gdal_sys::GDALInvGeoTransform(
                            transform.as_mut_ptr(),
                            inv_transform.as_mut_ptr(),
                        ),1,"InvGeoTransform failed");
                    }

                    let rasterband = dataset.rasterband(1).unwrap();
                    // let stats = rasterband.get_statistics(false, true).unwrap();
                    // let elevation_difference = stats.max - stats.min;
                    for (vertex_index, vertex) in self.vulkan_data.vertices[vertex_start..][..vertex_count].iter().enumerate() {
                        let latitude = ((vertex.position.y / RADIUS as f32).asin() as f64).to_degrees();
                        let longitude = (vertex.position.z.atan2(vertex.position.x) as f64).to_degrees();


                        // let pixel = inv_transform * Vector2::new(longitude,latitude).cast();
                        let x: f64 = inv_transform[0] + longitude * inv_transform[1] + latitude * inv_transform[2];
                        let y: f64 = inv_transform[3] + longitude * inv_transform[4] + latitude * inv_transform[5];
                        let pixel = Vector2::new(x, y);


                        if pixel.x > 0.0 && pixel.x < rasterband.size().0 as f64 && pixel.y > 0.0 && pixel.y < rasterband.size().1 as f64 {
                            let slice = &mut [0.0f64];

                            rasterband.read_into_slice((pixel.x as isize, pixel.y as isize),
                                                       (1, 1),
                                                       (1, 1),
                                                       slice,
                                                       Some(gdal::raster::ResampleAlg::Bilinear)).unwrap();
                            // if slice[0] > stats.max{
                            //     println!("HIT");
                            //     println!("min: {:}, max: {:}, actual: {:}, multiplied: {:}",stats.min,stats.max,slice[0],stats.min + (slice[0] as f64 / (i16::MAX as f64)) * elevation_difference);
                            // }
                            // vertices_elevation[vertex_index].sum += stats.min + (slice[0] as f64 / (i16::MAX as f64)) * elevation_difference;
                            vertices_elevation[vertex_index].sum += slice[0] as f64;

                            vertices_elevation[vertex_index].sample_count += 1
                        }
                    }

                }
                std::fs::write("vertex_elevations.bin", bincode::encode_to_vec(vertices_elevation.clone(), bincode_config).unwrap());
            }
        }
        for i in 0..vertices_elevation.len(){
            self.vulkan_data.vertices[vertex_start + i].elevation = (vertices_elevation[i].sum / vertices_elevation[i].sample_count as f64) as f32;
            self.vulkan_data.vertices[vertex_start + i].normal = Vector3::zeros();

            // println!("Elevation: {:}", (vertices_elevation[i].sum / (vertices_elevation[i].sample_count as f64)) as f32);
        }
        for triangle_index in (vertex_start..(vertex_start + vertex_count)).step_by(3){
            let edge1 = self.vulkan_data.vertices[triangle_index].position - self.vulkan_data.vertices[triangle_index + 1].position;
            let edge2 = self.vulkan_data.vertices[triangle_index].position - self.vulkan_data.vertices[triangle_index + 2].position;
            let weighted_normal = (edge1.cross(&edge2));
            for i in 0..3{
                self.vulkan_data.vertices[triangle_index + i].normal += weighted_normal
            }
        }
        for i in 0..vertices_elevation.len(){
            self.vulkan_data.vertices[vertex_start + i].normal = self.vulkan_data.vertices[vertex_start + i].normal.normalize();
        }
        self.vulkan_data.update_vertex_buffer();
    }

    pub(crate) fn process(&mut self) {
        let delta_time = self.last_frame_instant.elapsed().as_secs_f64();
        self.last_frame_instant = std::time::Instant::now();

        for i in 0..self.objects.len() {
            self.objects[i].process(delta_time);
        }
        self.camera.position += Vector3::new(0.0,self.inputs.camera_y * 2_000_000.0 * delta_time,0.0);
        self.camera.position = UnitQuaternion::from_euler_angles(0.0, -self.inputs.camera_x * 0.3 * delta_time,0.0) * self.camera.position;
        self.camera.rotation = UnitQuaternion::face_towards(&(self.camera.position), &Vector3::new(0.0,-1.0,0.0));

        self.update_renderer();
        self.vulkan_data.transfer_data_to_gpu();
        self.vulkan_data.draw_frame();
    }

    fn update_renderer(&mut self){
        let projection = self.vulkan_data.get_projection_matrix();

        let cubemap_projection = self.vulkan_data.get_cubemap_projection_matrix();
        let view_matrix = (Translation3::from(self.camera.position) * Rotation3::from(self.camera.rotation)).inverse().to_homogeneous().cast::<f64>();
        let view_matrix_no_translation = (Matrix4::from(self.camera.rotation)).try_inverse().unwrap().cast();


        self.vulkan_data.cubemap.as_mut().unwrap().process(view_matrix_no_translation, cubemap_projection);


        for i in 0..self.objects.len() {
            self.vulkan_data.objects[self.objects[i].render_object_index].model =
                (Matrix4::from(Translation3::from(self.objects[i].position))
                    * Matrix4::from(Rotation3::from(self.objects[i].rotation))).cast();

            self.vulkan_data.objects[self.objects[i].render_object_index].view = view_matrix.cast();
            self.vulkan_data.objects[self.objects[i].render_object_index].proj = projection;
        }

        for quad in self.vulkan_data.fullscreen_fragment_quads.iter_mut(){
            quad.model = view_matrix_no_translation.try_inverse().unwrap().cast();
            quad.view = view_matrix.try_inverse().unwrap().cast();
            quad.proj = projection;

        }

        self.vulkan_data.uniform_buffer_object.mouse_position = Vector2::new(
            (self.mouse_buffer.x.clamp(0.0,self
                .vulkan_data
                .surface_capabilities
                .unwrap()
                .current_extent
                .width as f64)
                ) as f32,
            (self.mouse_buffer.y.clamp(0.0,self
                .vulkan_data
                .surface_capabilities
                .unwrap()
                .current_extent
                .height as f64)
                ) as f32,
        );
    }
}
