use std::path::Path;
use crate::renderer::*;
use nalgebra::{Matrix4, Rotation3, Translation3, UnitQuaternion, Vector2, Vector3};
use std::time::{Instant};
use gdal::errors::GdalError;
use winit::window::Window;

const RADIUS: f64 = 6_371_000.0;
const NEW_SEA_LEVEL: f64 = 76.0;
const EARTH_SURFACE_AREA: f64 = 510_000_000_000.0;
const GLOBAL_TEMPERATURE_CHANGE: f64 = 10.0; //degrees K
const POLAR_AMPLIFICATION_FACTOR: f64 = 3.0;

//Name ideas:
//Greenhouse Earth
//76 Below

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
    pub(crate) map_mode: u8,
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
            map_mode: 0
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

#[derive(Copy, Clone)]
struct RawVertexData {
    sum: f64,
    sample_count: usize,
    data_priority: usize,
}
#[derive(bincode::Encode, bincode::Decode)]
struct VertexData {
    elevation: f64,
    aridity: f64,
    population: f64,
    warmest_temperature: f64,
    coolest_temperature: f64,
}

pub(crate) struct Game {
    game_start: std::time::Instant,
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

        vulkan_data.load_folder("models/planet/deep_water".parse().unwrap()); //Dummy objects to fill offsets
        vulkan_data.load_folder("models/planet/shallow_water".parse().unwrap()); //Dummy objects to fill offsets
        vulkan_data.load_folder("models/planet/foliage".parse().unwrap());
        vulkan_data.load_folder("models/planet/desert".parse().unwrap());
        vulkan_data.load_folder("models/planet/mountain".parse().unwrap());
        vulkan_data.load_folder("models/planet/snow".parse().unwrap());
        objects[planet_index].position = Vector3::new(0.0,0.0,0.0);
        objects[planet_index].rotation *= UnitQuaternion::from_euler_angles(23.43644f64.to_radians(),0.0,0.0);

        vulkan_data.objects[objects[planet_index].render_object_index].is_globe = true;

        vulkan_data.update_vertex_and_index_buffers();

        let mut game = Game {
            game_start: std::time::Instant::now(),
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
        // let paths = std::fs::read_dir("../GSG/elevation").unwrap();
        let paths = std::fs::read_dir("../GSG/elevation_gebco").unwrap();


        let vertex_count = self.vulkan_data.objects[self.objects[self.planet_index].render_object_index].vertex_count as usize;
        let vertex_start = self.vulkan_data.objects[self.objects[self.planet_index].render_object_index].vertex_start as usize;
        let _index_count = self.vulkan_data.objects[self.objects[self.planet_index].render_object_index].index_count as usize;
        let _index_start = self.vulkan_data.objects[self.objects[self.planet_index].render_object_index].index_start as usize;

        let mut unprocessed_vertices_elevation = std::iter::repeat(
            RawVertexData { sum: 0.0, sample_count: 0, data_priority: 0 })
            .take(vertex_count)
            .collect::<Vec<_>>();

        let bincode_config = bincode::config::Configuration::standard();
        let wgs_84 = gdal::spatial_ref::SpatialRef::from_epsg(4326).unwrap();

        let vertex_data = match std::fs::read(Path::new("vertex_elevations.bin")){
            Ok(bytes) => {
                 let in_data: Vec<VertexData> = bincode::decode_from_slice(&bytes, bincode_config).unwrap();
                assert_eq!(in_data.len(), vertex_count);
                in_data
            }
            Err(_) => {
                for folder in paths{
                    let data_priority = if folder.as_ref().unwrap().file_name().to_str().unwrap() == "arctic" {1} else {0};
                    println!("priority: {:}", data_priority);
                    let mut images = std::fs::read_dir(folder.unwrap().path()).unwrap();
                    let path = images.find(|file|{
                        file.as_ref().unwrap().file_name().to_str().unwrap().to_lowercase().contains("mea")
                    }).unwrap().unwrap().path();
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
                    println!("transform: {:?}", transform);
                    println!("inv transform: {:?}", inv_transform);


                    let rasterband = dataset.rasterband(1).unwrap();
                    let no_data_value = rasterband.no_data_value().unwrap();

                    let spatial_ref = dataset.spatial_ref().unwrap();

                    println!("Spatial Ref Authority: {:}", spatial_ref.authority().unwrap());
                        // let stats = rasterband.get_statistics(false, true).unwrap();
                    // let elevation_difference = stats.max - stats.min;
                    let mut instant = std::time::Instant::now();
                    for (vertex_index, vertex) in self.vulkan_data.vertices[vertex_start..][..vertex_count].iter().enumerate() {
                        let time_elapsed = instant.elapsed().as_secs_f64();
                        if time_elapsed > 1.0{
                            instant = std::time::Instant::now();
                            println!("Percent done file: {:}", vertex_index as f64 / vertex_count as f64);
                        };
                        let latitude = ((vertex.position.y / RADIUS as f32).asin() as f64).to_degrees();
                        let longitude = (vertex.position.z.atan2(vertex.position.x) as f64).to_degrees();
                        // println!("Pre Transform: {:}" , vertex.position);
                        let transformed_coords = if spatial_ref.name().unwrap() != wgs_84.name().unwrap(){
                            let coord_transform = gdal::spatial_ref::CoordTransform::new(&wgs_84, &spatial_ref).unwrap();
                            let mut x = [latitude];
                            let mut y = [longitude];
                            let mut z = [];
                            match coord_transform.transform_coords(&mut x,&mut y,&mut z){
                                Ok(_) => {}
                                Err(error) => {
                                    match error {
                                        GdalError::InvalidCoordinateRange { .. } => {}
                                        _ => {
                                            println!("Unknown Transform Coords error: {:?}", error);
                                        }
                                    }
                                    continue;
                                }
                            }
                            (x[0], y[0])
                        }else{
                            (latitude, longitude)
                        };
                        let x= (inv_transform[0] + transformed_coords.1 * inv_transform[1] + transformed_coords.0 * inv_transform[2]);
                        let y= (inv_transform[3] + transformed_coords.1 * inv_transform[4] + transformed_coords.0 * inv_transform[5]);

                        if x >= 0.0 && x < rasterband.size().0 as f64 && y >= 0.0 && y < rasterband.size().1 as f64 {
                            let mut slice = [0f64];

                            // println!("x: {:}, y: {:}", x, y);
                            match rasterband.read_into_slice((x as isize, y as isize),
                                                             (1, 1),
                                                             (1, 1),
                                                             &mut slice,
                                                             Some(gdal::raster::ResampleAlg::Bilinear))
                            {
                                Err(error) => {println!("GDAL Error: {:?}", error)}
                                Ok(..) => {
                                    if slice[0] != no_data_value && data_priority >= unprocessed_vertices_elevation[vertex_index].data_priority{
                                        if data_priority > unprocessed_vertices_elevation[vertex_index].data_priority{
                                            unprocessed_vertices_elevation[vertex_index].sum = 0.0;
                                            unprocessed_vertices_elevation[vertex_index].sample_count = 0;
                                            unprocessed_vertices_elevation[vertex_index].data_priority = data_priority;

                                        }
                                        // println!("high priority elevation data: {:}", slice[0]);
                                        unprocessed_vertices_elevation[vertex_index].sum += slice[0];
                                        unprocessed_vertices_elevation[vertex_index].sample_count += 1;
                                    }
                                }
                            }
                        }
                    }

                }
                let aridity_dataset = gdal::Dataset::open(Path::new("../GSG/aridity/ai_et0.tif")).unwrap();
                let aridity_raster_band = aridity_dataset.rasterband(1).unwrap();
                let aridity_no_data_value = aridity_raster_band.no_data_value().unwrap();

                let population_dataset = gdal::Dataset::open(Path::new("../GSG/population/gpw_v4_population_density_rev11_2020_30_sec.tif")).unwrap();
                let population_raster_band = population_dataset.rasterband(1).unwrap();
                let population_no_data_value = population_raster_band.no_data_value().unwrap();

                let mut aridity_transform = aridity_dataset.geo_transform().unwrap();
                let mut aridity_inv_transform = [0.0f64; 6];
                unsafe {
                    assert_eq!(gdal_sys::GDALInvGeoTransform(
                        aridity_transform.as_mut_ptr(),
                        aridity_inv_transform.as_mut_ptr(),
                    ),1,"InvGeoTransform failed");
                }
                let mut population_transform = population_dataset.geo_transform().unwrap();
                let mut population_inv_transform = [0.0f64; 6];
                unsafe {
                    assert_eq!(gdal_sys::GDALInvGeoTransform(
                        population_transform.as_mut_ptr(),
                        population_inv_transform.as_mut_ptr(),
                    ),1,"InvGeoTransform failed");
                }

                let mut unprocessed_vertex_aridity = std::iter::repeat(
                    RawVertexData { sum: 0.0, sample_count: 0, data_priority: 0 })
                    .take(vertex_count)
                    .collect::<Vec<_>>();
                let mut unprocessed_vertex_population = std::iter::repeat(
                    RawVertexData { sum: 0.0, sample_count: 0, data_priority: 0 })
                    .take(vertex_count)
                    .collect::<Vec<_>>();

                let unprocessed_warm_temps = self.load_raster_file(Path::new("../GSG/temperature5/HotFilled.tif"), vertex_start, vertex_count, |value, index|{
                    // let elevation = if unprocessed_vertices_elevation[index].sample_count > 0 {
                    //     unprocessed_vertices_elevation[index].sum / unprocessed_vertices_elevation[index].sample_count as f64} else {0.0};
                    // let elevation = elevation.abs();
                    // let elevation = 1.0 - (elevation / 2000.0).clamp(0.0,1.0);
                    let latitude = (self.vulkan_data.vertices[vertex_start + index].position.y / RADIUS as f32).asin() as f64;
                    return value + GLOBAL_TEMPERATURE_CHANGE * (1.0 + latitude.abs().sin()*(POLAR_AMPLIFICATION_FACTOR - 1.0));
                });
                let unprocessed_cold_temps = self.load_raster_file(Path::new("../GSG/temperature5/ColdFilled.tif"), vertex_start, vertex_count, |value, index|{
                    // let elevation = if unprocessed_vertices_elevation[index].sample_count > 0 {
                    //     unprocessed_vertices_elevation[index].sum / unprocessed_vertices_elevation[index].sample_count as f64} else {0.0};
                    // let elevation = elevation.abs();
                    // let elevation = 1.0 - (elevation / 2000.0).clamp(0.0,1.0);

                    let latitude = (self.vulkan_data.vertices[vertex_start + index].position.y / RADIUS as f32).asin() as f64;
                    return value + GLOBAL_TEMPERATURE_CHANGE * (1.0 + latitude.abs().sin()*(POLAR_AMPLIFICATION_FACTOR - 1.0));
                });


                let mut instant = std::time::Instant::now();
                for i in 0..vertex_count{
                    let time_elapsed = instant.elapsed().as_secs_f64();
                    if time_elapsed > 1.0{
                        instant = std::time::Instant::now();
                        println!("Percent done file: {:}", i as f64 / vertex_count as f64);
                    };
                    let latitude = ((self.vulkan_data.vertices[vertex_start + i].position.y / RADIUS as f32).asin() as f64).to_degrees();
                    let longitude = (self.vulkan_data.vertices[vertex_start + i].position.z.atan2(self.vulkan_data.vertices[vertex_start + i].position.x) as f64).to_degrees();

                    let aridity_x= (aridity_inv_transform[0] + longitude * aridity_inv_transform[1] + latitude * aridity_inv_transform[2]);
                    let aridity_y= (aridity_inv_transform[3] + longitude * aridity_inv_transform[4] + latitude * aridity_inv_transform[5]);

                    let population_x= (population_inv_transform[0] + longitude * population_inv_transform[1] + latitude * population_inv_transform[2]);
                    let population_y= (population_inv_transform[3] + longitude * population_inv_transform[4] + latitude * population_inv_transform[5]);

                    if aridity_x >= 0.0 && aridity_x < aridity_raster_band.size().0 as f64 && aridity_y >= 0.0 && aridity_y < aridity_raster_band.size().1 as f64 {
                        let mut aridity_slice = [0f64];
                        match aridity_raster_band.read_into_slice((aridity_x as isize, aridity_y as isize),
                                                                  (1, 1),
                                                                  (1, 1),
                                                                  &mut aridity_slice,
                                                                  Some(gdal::raster::ResampleAlg::Bilinear))
                        {
                            Err(error) => { println!("GDAL Error: {:?}", error) }
                            Ok(..) => {
                                if aridity_slice[0] != aridity_no_data_value {
                                    unprocessed_vertex_aridity[i].sum += aridity_slice[0] / 10_000.0;
                                    unprocessed_vertex_aridity[i].sample_count += 1;
                                }
                            }
                        }
                    }
                    if population_x >= 0.0 && population_x < population_raster_band.size().0 as f64 && population_y >= 0.0 && population_y < population_raster_band.size().1 as f64 {
                        let mut population_slice = [0f64];
                        match population_raster_band.read_into_slice((population_x as isize, population_y as isize),
                                                                     (1, 1),
                                                                     (1, 1),
                                                                     &mut population_slice,
                                                                     Some(gdal::raster::ResampleAlg::Bilinear))
                        {
                            Err(error) => { println!("GDAL Error: {:?}", error) }
                            Ok(..) => {
                                if population_slice[0] != population_no_data_value {
                                    unprocessed_vertex_population[i].sum += population_slice[0] * (EARTH_SURFACE_AREA / vertex_count as f64);
                                    unprocessed_vertex_population[i].sample_count += 1;
                                }
                            }
                        }
                    }

                }

                let mut out_data = vec![];
                for i in 0..vertex_count {
                    out_data.push(VertexData {
                        elevation: if unprocessed_vertices_elevation[i].sample_count > 0 {
                            unprocessed_vertices_elevation[i].sum / unprocessed_vertices_elevation[i].sample_count as f64
                        } else { 0.0 },
                        aridity: if unprocessed_vertex_aridity[i].sample_count > 0 {
                            unprocessed_vertex_aridity[i].sum / unprocessed_vertex_aridity[i].sample_count as f64
                        } else { 0.0 },
                        population: if unprocessed_vertex_population[i].sample_count > 0 {
                            unprocessed_vertex_population[i].sum / unprocessed_vertex_population[i].sample_count as f64
                        } else { 0.0 },
                        warmest_temperature: if unprocessed_warm_temps[i].sample_count > 0 {
                            unprocessed_warm_temps[i].sum / unprocessed_warm_temps[i].sample_count as f64
                        } else { 0.0 },
                        coolest_temperature: if unprocessed_cold_temps[i].sample_count > 0 {
                            unprocessed_cold_temps[i].sum / unprocessed_cold_temps[i].sample_count as f64
                        } else { 0.0 }
                    });
                }
                std::fs::write("vertex_elevations.bin", bincode::encode_to_vec(&out_data, bincode_config).unwrap()).unwrap();
                out_data
            }
        };
        for i in 0..vertex_data.len(){
            let elevation_exaggeration = 1.0;
            let elevation = (vertex_data[i].elevation - NEW_SEA_LEVEL) * elevation_exaggeration;
            let aridity = vertex_data[i].aridity;
            let population = vertex_data[i].population;
            self.vulkan_data.vertices[vertex_start + i].elevation = elevation as f32;
            self.vulkan_data.vertices[vertex_start + i].aridity = aridity as f32;
            self.vulkan_data.vertices[vertex_start + i].population = population as f32;
            self.vulkan_data.vertices[vertex_start + i].warmest_temperature = vertex_data[i].warmest_temperature as f32;
            self.vulkan_data.vertices[vertex_start + i].coldest_temperature = vertex_data[i].coolest_temperature as f32;
            self.vulkan_data.vertices[vertex_start + i].position = self.vulkan_data.vertices[vertex_start + i].position + self.vulkan_data.vertices[vertex_start + i].normal*(elevation.max(0.0) as f32);

            self.vulkan_data.vertices[vertex_start + i].normal = Vector3::zeros();
        }
        for triangle_index in (_index_start..(_index_start + _index_count)).step_by(3){
            let edge1 = self.vulkan_data.vertices[self.vulkan_data.indices[triangle_index] as usize].position - self.vulkan_data.vertices[self.vulkan_data.indices[triangle_index + 1] as usize].position;
            let edge2 = self.vulkan_data.vertices[self.vulkan_data.indices[triangle_index] as usize].position - self.vulkan_data.vertices[self.vulkan_data.indices[triangle_index + 2] as usize].position;
            let weighted_normal = (edge1.cross(&edge2));
            for i in 0..3{
                self.vulkan_data.vertices[self.vulkan_data.indices[triangle_index + i] as usize].normal += weighted_normal
            }
        }
        for i in 0..vertex_count{
            self.vulkan_data.vertices[vertex_start + i].normal = self.vulkan_data.vertices[vertex_start + i].normal.normalize();
        }
        self.vulkan_data.update_vertex_and_index_buffers();
    }

    fn load_raster_file<F>(&self, path: &Path, vertex_start: usize, vertex_count: usize, additional_processing: F) -> Vec<RawVertexData>
    where F: Fn(f64, usize)-> f64
    {
        println!("Loading file: {:?}", path);
        let mut output = std::iter::repeat(
            RawVertexData { sum: 0.0, sample_count: 0, data_priority: 0 })
            .take(vertex_count)
            .collect::<Vec<_>>();

        let dataset = gdal::Dataset::open(path).unwrap();
        let raster_band = dataset.rasterband(1).unwrap();
        let no_data_value = raster_band.no_data_value().unwrap();

        println!("No data value: {:}", no_data_value);

        let mut transform = dataset.geo_transform().unwrap();
        let mut inv_transform = [0.0f64; 6];
        unsafe {
            assert_eq!(gdal_sys::GDALInvGeoTransform(
                transform.as_mut_ptr(),
                inv_transform.as_mut_ptr(),
            ),1,"InvGeoTransform failed");
        }
        let mut instant = std::time::Instant::now();
        for i in 0..vertex_count {
            let time_elapsed = instant.elapsed().as_secs_f64();
            if time_elapsed > 1.0 {
                instant = std::time::Instant::now();
                println!("Percent done file: {:}", i as f64 / vertex_count as f64);
            };

            let latitude = ((self.vulkan_data.vertices[vertex_start + i].position.y / RADIUS as f32).asin() as f64).to_degrees();
            let longitude = (self.vulkan_data.vertices[vertex_start + i].position.z.atan2(self.vulkan_data.vertices[vertex_start + i].position.x) as f64).to_degrees();

            let x = (inv_transform[0] + longitude * inv_transform[1] + latitude * inv_transform[2]);
            let y = (inv_transform[3] + longitude * inv_transform[4] + latitude * inv_transform[5]);
            if x >= 0.0 && x < raster_band.size().0 as f64 && y >= 0.0 && y < raster_band.size().1 as f64 {
                let mut slice = [0f64];
                match raster_band.read_into_slice((x as isize, y as isize),
                                                          (1, 1),
                                                          (1, 1),
                                                          &mut slice,
                                                          Some(gdal::raster::ResampleAlg::Bilinear))
                {
                    Err(error) => { println!("GDAL Error: {:?}", error) }
                    Ok(..) => {
                        if slice[0] != no_data_value && !slice[0].is_nan(){
                            output[i].sum += additional_processing(slice[0], i);
                            output[i].sample_count += 1;
                        }
                    }
                }

            }

        }

        return output;
    }

    fn get_year(&self) -> f64{
        self.game_start.elapsed().as_secs_f64() / 6000.0
        // self.game_start.elapsed().as_secs_f64() / 1.0
    }

    pub(crate) fn process(&mut self) {
        let delta_time = self.last_frame_instant.elapsed().as_secs_f64();
        self.last_frame_instant = std::time::Instant::now();

        for i in 0..self.objects.len() {
            self.objects[i].process(delta_time);
        }
        self.objects[self.planet_index].rotation = UnitQuaternion::from_euler_angles(23.43644f64.to_radians(),0.0 ,0.0) * UnitQuaternion::from_euler_angles(0.0,-std::f64::consts::PI * 2.0 * 365.25 * self.get_year() ,0.0);
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
        self.vulkan_data.uniform_buffer_object.time = self.get_year() as f32;
        self.vulkan_data.uniform_buffer_object.map_mode = self.inputs.map_mode as u32;
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
