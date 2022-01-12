use std::f64::consts::PI;
use crate::renderer::*;
use gdal::errors::GdalError;
use nalgebra::{Matrix4, Point3, Rotation3, Translation3, UnitQuaternion, Vector2, Vector3};
use std::path::Path;
use std::time::Instant;
use winit::window::Window;
use crate::pop::{Education, Identifier, Job, PopSlices};
use strum::{IntoEnumIterator};
use crate::market::{Good, Market};
use crate::organization::Organization;

const RADIUS: f64 = 6_371_000.0;
const NEW_SEA_LEVEL: f64 = 76.0;
const EARTH_SURFACE_AREA: f64 = 510_000_000.0;
const GLOBAL_TEMPERATURE_CHANGE: f64 = 10.0; //degrees K
const POLAR_AMPLIFICATION_FACTOR: f64 = 3.0;
const NUM_PROVINCES: usize = 1 << 14;

const SIMULATION_DELAY: f64 = 1.0;

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
    latitude: f64,
    longitude: f64
}
impl Camera {
    fn new() -> Self {

        Self {latitude: 0.0, longitude: 0.0 }
    }
    fn get_rotation(&self) -> UnitQuaternion<f64>{
        return UnitQuaternion::face_towards(&(self.get_position()), &Vector3::new(0.0, -1.0, 0.0))
    }
    fn get_position(&self) -> Vector3<f64>{
        return UnitQuaternion::from_euler_angles(0.0, self.longitude, 0.0)*
        UnitQuaternion::from_euler_angles(0.0, 0.0, self.latitude)
            * Vector3::new(10_000_000.0,0.0,0.0);
    }
}

#[derive(Copy, Clone)]
struct RawVertexData {
    sum: f64,
    sample_count: usize,
    data_priority: usize,
}
#[derive(bincode::Encode, bincode::Decode, Copy, Clone)]
struct VertexData {
    elevation: f64,
    aridity: f64,
    population: f64,
    warmest_temperature: f64,
    coolest_temperature: f64,
    province_id: Option<usize>, //index in province vec
}

pub(crate) struct Province {
    pub(crate) index_of_vertices: Vec<usize>,
    pub(crate) pop_slices: Option<PopSlices>,
    pub(crate) market: Market,
}
impl Province{
    pub(crate) fn get_population(&self) -> f64{
        let mut output = 0.0;

        for education in Education::iter(){
            for job in Job::iter(){
                let identifer = Identifier{
                    education,
                    job
                };
                output += self.pop_slices.as_ref().unwrap()[identifer].population;
            }
        }
        return output;
    }

    fn process(&mut self, delta_year: f64){
        for good in Good::iter(){
            let supply = self.pop_slices.as_ref().unwrap().get_good_amount(good);
            let demand = self.pop_slices.as_ref().unwrap().get_need_amount(good);
            self.market[good].supply = supply;
            self.market[good].demand = demand;
        }
        self.pop_slices.as_mut().unwrap().process(delta_year);
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
    planet_index: usize,
    vertex_data: Vec<VertexData>,
    pub(crate) provinces: Vec<Province>,
    pub(crate) selected_province: Option<usize>,
    pub(crate) organizations: Vec<Organization>,
    last_sim_year: f64,
    last_sim_tick: Instant,
}

impl Game {
    pub(crate) fn new(window: &Window) -> Self {
        let mut objects = vec![];

        let mut vulkan_data = VulkanData::new();
        vulkan_data.init_vulkan(&window);

        let planet_index = objects.len();
        objects.push(GameObject::new(
            vulkan_data.load_folder("models/planet".parse().unwrap()),
        ));

        vulkan_data.load_folder("models/planet/deep_water".parse().unwrap()); //Dummy objects to fill offsets
        vulkan_data.load_folder("models/planet/shallow_water".parse().unwrap()); //Dummy objects to fill offsets
        vulkan_data.load_folder("models/planet/foliage".parse().unwrap());
        vulkan_data.load_folder("models/planet/desert".parse().unwrap());
        vulkan_data.load_folder("models/planet/mountain".parse().unwrap());
        vulkan_data.load_folder("models/planet/snow".parse().unwrap());
        objects[planet_index].position = Vector3::new(0.0, 0.0, 0.0);
        objects[planet_index].rotation *=
            UnitQuaternion::from_euler_angles(23.43644f64.to_radians(), 0.0, 0.0);

        vulkan_data.objects[objects[planet_index].render_object_index].is_globe = true;

        vulkan_data.update_vertex_and_index_buffers();

        let mut game = Game {
            game_start: std::time::Instant::now(),
            objects,
            mouse_position: Vector2::new(0.0, 0.0),
            last_mouse_position: Vector2::new(0.0,0.0),
            inputs: Inputs::new(),
            focused: false,
            last_frame_instant: Instant::now(),
            vulkan_data,
            camera: Camera::new(),
            planet_index,
            vertex_data: vec![],
            // provinces: vec![
            //     Province {
            //         index_of_vertices: vec![],
            //         pop_slices: None,
            //     };
            //     NUM_PROVINCES
            // ],
            provinces: vec![],
            selected_province: None,
            organizations: vec![
                Organization::new("Test Organization", 10000.0)
            ],
            last_sim_year: 0.0,
            last_sim_tick: Instant::now(),
        };
        game.load_planet_data();
        return game;
    }

    // pub(crate) fn get_province_population(&self, province_index: usize)-> f64{
    //     let mut population_sum = 0.0;
    //     for vertex in &self.provinces[province_index].index_of_vertices{
    //         population_sum += self.vertex_data[*vertex].population
    //     }
    //     return population_sum;
    // }

    fn load_planet_data(&mut self) {
        // let paths = std::fs::read_dir("../GSG/elevation").unwrap();
        let paths = std::fs::read_dir("../GSG/elevation_gebco").unwrap();

        let vertex_count = self.vulkan_data.objects
            [self.objects[self.planet_index].render_object_index]
            .vertex_count as usize;
        let vertex_start = self.vulkan_data.objects
            [self.objects[self.planet_index].render_object_index]
            .vertex_start as usize;
        let index_count = self.vulkan_data.objects
            [self.objects[self.planet_index].render_object_index]
            .index_count as usize;
        let index_start = self.vulkan_data.objects
            [self.objects[self.planet_index].render_object_index]
            .index_start as usize;

        let mut unprocessed_vertices_elevation = std::iter::repeat(RawVertexData {
            sum: 0.0,
            sample_count: 0,
            data_priority: 0,
        })
        .take(vertex_count)
        .collect::<Vec<_>>();

        let bincode_config = bincode::config::Configuration::standard();
        let wgs_84 = gdal::spatial_ref::SpatialRef::from_epsg(4326).unwrap();

        let mut vertex_data = match std::fs::read(Path::new("vertex_elevations.bin")) {
            Ok(bytes) => {
                let in_data: Vec<VertexData> =
                    bincode::decode_from_slice(&bytes, bincode_config).unwrap();
                assert_eq!(in_data.len(), vertex_count);
                in_data
            }
            Err(_) => {
                for folder in paths {
                    let data_priority =
                        if folder.as_ref().unwrap().file_name().to_str().unwrap() == "arctic" {
                            1
                        } else {
                            0
                        };
                    println!("priority: {:}", data_priority);
                    let mut images = std::fs::read_dir(folder.unwrap().path()).unwrap();
                    let path = images
                        .find(|file| {
                            file.as_ref()
                                .unwrap()
                                .file_name()
                                .to_str()
                                .unwrap()
                                .to_lowercase()
                                .contains("mea")
                        })
                        .unwrap()
                        .unwrap()
                        .path();
                    println!("Path: {:?}", path);
                    let dataset = gdal::Dataset::open(path).unwrap();
                    let mut transform = dataset.geo_transform().unwrap();
                    let mut inv_transform = [0.0f64; 6];
                    unsafe {
                        assert_eq!(
                            gdal_sys::GDALInvGeoTransform(
                                transform.as_mut_ptr(),
                                inv_transform.as_mut_ptr(),
                            ),
                            1,
                            "InvGeoTransform failed"
                        );
                    }
                    println!("transform: {:?}", transform);
                    println!("inv transform: {:?}", inv_transform);

                    let rasterband = dataset.rasterband(1).unwrap();
                    let no_data_value = rasterband.no_data_value().unwrap();

                    let spatial_ref = dataset.spatial_ref().unwrap();

                    println!(
                        "Spatial Ref Authority: {:}",
                        spatial_ref.authority().unwrap()
                    );
                    // let stats = rasterband.get_statistics(false, true).unwrap();
                    // let elevation_difference = stats.max - stats.min;
                    let mut instant = std::time::Instant::now();
                    for (vertex_index, vertex) in self.vulkan_data.vertices[vertex_start..]
                        [..vertex_count]
                        .iter()
                        .enumerate()
                    {
                        let time_elapsed = instant.elapsed().as_secs_f64();
                        if time_elapsed > 1.0 {
                            instant = std::time::Instant::now();
                            println!(
                                "Percent done file: {:}",
                                vertex_index as f64 / vertex_count as f64
                            );
                        };
                        let latitude =
                            ((vertex.position.y / RADIUS as f32).asin() as f64).to_degrees();
                        let longitude =
                            (vertex.position.z.atan2(vertex.position.x) as f64).to_degrees();
                        // println!("Pre Transform: {:}" , vertex.position);
                        let transformed_coords = if spatial_ref.name().unwrap()
                            != wgs_84.name().unwrap()
                        {
                            let coord_transform =
                                gdal::spatial_ref::CoordTransform::new(&wgs_84, &spatial_ref)
                                    .unwrap();
                            let mut x = [latitude];
                            let mut y = [longitude];
                            let mut z = [];
                            match coord_transform.transform_coords(&mut x, &mut y, &mut z) {
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
                        } else {
                            (latitude, longitude)
                        };
                        let x = inv_transform[0]
                            + transformed_coords.1 * inv_transform[1]
                            + transformed_coords.0 * inv_transform[2];
                        let y = inv_transform[3]
                            + transformed_coords.1 * inv_transform[4]
                            + transformed_coords.0 * inv_transform[5];

                        if x >= 0.0
                            && x < rasterband.size().0 as f64
                            && y >= 0.0
                            && y < rasterband.size().1 as f64
                        {
                            let mut slice = [0f64];

                            // println!("x: {:}, y: {:}", x, y);
                            match rasterband.read_into_slice(
                                (x as isize, y as isize),
                                (1, 1),
                                (1, 1),
                                &mut slice,
                                Some(gdal::raster::ResampleAlg::Bilinear),
                            ) {
                                Err(error) => {
                                    println!("GDAL Error: {:?}", error)
                                }
                                Ok(..) => {
                                    if slice[0] != no_data_value
                                        && data_priority
                                            >= unprocessed_vertices_elevation[vertex_index]
                                                .data_priority
                                    {
                                        if data_priority
                                            > unprocessed_vertices_elevation[vertex_index]
                                                .data_priority
                                        {
                                            unprocessed_vertices_elevation[vertex_index].sum = 0.0;
                                            unprocessed_vertices_elevation[vertex_index]
                                                .sample_count = 0;
                                            unprocessed_vertices_elevation[vertex_index]
                                                .data_priority = data_priority;
                                        }
                                        // println!("high priority elevation data: {:}", slice[0]);
                                        unprocessed_vertices_elevation[vertex_index].sum +=
                                            slice[0];
                                        unprocessed_vertices_elevation[vertex_index]
                                            .sample_count += 1;
                                    }
                                }
                            }
                        }
                    }
                }
                let aridity_dataset =
                    gdal::Dataset::open(Path::new("../GSG/aridity/ai_et0.tif")).unwrap();
                let aridity_raster_band = aridity_dataset.rasterband(1).unwrap();
                let aridity_no_data_value = aridity_raster_band.no_data_value().unwrap();

                let population_dataset = gdal::Dataset::open(Path::new(
                    "../GSG/population/gpw_v4_population_density_rev11_2020_30_sec.tif",
                ))
                .unwrap();
                let population_raster_band = population_dataset.rasterband(1).unwrap();
                let population_no_data_value = population_raster_band.no_data_value().unwrap();

                let mut aridity_transform = aridity_dataset.geo_transform().unwrap();
                let mut aridity_inv_transform = [0.0f64; 6];
                unsafe {
                    assert_eq!(
                        gdal_sys::GDALInvGeoTransform(
                            aridity_transform.as_mut_ptr(),
                            aridity_inv_transform.as_mut_ptr(),
                        ),
                        1,
                        "InvGeoTransform failed"
                    );
                }
                let mut population_transform = population_dataset.geo_transform().unwrap();
                let mut population_inv_transform = [0.0f64; 6];
                unsafe {
                    assert_eq!(
                        gdal_sys::GDALInvGeoTransform(
                            population_transform.as_mut_ptr(),
                            population_inv_transform.as_mut_ptr(),
                        ),
                        1,
                        "InvGeoTransform failed"
                    );
                }

                let mut unprocessed_vertex_aridity = std::iter::repeat(RawVertexData {
                    sum: 0.0,
                    sample_count: 0,
                    data_priority: 0,
                })
                .take(vertex_count)
                .collect::<Vec<_>>();
                let mut unprocessed_vertex_population = std::iter::repeat(RawVertexData {
                    sum: 0.0,
                    sample_count: 0,
                    data_priority: 0,
                })
                .take(vertex_count)
                .collect::<Vec<_>>();

                let unprocessed_warm_temps = self.load_raster_file(
                    Path::new("../GSG/temperature5/HotFilled.tif"),
                    vertex_start,
                    vertex_count,
                    |value, index| {
                        // let elevation = if unprocessed_vertices_elevation[index].sample_count > 0 {
                        //     unprocessed_vertices_elevation[index].sum / unprocessed_vertices_elevation[index].sample_count as f64} else {0.0};
                        // let elevation = elevation.abs();
                        // let elevation = 1.0 - (elevation / 2000.0).clamp(0.0,1.0);
                        let latitude = (self.vulkan_data.vertices[vertex_start + index].position.y
                            / RADIUS as f32)
                            .asin() as f64;
                        return value
                            + GLOBAL_TEMPERATURE_CHANGE
                                * (1.0
                                    + latitude.abs().sin() * (POLAR_AMPLIFICATION_FACTOR - 1.0));
                    },
                );
                let unprocessed_cold_temps = self.load_raster_file(
                    Path::new("../GSG/temperature5/ColdFilled.tif"),
                    vertex_start,
                    vertex_count,
                    |value, index| {
                        // let elevation = if unprocessed_vertices_elevation[index].sample_count > 0 {
                        //     unprocessed_vertices_elevation[index].sum / unprocessed_vertices_elevation[index].sample_count as f64} else {0.0};
                        // let elevation = elevation.abs();
                        // let elevation = 1.0 - (elevation / 2000.0).clamp(0.0,1.0);

                        let latitude = (self.vulkan_data.vertices[vertex_start + index].position.y
                            / RADIUS as f32)
                            .asin() as f64;
                        return value
                            + GLOBAL_TEMPERATURE_CHANGE
                                * (1.0
                                    + latitude.abs().sin() * (POLAR_AMPLIFICATION_FACTOR - 1.0));
                    },
                );

                let mut instant = std::time::Instant::now();
                for i in 0..vertex_count {
                    let time_elapsed = instant.elapsed().as_secs_f64();
                    if time_elapsed > 1.0 {
                        instant = std::time::Instant::now();
                        println!("Percent done file: {:}", i as f64 / vertex_count as f64);
                    };
                    let latitude = ((self.vulkan_data.vertices[vertex_start + i].position.y
                        / RADIUS as f32)
                        .asin() as f64)
                        .to_degrees();
                    let longitude = (self.vulkan_data.vertices[vertex_start + i]
                        .position
                        .z
                        .atan2(self.vulkan_data.vertices[vertex_start + i].position.x)
                        as f64)
                        .to_degrees();

                    let aridity_x = aridity_inv_transform[0]
                        + longitude * aridity_inv_transform[1]
                        + latitude * aridity_inv_transform[2];
                    let aridity_y = aridity_inv_transform[3]
                        + longitude * aridity_inv_transform[4]
                        + latitude * aridity_inv_transform[5];

                    let population_x = population_inv_transform[0]
                        + longitude * population_inv_transform[1]
                        + latitude * population_inv_transform[2];
                    let population_y = population_inv_transform[3]
                        + longitude * population_inv_transform[4]
                        + latitude * population_inv_transform[5];

                    if aridity_x >= 0.0
                        && aridity_x < aridity_raster_band.size().0 as f64
                        && aridity_y >= 0.0
                        && aridity_y < aridity_raster_band.size().1 as f64
                    {
                        let mut aridity_slice = [0f64];
                        match aridity_raster_band.read_into_slice(
                            (aridity_x as isize, aridity_y as isize),
                            (1, 1),
                            (1, 1),
                            &mut aridity_slice,
                            Some(gdal::raster::ResampleAlg::Bilinear),
                        ) {
                            Err(error) => {
                                println!("GDAL Error: {:?}", error)
                            }
                            Ok(..) => {
                                if aridity_slice[0] != aridity_no_data_value {
                                    unprocessed_vertex_aridity[i].sum +=
                                        aridity_slice[0] / 10_000.0;
                                    unprocessed_vertex_aridity[i].sample_count += 1;
                                }
                            }
                        }
                    }
                    if population_x >= 0.0
                        && population_x < population_raster_band.size().0 as f64
                        && population_y >= 0.0
                        && population_y < population_raster_band.size().1 as f64
                    {
                        let mut population_slice = [0f64];
                        match population_raster_band.read_into_slice(
                            (population_x as isize, population_y as isize),
                            (1, 1),
                            (1, 1),
                            &mut population_slice,
                            Some(gdal::raster::ResampleAlg::Bilinear),
                        ) {
                            Err(error) => {
                                println!("GDAL Error: {:?}", error)
                            }
                            Ok(..) => {
                                if population_slice[0] != population_no_data_value {
                                    unprocessed_vertex_population[i].sum += population_slice[0]
                                        * (EARTH_SURFACE_AREA / vertex_count as f64);
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
                            unprocessed_vertices_elevation[i].sum
                                / unprocessed_vertices_elevation[i].sample_count as f64
                        } else {
                            0.0
                        },
                        aridity: if unprocessed_vertex_aridity[i].sample_count > 0 {
                            unprocessed_vertex_aridity[i].sum
                                / unprocessed_vertex_aridity[i].sample_count as f64
                        } else {
                            0.0
                        },
                        population: if unprocessed_vertex_population[i].sample_count > 0 {
                            unprocessed_vertex_population[i].sum
                                / unprocessed_vertex_population[i].sample_count as f64
                        } else {
                            0.0
                        },
                        warmest_temperature: if unprocessed_warm_temps[i].sample_count > 0 {
                            unprocessed_warm_temps[i].sum
                                / unprocessed_warm_temps[i].sample_count as f64
                        } else {
                            0.0
                        },
                        coolest_temperature: if unprocessed_cold_temps[i].sample_count > 0 {
                            unprocessed_cold_temps[i].sum
                                / unprocessed_cold_temps[i].sample_count as f64
                        } else {
                            0.0
                        },
                        province_id: None,
                    });
                }
                std::fs::write(
                    "vertex_elevations.bin",
                    bincode::encode_to_vec(&out_data, bincode_config).unwrap(),
                )
                .unwrap();
                out_data
            }
        };

        let mut population_underwater = 0.0;
        let mut land_population = 0.0;
        let mut land_vertex_data = vertex_data.clone();

        for vertex in &mut land_vertex_data {
            let elevation_exaggeration = 1.0;
            let elevation = (vertex.elevation - NEW_SEA_LEVEL) * elevation_exaggeration;

            vertex.elevation = elevation;
            let population = if elevation > 0.0{
                land_population += vertex.population;
                vertex.population
            }else{
                population_underwater += vertex.population;
                0.0
            };
            vertex.population = population;
        }
        let survive_ratio = 0.6;
        println!("Total population: {:}", population_underwater + land_population);
        println!("People dead: {:}", population_underwater * (1.0 - survive_ratio));
        let mut new_vertex_data = land_vertex_data.clone();
        for (index,vertex) in new_vertex_data.iter_mut().enumerate() {
            let population_ratio = (land_vertex_data[index].population / land_population).clamp(0.0,1.0);
            vertex.population += population_ratio * population_underwater * survive_ratio;
        }
        vertex_data = new_vertex_data;
        for i in 0..vertex_data.len() {
            let aridity = vertex_data[i].aridity;
            let population = vertex_data[i].population;
            let elevation = vertex_data[i].elevation;
            self.vulkan_data.vertices[vertex_start + i].elevation = elevation as f32;
            self.vulkan_data.vertices[vertex_start + i].aridity = aridity as f32;
            self.vulkan_data.vertices[vertex_start + i].population = population as f32;
            self.vulkan_data.vertices[vertex_start + i].warmest_temperature =
                vertex_data[i].warmest_temperature as f32;
            self.vulkan_data.vertices[vertex_start + i].coldest_temperature =
                vertex_data[i].coolest_temperature as f32;
            self.vulkan_data.vertices[vertex_start + i].position = self.vulkan_data.vertices
                [vertex_start + i]
                .position
                + self.vulkan_data.vertices[vertex_start + i].normal * (elevation.max(0.0) as f32);

            self.vulkan_data.vertices[vertex_start + i].normal = Vector3::zeros();
        }

        //Create provinces randomly
        let rng = fastrand::Rng::new();
        for _ in 0..NUM_PROVINCES {
            let index = rng.usize(0..vertex_data.len());
            let mut tries_left = 10u8;
            loop {
                if vertex_data[index].province_id.is_none() && vertex_data[index].elevation > 0.0 {
                    let new_province_index = self.provinces.len();
                    self.provinces.push(Province{
                        index_of_vertices: vec![],
                        pop_slices: None,
                        market: Market::new()
                    });
                    vertex_data[index].province_id = Some(new_province_index);
                    break;
                }
                tries_left -= 1;
                if tries_left <= 0 {
                    break;
                }
            }
        }

        for triangle_index in (index_start..(index_start + index_count)).step_by(3) {
            let edge1 = self.vulkan_data.vertices
                [self.vulkan_data.indices[triangle_index] as usize]
                .position
                - self.vulkan_data.vertices[self.vulkan_data.indices[triangle_index + 1] as usize]
                    .position;
            let edge2 = self.vulkan_data.vertices
                [self.vulkan_data.indices[triangle_index] as usize]
                .position
                - self.vulkan_data.vertices[self.vulkan_data.indices[triangle_index + 2] as usize]
                    .position;
            let weighted_normal = edge1.cross(&edge2);
            for i in 0..3 {
                self.vulkan_data.vertices[self.vulkan_data.indices[triangle_index + i] as usize]
                    .normal += weighted_normal
            }
        }

        for _ in 0..50 {
            let mut new_vertex_data = vertex_data.to_vec();
            for triangle_index in (index_start..(index_start + index_count)).step_by(3) {
                let index1 = self.vulkan_data.indices[triangle_index + 0] as usize - vertex_start;
                let index2 = self.vulkan_data.indices[triangle_index + 1] as usize - vertex_start;
                let index3 = self.vulkan_data.indices[triangle_index + 2] as usize - vertex_start;

                let new_province = if vertex_data[index1].province_id.is_some() {
                    vertex_data[index1].province_id.unwrap()
                } else if vertex_data[index2].province_id.is_some() {
                    vertex_data[index2].province_id.unwrap()
                } else if vertex_data[index3].province_id.is_some() {
                    vertex_data[index3].province_id.unwrap()
                } else {
                    continue;
                };

                for j in 0..3 {
                    if vertex_data
                        [self.vulkan_data.indices[triangle_index + j] as usize - vertex_start]
                        .province_id
                        .is_none()
                        && vertex_data
                            [self.vulkan_data.indices[triangle_index + j] as usize - vertex_start]
                            .elevation
                            > 0.0
                    {
                        new_vertex_data[self.vulkan_data.indices[triangle_index + j] as usize
                            - vertex_start]
                            .province_id = Some(new_province);
                    }
                }
            }
            vertex_data = new_vertex_data;
        }

        //unassigned vertices get assigned to new provinces
        for (index, vertex) in vertex_data.iter_mut().enumerate() {
            if vertex.province_id.is_none() && vertex.elevation > 0.0 {
                vertex.province_id = Some(self.provinces.len());
                self.provinces.push(Province {
                    index_of_vertices: vec![],
                    pop_slices: None,
                    market: Market::new()
                });
            }
        }

        for (i, vertex) in self.vulkan_data.vertices[vertex_start..][..vertex_count]
            .iter_mut()
            .enumerate()
        {
            vertex.province_id = match vertex_data[i].province_id {
                Some(id) => {
                    self.provinces[id].index_of_vertices.push(i);
                    id as u32
                }
                None => u32::MAX,
            };
        }

        //Create PopSlices for each province
        for province in &mut self.provinces{
            let population = {
                let mut population_sum = 0.0;
                for vertex in &province.index_of_vertices{
                    population_sum += vertex_data[*vertex].population
                }
                population_sum
            };
            province.pop_slices = Some(PopSlices::new(population, 1000.0));
        }



        for i in 0..vertex_count {
            self.vulkan_data.vertices[vertex_start + i].normal = self.vulkan_data.vertices
                [vertex_start + i]
                .normal
                .normalize();
        }
        self.vulkan_data.update_vertex_and_index_buffers();
        self.vertex_data = vertex_data;
    }

    fn load_raster_file<F>(
        &self,
        path: &Path,
        vertex_start: usize,
        vertex_count: usize,
        additional_processing: F,
    ) -> Vec<RawVertexData>
    where
        F: Fn(f64, usize) -> f64,
    {
        println!("Loading file: {:?}", path);
        let mut output = std::iter::repeat(RawVertexData {
            sum: 0.0,
            sample_count: 0,
            data_priority: 0,
        })
        .take(vertex_count)
        .collect::<Vec<_>>();

        let dataset = gdal::Dataset::open(path).unwrap();
        let raster_band = dataset.rasterband(1).unwrap();
        let no_data_value = raster_band.no_data_value().unwrap();

        println!("No data value: {:}", no_data_value);

        let mut transform = dataset.geo_transform().unwrap();
        let mut inv_transform = [0.0f64; 6];
        unsafe {
            assert_eq!(
                gdal_sys::GDALInvGeoTransform(transform.as_mut_ptr(), inv_transform.as_mut_ptr(),),
                1,
                "InvGeoTransform failed"
            );
        }
        let mut instant = std::time::Instant::now();
        for i in 0..vertex_count {
            let time_elapsed = instant.elapsed().as_secs_f64();
            if time_elapsed > 1.0 {
                instant = std::time::Instant::now();
                println!("Percent done file: {:}", i as f64 / vertex_count as f64);
            };

            let latitude =
                ((self.vulkan_data.vertices[vertex_start + i].position.y / RADIUS as f32).asin()
                    as f64)
                    .to_degrees();
            let longitude = (self.vulkan_data.vertices[vertex_start + i]
                .position
                .z
                .atan2(self.vulkan_data.vertices[vertex_start + i].position.x)
                as f64)
                .to_degrees();

            let x = inv_transform[0] + longitude * inv_transform[1] + latitude * inv_transform[2];
            let y = inv_transform[3] + longitude * inv_transform[4] + latitude * inv_transform[5];
            if x >= 0.0
                && x < raster_band.size().0 as f64
                && y >= 0.0
                && y < raster_band.size().1 as f64
            {
                let mut slice = [0f64];
                match raster_band.read_into_slice(
                    (x as isize, y as isize),
                    (1, 1),
                    (1, 1),
                    &mut slice,
                    Some(gdal::raster::ResampleAlg::Bilinear),
                ) {
                    Err(error) => {
                        println!("GDAL Error: {:?}", error)
                    }
                    Ok(..) => {
                        if slice[0] != no_data_value && !slice[0].is_nan() {
                            output[i].sum += additional_processing(slice[0], i);
                            output[i].sample_count += 1;
                        }
                    }
                }
            }
        }

        return output;
    }

    fn get_closest_vertex(&self, position: Vector3<f64>) -> usize{
        let vertex_start = self.vulkan_data.objects[self.planet_index].vertex_start as usize;
        let vertex_count = self.vulkan_data.objects[self.planet_index].vertex_count as usize;

        let mut closest_index = 0;
        let mut closest_distance = f64::INFINITY;

        for (i,vertex) in self.vulkan_data.vertices[vertex_start..][..vertex_count].iter().enumerate(){
            let distance = (vertex.position.cast() - position).magnitude();
            if distance < closest_distance{
                closest_distance = distance;
                closest_index = i;
            }
        }

        return closest_index;
    }

    fn intersect_planet(ray_origin: Vector3<f64>, ray_direction: Vector3<f64>) -> Option<Vector3<f64>>{
        let a = ray_direction.dot(&ray_direction);
        let b = 2.0 * ray_origin.dot(&ray_direction);
        let c = ray_origin.dot(&ray_origin) - (RADIUS * RADIUS);
        let discriminant = b*b - 4.0*a*c;
        if discriminant < 0.0{
            return None;
        }else{
            let ray_ratio = (-b - discriminant.sqrt()) / (2.0 * a);
            return Some(ray_origin + ray_direction * ray_ratio);
        }
    }

    fn get_year(&self) -> f64 {
        self.game_start.elapsed().as_secs_f64() / 36000.0
        // self.game_start.elapsed().as_secs_f64() / 4.0
    }

    pub(crate) fn process(&mut self) {
        let delta_time = self.last_frame_instant.elapsed().as_secs_f64();
        self.last_frame_instant = std::time::Instant::now();
        let delta_mouse = self.last_mouse_position - self.mouse_position;
        self.last_mouse_position = self.mouse_position;

        for i in 0..self.objects.len() {
            self.objects[i].process(delta_time);
        }
        self.objects[self.planet_index].rotation =
            UnitQuaternion::from_euler_angles(23.43644f64.to_radians(), 0.0, 0.0)
                * UnitQuaternion::from_euler_angles(
                    0.0,
                    -std::f64::consts::PI * 2.0 * 365.25 * self.get_year(),
                    0.0,
                );
        if self.inputs.panning{
            self.camera.latitude = (self.camera.latitude - 0.001*delta_mouse.y).clamp(-PI/2.01, PI/2.01);
            self.camera.longitude -= delta_mouse.x *0.001;
        }

        if self.inputs.left_click{
            // println!("click!");
            self.inputs.left_click = false;
            let projection = self.vulkan_data.get_projection(self.inputs.zoom);

            let planet_transform =
                Matrix4::from(Translation3::from(self.objects[self.planet_index].position))
                    * self.objects[self.planet_index].rotation.to_homogeneous();

            let view_matrix = (planet_transform
                * Matrix4::from(Translation3::from(self.camera.get_position()))
                * self.camera.get_rotation().to_homogeneous())
                .try_inverse()
                .unwrap();
            let view_matrix_no_translation =
                (self.objects[self.planet_index].rotation.to_homogeneous()
                    * self.camera.get_rotation().to_homogeneous())
                    .try_inverse()
                    .unwrap();

            let origin = view_matrix.try_inverse().unwrap().transform_point(&Point3::from(Vector3::zeros())).coords;
            // dbg!(origin);
            let mouse_screen_space = Vector3::new(
                ((self.mouse_position.x/self.vulkan_data.surface_capabilities.unwrap().current_extent.width as f64) * 2.0 -1.0)* 1.0,
                ((self.mouse_position.y/self.vulkan_data.surface_capabilities.unwrap().current_extent.height as f64) * 2.0 -1.0) * 1.0,1.0);

            let direction = view_matrix_no_translation.try_inverse().unwrap().transform_vector(&projection.unproject_point(&Point3::from(mouse_screen_space)).coords).normalize();
            // dbg!(direction);
            self.selected_province = match Self::intersect_planet(origin, direction){
                None => {None}
                Some(position) => {
                    self.vertex_data[self.get_closest_vertex(planet_transform.try_inverse().unwrap().transform_vector(&position))].province_id
                }
            }

        }

        if self.last_sim_tick.elapsed().as_secs_f64() > SIMULATION_DELAY {
            self.last_sim_tick = Instant::now();
            let delta_year = self.get_year() - self.last_sim_year;
            self.last_sim_year = self.get_year();
            for province in &mut self.provinces{
                province.process(delta_year);
            }
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

        let planet_transform =
            Matrix4::from(Translation3::from(self.objects[self.planet_index].position))
                * self.objects[self.planet_index].rotation.to_homogeneous();

        let view_matrix = (planet_transform
            * Matrix4::from(Translation3::from(self.camera.get_position()))
            * self.camera.get_rotation().to_homogeneous())
        .try_inverse()
        .unwrap();
        let view_matrix_no_translation =
            ((self.objects[self.planet_index].rotation.to_homogeneous()
                * self.camera.get_rotation().to_homogeneous())
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
            if i == self.planet_index {
                self.vulkan_data.uniform_buffer_object.planet_model_matrix = model_matrix;
                self.vulkan_data.uniform_buffer_object.b = 69.0;
                self.vulkan_data.uniform_buffer_object.c = 42.0;
                self.vulkan_data.uniform_buffer_object.d = 122.0;
            }
            self.vulkan_data.objects[self.objects[i].render_object_index].view = view_matrix.cast();
            self.vulkan_data.objects[self.objects[i].render_object_index].proj = projection_matrix.cast();
        }

        self.vulkan_data.uniform_buffer_object.selected_province = match self.selected_province{
            None => {u32::MAX}
            Some(province_index) => {province_index as u32}
        };

        self.vulkan_data.uniform_buffer_object.time = self.get_year() as f32;
        self.vulkan_data.uniform_buffer_object.map_mode = self.inputs.map_mode as u32;
        self.vulkan_data.uniform_buffer_object.mouse_position = Vector2::new(
            (self.mouse_position.x) as f32,
            (self.mouse_position.y) as f32,
        );
    }
}
