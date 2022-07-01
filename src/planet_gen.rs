use crate::{
    support::{coordinate_to_index, index_to_coordinate, Vertex, CUBEMAP_WIDTH},
    world::World,
};
use float_ord::FloatOrd;
use gdal::raster::GdalType;
use genmesh::generators::{IcoSphere, IndexedPolygon, SharedVertex};
use nalgebra::{Vector2, Vector3, Vector4};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fmt::Debug,
    fs::read_dir,
    ops::{Add, AddAssign, Mul},
    path::{Path, PathBuf},
};

pub struct MeshOutput {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<usize>,
}

#[derive(Copy, Clone)]
struct RawVertexData {
    sum: f64,
    sample_count: usize,
    data_priority: usize,
}

#[derive(Clone, Debug)]
struct VertexData {
    elevation: f32,
    quantized_elevation: i8,
    neighbours: Vec<usize>,
}

const KELVIN_TO_CELSIUS: f64 = -273.15;

//https://answers.unity.com/questions/383804/calculate-uv-coordinates-of-3d-point-on-plane-of-m.html
pub fn triangle_interpolate<T>(
    target_point: Vector3<f32>,
    first_point: (Vector3<f32>, T),
    second_point: (Vector3<f32>, T),
    third_point: (Vector3<f32>, T),
) -> Option<T>
where
    T: Mul<f32, Output = T> + Add<Output = T> + Copy,
{
    let f1 = target_point - first_point.0;
    let f2 = target_point - second_point.0;
    let f3 = target_point - third_point.0;

    let main_cross = (first_point.0 - second_point.0).cross(&(first_point.0 - third_point.0));
    let cross1 = f2.cross(&f3);
    let cross2 = f3.cross(&f1);
    let cross3 = f1.cross(&f2);

    let total_area = main_cross.magnitude();
    let ratio1 = cross1.magnitude() / total_area * main_cross.dot(&cross1).signum();
    let ratio2 = cross2.magnitude() / total_area * main_cross.dot(&cross2).signum();
    let ratio3 = cross3.magnitude() / total_area * main_cross.dot(&cross3).signum();

    if ratio1 >= -0.1
        && ratio2 >= -0.1
        && ratio3 >= -0.1
        && ratio1 <= 1.0
        && ratio2 <= 1.0
        && ratio3 <= 1.0
    {
        Some(first_point.1 * ratio1 + second_point.1 * ratio2 + third_point.1 * ratio3)
    } else {
        None
    }
}

pub fn interoplate_on_mesh<T>(
    point: Vector3<f32>,
    vertices: &[(Vector3<f32>, T)],
    indices: &[usize],
) -> T
where
    T: Mul<f32, Output = T> + Add<Output = T> + Copy + Debug,
{
    for triangle in indices.chunks(3) {
        let first_point = vertices[triangle[0]];
        let second_point = vertices[triangle[1]];
        let third_point = vertices[triangle[2]];

        if let Some(value) = triangle_interpolate(point, first_point, second_point, third_point) {
            return value;
        }
    }
    panic!("Failed to find point on mesh: point: {point}")
}

pub fn are_points_contiguous(
    point1: Vector3<f32>,
    point2: Vector3<f32>,
    vertices: &[(Vector3<f32>, f32)],
    indices: &[usize],
) -> bool {
    const NUM_SAMPLES: usize = 100;
    for sample in 0..NUM_SAMPLES {
        let progress = (sample as f32 + 0.5) / NUM_SAMPLES as f32;

        let sample_vector = point1.slerp(&point2, progress) * World::RADIUS as f32;
        let interpolate = interoplate_on_mesh(sample_vector, vertices, indices);

        if interpolate < 0.0 {
            return false;
        }
    }
    true
}

pub fn get_planet(radius: f32) -> MeshOutput {
    let sphere = IcoSphere::subdivide(3);

    let vertices: Vec<_> = sphere
        .shared_vertex_iter()
        .map(|vertex| Vertex {
            position: Vector3::new(vertex.pos.x, vertex.pos.y, vertex.pos.z).normalize() * radius,
            normal: Vector3::new(vertex.pos.x, vertex.pos.y, vertex.pos.z).normalize(),
            tangent: Vector4::zeros(),
            texture_coordinate: Vector2::zeros(),
            texture_type: 0,
            bone_indices: Vector4::zeros(),
            bone_weights: Vector4::zeros(),
        })
        .collect();

    let indices: Vec<_> = sphere
        .indexed_polygon_iter()
        .map(|triangle| [triangle.x, triangle.y, triangle.z].into_iter())
        .flatten()
        .collect();
    return MeshOutput { vertices, indices };
}

pub fn get_elevations() -> Vec<f32> {
    // let paths = std::fs::read_dir("../GSG/elevation_gebco_small").unwrap();
    let mut samples = vec![(0.0, 0); (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    // for folder in paths.filter_map(|s| s.ok()) {
    //     let image_files = read_dir(folder.path()).unwrap();
    //     let path = image_files
    //         .filter_map(|s| s.ok())
    //         .find(|image_file| {
    //             if let Ok(file_name) = image_file.file_name().into_string() {
    //                 file_name.to_lowercase().contains("mea")
    //             } else {
    //                 false
    //             }
    //         })
    //         .unwrap()
    //         .path();

    //     load_raster_file(&path, &mut samples, |a|{a - 76.0});
    // }

    // for path in paths{
    //     if let Ok(path) = path{
    //         load_raster_file(&path.path(), &mut samples, |a|{a - 76.0})
    //     }
    // }

    load_raster_file(
        &PathBuf::from("../GSG/elevation_gebco_small/gebco_combined.tif"),
        &mut samples,
        |a, _| a - 76.0,
    );
    samples
        .iter()
        .map(|&(sum, count)| {
            if count > 0 {
                (sum / count as f64) as f32
            } else {
                0.0
            }
        })
        .collect()
}

pub fn get_aridity() -> Vec<f32> {
    let mut samples = vec![(0.0, 0); (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    // load_raster_file(&PathBuf::from("../GSG/aridity/ai_v3_yr.tif"), &mut samples, |a|{a / 10_000.0});
    load_raster_file(
        &PathBuf::from("../GSG/aridity_small/aridity.tif"),
        &mut samples,
        |a, _| a / 10_000.0,
    );

    samples
        .iter()
        .map(|&(sum, count)| {
            if count > 0 {
                (sum / count as f64) as f32
            } else {
                0.0
            }
        })
        .collect()
}

pub fn get_feb_temps() -> Vec<f32> {
    //TODO: global temperature change

    let mut samples = vec![(0.0, 0); (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    load_raster_file(
        &PathBuf::from("../GSG/temperature6/FebruaryTemp.tif"),
        &mut samples,
        |a, _| a + KELVIN_TO_CELSIUS,
    );

    samples
        .iter()
        .map(|&(sum, count)| {
            if count > 0 {
                (sum / count as f64) as f32
            } else {
                0.0
            }
        })
        .collect()
}

pub fn get_july_temps() -> Vec<f32> {
    let mut samples = vec![(0.0, 0); (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    load_raster_file(
        &PathBuf::from("../GSG/temperature6/JulyTemp.tif"),
        &mut samples,
        |a, _| a + KELVIN_TO_CELSIUS,
    );

    samples
        .iter()
        .map(|&(sum, count)| {
            if count > 0 {
                (sum / count as f64) as f32
            } else {
                0.0
            }
        })
        .collect()
}

pub fn get_populations(elevations: &[f32]) -> (Vec<f32>, f32) {
    let mut samples = vec![(0.0, 0); (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    // load_raster_file(&PathBuf::from("../GSG/population2/gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2020_30_sec.tif"), &mut samples, |a|{a});
    load_raster_file(
        &PathBuf::from("../GSG/population2/small.tif"),
        &mut samples,
        |a, _| a,
    );
    // load_raster_file(&PathBuf::from("../GSG/population_small/population.tif"), &mut samples, |a|{a});

    let mut pops_underwater = 0.0;
    (
        samples
            .iter()
            .enumerate()
            .map(|(i, &(sum, _))| {
                let total_population = sum as f32;
                if elevations[i] <= 0.0 {
                    pops_underwater += total_population;
                }
                total_population
            })
            .collect(),
        pops_underwater,
    )
}

trait Convert<T> {
    fn convert(self) -> T;
}

impl Convert<u16> for f64 {
    fn convert(self) -> u16 {
        self as u16
    }
}

impl<T> Convert<Self> for T {
    fn convert(self) -> Self {
        self
    }
}

pub fn get_countries() -> (Box<[Option<u16>]>, Box<[String]>) {
    let mut samples = vec![(0u16, 0); (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    load_raster_file(
        &PathBuf::from("../GSG/nations/nations.tif"),
        &mut samples,
        |a, samples| if samples == 0 { a } else { 0 },
    );
    let ids = samples
        .iter()
        .map(|&(a, _)| if a == 0 { None } else { Some(a - 1) })
        .collect();

    #[derive(Debug, Deserialize)]
    struct CsvRow {
        name: String,
        country_id: u16,
    }
    let mut reader =
        csv::Reader::from_path("../GSG/nations/nations.csv").expect("Failed to open nations csv");
    let country_names: Box<[String]> = reader
        .deserialize()
        .map(|a: Result<CsvRow, _>| a.expect("failed to deserialize nations csv").name)
        .collect();

    (ids, country_names)
}

fn load_raster_file<F, T>(path: &Path, samples: &mut Vec<(T, u32)>, additional_processing: F)
where
    F: Fn(T, u32) -> T,
    T: AddAssign + Default + GdalType + Clone + Copy + PartialEq,
    f64: Convert<T>,
{
    println!("Loading file: {:?}", path);
    let dataset = gdal::Dataset::open(path).unwrap();
    let raster_band = dataset.rasterband(1).unwrap();

    let transform = dataset.geo_transform().unwrap();
    let mut instant = std::time::Instant::now();

    let num_pixels = raster_band.size().0 * raster_band.size().1;
    dbg!(raster_band.size());
    let mut slice = vec![T::default(); num_pixels].into_boxed_slice();
    match raster_band.read_into_slice(
        (0, 0),
        raster_band.size(),
        raster_band.size(),
        &mut slice,
        Some(gdal::raster::ResampleAlg::Bilinear),
    ) {
        Err(error) => {
            println!("GDAL Error: {:?}", error)
        }
        Ok(..) => {
            for (index, pixel) in slice.iter().enumerate() {
                let time_elapsed = instant.elapsed().as_secs_f64();
                if time_elapsed > 1.0 {
                    instant = std::time::Instant::now();
                    // let index = longitude + latitude * raster_band.size().0;
                    // let index = latitude + longitude * raster_band.size().1;
                    println!("Percent done file: {:}", index as f64 / num_pixels as f64);
                };
                let longitude = index % raster_band.size().0;
                let latitude = index / raster_band.size().0;
                // dbg!(longitude);
                // dbg!(latitude);
                let transformed_longitude = (transform[0]
                    + longitude as f64 * transform[1]
                    + latitude as f64 * transform[2])
                    .to_radians();
                let transformed_latitude = (transform[3]
                    + longitude as f64 * transform[4]
                    + latitude as f64 * transform[5])
                    .to_radians();
                // dbg!(transformed_longitude);
                // dbg!(transformed_latitude);
                let raster_coordinate = Vector3::new(
                    transformed_latitude.cos() * transformed_longitude.cos(),
                    transformed_latitude.sin(),
                    transformed_latitude.cos() * transformed_longitude.sin(),
                );
                // dbg!(raster_coordinate);
                // let closest_index = samples.iter().enumerate().min_by_key(|&(index, _)|{
                //     let coordinate = index_to_coordinate(index).normalize().cast();
                //     FloatOrd((raster_coordinate - coordinate).magnitude())
                // }).unwrap().0;
                let mut closest_index = coordinate_to_index(raster_coordinate);
                if closest_index >= CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6 {
                    dbg!(closest_index);
                    closest_index = CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6 - 1;
                }
                // dbg!(closest_index);
                // dbg!(index_to_coordinate(closest_index).normalize());
                // dbg!(pixel);
                let num_samples = samples[closest_index].1;
                match raster_band.no_data_value() {
                    Some(no_data_value) => {
                        if *pixel != no_data_value.convert() {
                            samples[closest_index].0 += additional_processing(*pixel, num_samples);
                            samples[closest_index].1 += 1;
                        }
                    }
                    None => {
                        samples[closest_index].0 += additional_processing(*pixel, num_samples);
                        samples[closest_index].1 += 1;
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Error {
    PropogationFailed,
}

#[derive(Clone, Debug)]
struct Cell {
    cell_state: CellState,
    neighbours: Vec<usize>,
}

#[derive(Clone, Debug)]
enum CellState {
    Collapsed(i8),                   //Collapsed cell has only one value
    Superposition(HashMap<i8, f32>), //Cell is in a superposition of all possible values
}

#[derive(Clone, Copy, Debug)]
enum Ratio {
    RawSamples { sample_count: u32 },
    Data(f32),
}
impl Ratio {
    fn add_sample(&mut self) {
        match self {
            Ratio::RawSamples { sample_count } => {
                *sample_count += 1;
            }
            Ratio::Data(_) => panic!("Already averaged"),
        }
    }
    // fn average(&mut self){
    //     match self{
    //         AverageRatio::RawData { sum, sample_count } => {
    //             let value = *sum as f32 / *sample_count as f32;
    //             *self = AverageRatio::AveragedData(value);
    //         },
    //         AverageRatio::AveragedData(_) => panic!("Already Averaged"),
    //     }
    // }
}

fn balance_ratios(hashmap: &mut HashMap<i8, f32>) {
    let ratio_sum: f32 = hashmap.values().sum();
    assert_ne!(ratio_sum, 0.0);

    for ratio in hashmap.values_mut() {
        *ratio /= ratio_sum;
    }
}
