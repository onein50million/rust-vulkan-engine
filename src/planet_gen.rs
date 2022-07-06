use crate::{
    support::{coordinate_to_index, index_to_coordinate, Vertex, CUBEMAP_WIDTH},
    world::{ProvinceMap, World},
};
use float_ord::FloatOrd;
use gdal::{
    errors::GdalError,
    raster::{GdalType, ResampleAlg},
    spatial_ref::SpatialRef,
    Driver, GeoTransform,
};
use gdal_sys::{
    CPLErr, GDALChunkAndWarpImage, GDALClose, GDALCreateGenImgProjTransformer,
    GDALCreateReprojectionTransformer, GDALCreateWarpOperation, GDALCreateWarpOptions,
    GDALDataType::{GDT_Float32, GDT_Unknown},
    GDALDatasetH, GDALGenImgProjTransform, GDALReprojectionTransform, GDALResampleAlg,
    GDALTermProgress, GDALWarpAppOptionsNew, GDALWarpOptions,
};
use genmesh::generators::{IcoSphere, IndexedPolygon, SharedVertex};
use nalgebra::{coordinates::X, ComplexField, Vector2, Vector3, Vector4};
use serde::{Deserialize};
use std::{
    collections::HashMap,
    f64::consts::PI,
    ffi::CString,
    fmt::Debug,
    fs::{read_dir, File},
    mem::MaybeUninit,
    ops::{Add, AddAssign, Mul},
    path::{Path, PathBuf},
    ptr::{null, null_mut}, any::TypeId,
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

const KELVIN_TO_CELSIUS: f32 = -273.15;
const EARTH_AREA: f64 = 5.1E14; //in meters

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
    let mut samples = vec![(0.0f32, 0); (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
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
        1
        // "bilinear",
    );
    samples.into_iter().map(|(value, sample_count)|{
        value / sample_count as f32
    }).collect()
}

pub fn get_aridity() -> Vec<f32> {
    let mut samples = vec![(0.0f32, 0); (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    // load_raster_file(&PathBuf::from("../GSG/aridity/ai_v3_yr.tif"), &mut samples, |a|{a / 10_000.0});
    load_raster_file(
        &PathBuf::from("../GSG/aridity_small/aridity.tif"),
        &mut samples,
        1
        // "bilinear",
    );

    samples.into_iter().map(|(s, sample_count)| (s / sample_count as f32) / 10_000.0).collect()
}

pub fn get_feb_temps() -> Vec<f32> {
    //TODO: global temperature change

    let mut samples = vec![(0.0, 0); (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    load_raster_file(
        &PathBuf::from("../GSG/temperature6/FebruaryTemp.tif"),
        &mut samples,
        1
        // "bilinear",
    );
    samples.into_iter().map(|(s, sample_count)| (s / sample_count as f32) + KELVIN_TO_CELSIUS).collect()
}

pub fn get_july_temps() -> Vec<f32> {
    let mut samples = vec![(0.0f32,0); (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    load_raster_file(
        &PathBuf::from("../GSG/temperature6/JulyTemp.tif"),
        &mut samples,
        1
        // "bilinear",
    );
    samples.into_iter().map(|(s, sample_count)| (s / sample_count as f32) + KELVIN_TO_CELSIUS).collect()
}

pub fn get_populations() -> Vec<f32> {
    let mut samples = vec![(0.0, 0); (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    // load_raster_file(
    //     &PathBuf::from("../GSG/population2/small.tif"),
    //     &mut samples,
    //     "sum",
    // );
    load_raster_file(
        &PathBuf::from("../GSG/population/population.tif"),
        &mut samples,
        1
    );  
    const PIXEL_AREA: f64 = EARTH_AREA * (1.0 / (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as f64);
    const SQUARE_KILOMETER_TO_SQUARE_METER_RATIO: f64 = 1000000.0;
    samples.into_iter().map(|(s, sample_count)|(if sample_count > 0 {s / sample_count as f64} else {0.0} / SQUARE_KILOMETER_TO_SQUARE_METER_RATIO * PIXEL_AREA) as f32).collect()
}

trait Convert<T> {
    fn convert(self) -> T;
}

impl Convert<u8> for f64 {
    fn convert(self) -> u8 {
        self as u8
    }
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
    // load_raster_file(
    //     &PathBuf::from("../GSG/nations/nations.tif"),
    //     &mut samples,
    //     "near",
    // );
    load_raster_file(
        &PathBuf::from("../GSG/nations/nations.tif"),
        &mut samples,
        0
    );
    let ids = samples
        .into_iter()
        .map(|(a, _)| if a == 0 { None } else { Some(a - 1) })
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

#[derive(Default, Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProvinceRoot {
    features: Vec<Feature>,
}


#[derive(Default, Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Feature {
    #[serde(rename = "type")]
    type_field: String,
    properties: Properties,
}

#[derive(Default, Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Properties {
    #[serde(rename = "adm1_code")]
    adm1_code: String,
    name: Option<String>,
    id: i64,
}



pub fn get_provinces() -> (
    Box<[Option<u16>]>,
    Box<[Vector3<f64>]>,
    ProvinceMap<Box<[usize]>>,
    ProvinceMap<String>,
) {
    let mut samples = vec![(0u16, 0); (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    // load_raster_file(
    //     &PathBuf::from("../GSG/provinces/provinces.tif"),
    //     &mut samples,
    //     "near",
    // );
    load_raster_file(
        &PathBuf::from("../GSG/provinces/provinces.tif"),
        &mut samples,
        0
    );
    let ids = samples
        .into_iter()
        .map(|(a, _)| if a == 0 { None } else { Some(a - 1) })
        .collect();

    let (vertices, indices) =
        load_vector_file(&PathBuf::from("../GSG/provinces/provinces.geojson"));
    let mut reader =
        File::open("../GSG/provinces/provinces.geojson").expect("Failed to open provinces");
    let province_root: ProvinceRoot = serde_json::from_reader(reader).expect("Failed to deserialize provinces");
    (ids, vertices, ProvinceMap(indices), ProvinceMap(province_root.features.into_iter().map(|f|f.properties.name.unwrap_or("UNNAMED".to_string())).collect()))
}

pub fn get_water() -> Vec<f32> {
    let mut samples = vec![(0u8,0); (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    // load_raster_file(&PathBuf::from("../GSG/population2/gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2020_30_sec.tif"), &mut samples, |a|{a});
    // load_raster_file(
    //     &PathBuf::from("../GSG/water/water.tif"),
    //     &mut samples,
    //     "near",
    // );
    load_raster_file(
        &PathBuf::from("../GSG/water/water.tif"),
        &mut samples,
        1
    );
    // for sample in &samples {
    //     if (sample - 1.0).abs() > 0.1 {
    //         dbg!(sample);
    //     }
    // }
    samples.into_iter().map(|(a, sample_count)|a as f32 / sample_count as f32).collect()
}


fn load_raster_file<T>(
    path: &Path,
    out_slice: &mut [(T, usize)],
    sample_distance: usize,
    // resample_alg: &str, //Resample alg type, safe bindings don't support the sum algorithm so this is sorta a workaround
    // additional_processing: F,
) where
    // F: Fn(T, u32) -> T,
    T: AddAssign + Default + GdalType + Clone + Copy + PartialEq + std::iter::Sum<T>,
    // f64: Convert<T>,
    T: GdalType + Clone + Copy,
{
    println!("Loading file: {:?}", path);
    let source_dataset = gdal::Dataset::open(path).unwrap();
    let raster_band = source_dataset.rasterband(1).unwrap();

    let mut transform = source_dataset.geo_transform().unwrap();

    let mut inv_transform = [0.0f64; 6];
    unsafe {
        assert_eq!(
            gdal_sys::GDALInvGeoTransform(transform.as_mut_ptr(), inv_transform.as_mut_ptr(),),
            1,
            "InvGeoTransform failed"
        );
    }

    let mut instant = std::time::Instant::now();

    let mut in_buf = vec![T::default(); raster_band.size().0 * raster_band.size().1];
    raster_band.read_into_slice(
                (0, 0),
                raster_band.size(),
                raster_band.size(),
                &mut in_buf,
                None,
            )
            .unwrap();
    let in_slice = in_buf.into_boxed_slice();

    for (index, pixel) in out_slice.iter_mut().enumerate() {
        let time_elapsed = instant.elapsed().as_secs_f64();
        if time_elapsed > 1.0 {
            instant = std::time::Instant::now();
            println!("Percent done file: {:}", index as f64 / (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as f64);
        };
        
        let coordinate = index_to_coordinate(index).normalize().cast::<f64>();
        let longitude = coordinate.z.atan2(coordinate.x).to_degrees();
        let latitude = coordinate.y.asin().to_degrees();

        let x_geo =( (inv_transform[0] + longitude * inv_transform[1] + latitude*inv_transform[2] - 1.0) as isize).rem_euclid(raster_band.x_size() as isize);
        let y_geo =( (inv_transform[3] + longitude * inv_transform[4] + latitude*inv_transform[5] - 1.0) as isize).rem_euclid(raster_band.y_size() as isize);

        let above_y = if y_geo == ((raster_band.y_size() as isize) - 1){0}else{y_geo + 1}.rem_euclid(raster_band.y_size() as isize);
        let below_y = if y_geo == 0{(raster_band.y_size() as isize) - 1}else{y_geo - 1}.rem_euclid(raster_band.y_size() as isize);
        let right_x = if x_geo == ((raster_band.x_size() as isize) - 1){0}else{x_geo + 1}.rem_euclid(raster_band.x_size() as isize);
        let left_x = if x_geo == 0{(raster_band.y_size() as isize) - 1}else{x_geo - 1}.rem_euclid(raster_band.x_size() as isize);
        let raster_value = in_slice[x_geo as usize + y_geo as usize * raster_band.x_size()];
        let raster_value_above = in_slice[x_geo as usize + above_y as usize * raster_band.x_size()];
        let raster_value_below = in_slice[x_geo as usize + below_y as usize * raster_band.x_size()];
        let raster_value_left = in_slice[left_x as usize + y_geo as usize * raster_band.x_size()];
        let raster_value_right = in_slice[right_x as usize + y_geo as usize * raster_band.x_size()];

        if sample_distance > 0{
            pixel.0 += [
                raster_value,
                raster_value_above,
                raster_value_below,
                raster_value_left,
                raster_value_right,
            ].into_iter().sum::<T>();
            pixel.1 += 5;
        }else{
            *pixel = (raster_value, 1)
        }

}

}

fn latlong_to_vector(latitude: f64, longitude: f64) -> Vector3<f64> {
    let latitude = latitude.to_radians();
    let longitude = longitude.to_radians();
    Vector3::new(
        latitude.cos() * longitude.cos(),
        latitude.sin(),
        latitude.cos() * longitude.sin(),
    ) * World::RADIUS
}

fn load_vector_file(path: &Path) -> (Box<[Vector3<f64>]>, Box<[Box<[usize]>]>) {
    println!("Loading file: {:?}", path);
    let dataset = gdal::Dataset::open(path).unwrap();
    let mut vector_layer = dataset.layer(0).unwrap();

    let mut vertices = vec![];
    let mut indices = vec![];
    let mut current_index = 0;
    for feature in vector_layer.features() {
        let mut current_indices = vec![];
        for i in 0..feature.geometry().geometry_count() {
            let geometry = unsafe { feature.geometry().get_unowned_geometry(i) };
            let point_vec = geometry.get_point_vec();
            assert_ne!(point_vec.len(), 0);

            let first_index = current_index;
            let last_point_index = point_vec.len() - 1;
            for &(longitude, latitude, _) in &point_vec[..last_point_index] {
                current_indices.push(current_index);
                current_index += 1;
                current_indices.push(current_index);
                vertices.push(latlong_to_vector(latitude, longitude));
            }
            let last_point = point_vec[last_point_index];
            vertices.push(latlong_to_vector(last_point.1, last_point.0));
            current_indices.push(current_index);
            current_index += 1;
            current_indices.push(first_index);
        }
        indices.push(current_indices.into_boxed_slice());
    }
    (vertices.into_boxed_slice(), indices.into_boxed_slice())
}
