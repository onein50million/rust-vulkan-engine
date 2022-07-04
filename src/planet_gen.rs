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
use serde::Deserialize;
use std::ffi::CStr;
use std::{
    collections::HashMap,
    f64::consts::PI,
    ffi::CString,
    fmt::Debug,
    fs::read_dir,
    mem::MaybeUninit,
    ops::{Add, AddAssign, Mul},
    path::{Path, PathBuf},
    ptr::{null, null_mut},
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
    let mut samples = vec![0.0; (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
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
        "bilinear",
    );
    samples
}

pub fn get_aridity() -> Vec<f32> {
    let mut samples = vec![0.0f32; (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    // load_raster_file(&PathBuf::from("../GSG/aridity/ai_v3_yr.tif"), &mut samples, |a|{a / 10_000.0});
    load_raster_file(
        &PathBuf::from("../GSG/aridity_small/aridity.tif"),
        &mut samples,
        "bilinear",
    );

    samples.iter().map(|s| s / 10_000.0).collect()
}

pub fn get_feb_temps() -> Vec<f32> {
    //TODO: global temperature change

    let mut samples = vec![0.0; (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    load_raster_file(
        &PathBuf::from("../GSG/temperature6/FebruaryTemp.tif"),
        &mut samples,
        "bilinear",
    );
    samples.iter().map(|s| s + KELVIN_TO_CELSIUS).collect()
}

pub fn get_july_temps() -> Vec<f32> {
    let mut samples = vec![0.0f32; (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    load_raster_file(
        &PathBuf::from("../GSG/temperature6/JulyTemp.tif"),
        &mut samples,
        "bilinear",
    );
    samples.iter().map(|s| s + KELVIN_TO_CELSIUS).collect()
}

pub fn get_populations() -> Vec<f32> {
    let mut samples = vec![0.0; (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    load_raster_file(
        &PathBuf::from("../GSG/population2/small.tif"),
        &mut samples,
        "sum",
    );
    samples
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
    let mut samples = vec![0u16; (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    load_raster_file(
        &PathBuf::from("../GSG/nations/nations.tif"),
        &mut samples,
        "near",
    );
    let ids = samples
        .iter()
        .map(|&a| if a == 0 { None } else { Some(a - 1) })
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

pub fn get_provinces() -> (
    Box<[Option<u16>]>,
    Box<[Vector3<f64>]>,
    ProvinceMap<Box<[usize]>>,
) {
    let mut samples = vec![0u16; (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    load_raster_file(
        &PathBuf::from("../GSG/provinces/provinces.tif"),
        &mut samples,
        "near",
    );
    let ids = samples
        .iter()
        .map(|&a| if a == 0 { None } else { Some(a - 1) })
        .collect();

    let (vertices, indices) =
        load_vector_file(&PathBuf::from("../GSG/provinces/provinces.geojson"));

    (ids, vertices, ProvinceMap(indices))
}

pub fn get_water() -> Vec<f32> {
    let mut samples = vec![0u8; (CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6) as usize];
    // load_raster_file(&PathBuf::from("../GSG/population2/gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2020_30_sec.tif"), &mut samples, |a|{a});
    load_raster_file(
        &PathBuf::from("../GSG/water/water.tif"),
        &mut samples,
        "near",
    );
    // for sample in &samples {
    //     if (sample - 1.0).abs() > 0.1 {
    //         dbg!(sample);
    //     }
    // }
    samples.iter().map(|&a|a as f32).collect()
}

// const PROJ4_FACES: &[&str] = &[
//     "+wktext +proj=qsc +axis=wnu +units=m +ellps=WGS84 +lat_0=0 +lon_0=-90",
//     "+wktext +proj=qsc +axis=wnu +units=m +ellps=WGS84 +lat_0=0 +lon_0=90",
//     "+wktext +proj=qsc +axis=wnu +units=m +ellps=WGS84 +lat_0=90 +lon_0=0",
//     "+wktext +proj=qsc +axis=wnu +units=m +ellps=WGS84 +lat_0=-90 +lon_0=0",
//     "+wktext +proj=qsc +axis=wnu +units=m +ellps=WGS84 +lat_0=0 +lon_0=0",
//     "+wktext +proj=qsc +axis=wnu +units=m +ellps=WGS84 +lat_0=0 +lon_0=180",
// ];

// const PROJ4_FACES: &[&str] = &[
//     "+wktext +proj=s2 +UVtoST=none +axis=wnu +ellps=WGS84 +lat_0=0 +lon_0=-90",
//     "+wktext +proj=s2 +UVtoST=none +axis=wnu +ellps=WGS84 +lat_0=0 +lon_0=90",
//     "+wktext +proj=s2 +UVtoST=none +axis=wnu +ellps=WGS84 +lat_0=90 +lon_0=0",
//     "+wktext +proj=s2 +UVtoST=none +axis=wnu +ellps=WGS84 +lat_0=-90 +lon_0=0",
//     "+wktext +proj=s2 +UVtoST=none +axis=wnu +ellps=WGS84 +lat_0=0 +lon_0=0",
//     "+wktext +proj=s2 +UVtoST=none +axis=wnu +ellps=WGS84 +lat_0=0 +lon_0=180",
// ];

// const PROJ4_FACES: &[&str] = &[
//     "+wktext +proj=ortho +axis=wnu +units=m +ellps=WGS84 +lat_0=0 +lon_0=-90",
//     "+wktext +proj=ortho +axis=wnu +units=m +ellps=WGS84 +lat_0=0 +lon_0=90",
//     "+wktext +proj=ortho +axis=wnu +units=m +ellps=WGS84 +lat_0=90 +lon_0=0",
//     "+wktext +proj=ortho +axis=wnu +units=m +ellps=WGS84 +lat_0=-90 +lon_0=0",
//     "+wktext +proj=ortho +axis=wnu +units=m +ellps=WGS84 +lat_0=0 +lon_0=0",
//     "+wktext +proj=ortho +axis=wnu +units=m +ellps=WGS84 +lat_0=0 +lon_0=180",
// ];

// const PROJ4_FACES: &[&str] = &[
//     "+wktext +proj=nsper +h=6378137 +axis=wnu +ellps=WGS84 +units=m +lat_0=0 +lon_0=-90",
//     "+wktext +proj=nsper +h=6378137 +axis=wnu +ellps=WGS84 +units=m +lat_0=0 +lon_0=90",
//     "+wktext +proj=nsper +h=6378137 +axis=wnu +ellps=WGS84 +units=m +lat_0=90 +lon_0=0",
//     "+wktext +proj=nsper +h=6378137 +axis=wnu +ellps=WGS84 +units=m +lat_0=-90 +lon_0=0",
//     "+wktext +proj=nsper +h=6378137 +axis=wnu +ellps=WGS84 +units=m +lat_0=0 +lon_0=0",
//     "+wktext +proj=nsper +h=6378137 +axis=wnu +ellps=WGS84 +units=m +lat_0=0 +lon_0=180",
// ];

// const PROJ4_FACES: &[&str] = &[
//     "+wktext +proj=nsper +h=6378137000000000 +axis=wnu +ellps=WGS84 +units=m +lat_0=0 +lon_0=-90",
//     "+wktext +proj=nsper +h=6378137000000000 +axis=wnu +ellps=WGS84 +units=m +lat_0=0 +lon_0=90",
//     "+wktext +proj=nsper +h=6378137000000000 +axis=wnu +ellps=WGS84 +units=m +lat_0=90 +lon_0=0",
//     "+wktext +proj=nsper +h=6378137000000000 +axis=wnu +ellps=WGS84 +units=m +lat_0=-90 +lon_0=0",
//     "+wktext +proj=nsper +h=6378137000000000 +axis=wnu +ellps=WGS84 +units=m +lat_0=0 +lon_0=0",
//     "+wktext +proj=nsper +h=6378137000000000 +axis=wnu +ellps=WGS84 +units=m +lat_0=0 +lon_0=180",
// ];
const PROJ4_FACES: &[&str] = &[
    "+wktext +proj=rhealpix -f '%.2f' +south_square=0 +north_square=2 +axis=wnu +ellps=WGS84 +units=m +lat_0=0 +lon_0=-90",
    "+wktext +proj=rhealpix -f '%.2f' +south_square=0 +north_square=2 +axis=wnu +ellps=WGS84 +units=m +lat_0=0 +lon_0=90",
    "+wktext +proj=rhealpix -f '%.2f' +south_square=0 +north_square=2 +axis=wnu +ellps=WGS84 +units=m +lat_0=90 +lon_0=0",
    "+wktext +proj=rhealpix -f '%.2f' +south_square=0 +north_square=2 +axis=wnu +ellps=WGS84 +units=m +lat_0=-90 +lon_0=0",
    "+wktext +proj=rhealpix -f '%.2f' +south_square=0 +north_square=2 +axis=wnu +ellps=WGS84 +units=m +lat_0=0 +lon_0=0",
    "+wktext +proj=rhealpix -f '%.2f' +south_square=0 +north_square=2 +axis=wnu +ellps=WGS84 +units=m +lat_0=0 +lon_0=180",
];

fn load_raster_file<T>(
    path: &Path,
    slice: &mut [T],
    resample_alg: &str, //Resample alg type, safe bindings don't support the sum algorithm so this is sorta a workaround
                       // additional_processing: F,
) where
    // F: Fn(T, u32) -> T,
    // T: AddAssign + Default + GdalType + Clone + Copy + PartialEq,
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
    let transform = transform;
    let spatial_ref = source_dataset.spatial_ref().unwrap();

    let mut instant = std::time::Instant::now();

    let num_pixels = raster_band.size().0 * raster_band.size().1;
    dbg!(raster_band.size());

    let wgs_84 = gdal::spatial_ref::SpatialRef::from_epsg(4326).unwrap();

    // let memory_driver = Driver::get("MEM").unwrap();
    // dbg!(memory_driver.long_name());
    for face in 0..6 {
        // let name_string = unsafe {
        //     CString::new(format!(
        //         "MEM:::DATAPOINTER={:?}
        //         ",
        //         slice
        //             .as_ptr()
        //             .offset((CUBEMAP_WIDTH * CUBEMAP_WIDTH * face) as isize)
        //     ))
        //     .unwrap()
        // };

        // println!("Creating face dataset");
        // let mut face_dataset = unsafe {
        //     gdal::Dataset::from_c_dataset({
        //         let result = gdal_sys::GDALCreate(
        //             memory_driver.c_driver(),
        //             null_mut(),
        //             CUBEMAP_WIDTH as i32,
        //             CUBEMAP_WIDTH as i32,
        //             1,
        //             T::gdal_type(),
        //             null_mut(),
        //         );
        //         if result.is_null() {
        //             panic!("GDALCreate returned null pointer")
        //         } else {
        //             result
        //         }
        //     })
        // };

        // println!("Creating face dataset");
        // let name_string = CString::new(format!("Face{:}", face))
        // .unwrap();
        // let mut face_dataset = unsafe {
        //     gdal::Dataset::from_c_dataset({
        //         let result = gdal_sys::GDALCreate(
        //             Driver::get("GTiff").unwrap().c_driver(),
        //             name_string.as_ptr(),
        //             CUBEMAP_WIDTH as i32,
        //             CUBEMAP_WIDTH as i32,
        //             1,
        //             T::gdal_type(),
        //             null_mut(),
        //         );
        //         if result.is_null() {
        //             panic!("GDALCreate returned null pointer")
        //         } else {
        //             result
        //         }
        //     })
        // };
        // face_dataset.set_geo_transform(transformation)

        //these two might get overriden because of the spatial ref but I couldn't find documentation on the difference
        // face_dataset.set_projection("qsc").unwrap();
        // face_dataset.set_geo_transform(&transform).unwrap();

        //https://proj.org/operations/projections/qsc.html
        // face_dataset
        //     .set_spatial_ref(&SpatialRef::from_proj4(PROJ4_FACES[face]).unwrap())
        //     .unwrap();
        // // face_dataset.set

        // let source_extra = CString::new("SOURCE_EXTRA=100").unwrap();
        // let sample_grid = CString::new("SAMPLE_GRID=YES").unwrap();
        // let mut warp_option_strings = Box::new([
        //     source_extra.as_ptr() as *mut i8,
        //     sample_grid.as_ptr() as *mut i8,
        // ]);

        // let mut options = unsafe{ GDALWarpOptions{
        //     papszWarpOptions: warp_option_strings.as_mut_ptr(),
        //     dfWarpMemoryLimit: 0.0,
        //     eResampleAlg: resample_alg,
        //     eWorkingDataType: T::gdal_type(),
        //     hSrcDS: source_dataset.c_dataset(),
        //     hDstDS: face_dataset.c_dataset(),
        //     nBandCount: 1,
        //     panSrcBands: [1].as_mut_ptr(),
        //     panDstBands: [1].as_mut_ptr(),
        //     nSrcAlphaBand: 0,
        //     nDstAlphaBand: 0,
        //     padfSrcNoDataReal: [raster_band.no_data_value().unwrap_or(f64::NAN)].as_mut_ptr(),
        //     padfSrcNoDataImag: null_mut(),
        //     padfDstNoDataReal: [raster_band.no_data_value().unwrap_or(f64::NAN)].as_mut_ptr(),
        //     padfDstNoDataImag: null_mut(),
        //     pfnProgress: Some(GDALTermProgress),
        //     pProgressArg: null_mut(),
        //     pfnTransformer: Some(GDALGenImgProjTransform),
        //     pTransformerArg: GDALCreateGenImgProjTransformer(source_dataset.c_dataset(), null(), face_dataset.c_dataset(), null(), 0, 0.0, 0),
        //     papfnSrcPerBandValidityMaskFunc: [None].as_mut_ptr(),
        //     papSrcPerBandValidityMaskFuncArg: null_mut(),
        //     pfnSrcValidityMaskFunc: None,
        //     pSrcValidityMaskFuncArg: null_mut(),
        //     pfnSrcDensityMaskFunc: None,
        //     pSrcDensityMaskFuncArg: null_mut(),
        //     pfnDstDensityMaskFunc: None,
        //     pDstDensityMaskFuncArg: null_mut(),
        //     pfnDstValidityMaskFunc: None,
        //     pDstValidityMaskFuncArg: null_mut(),
        //     pfnPreWarpChunkProcessor: None,
        //     pPreWarpProcessorArg: null_mut(),
        //     pfnPostWarpChunkProcessor: None,
        //     pPostWarpProcessorArg: null_mut(),
        //     hCutline: null_mut(),
        //     dfCutlineBlendDist: 0.0,
        // }};
        // let source_wkt = CString::new(source_dataset.spatial_ref().unwrap().to_wkt().unwrap()).unwrap();
        // let dest_wkt = CString::new(face_dataset.spatial_ref().unwrap().to_wkt().unwrap()).unwrap();
        // let options = unsafe {
        //     let mut options = *GDALCreateWarpOptions();
        //     // options.papszWarpOptions = warp_option_strings.as_mut_ptr();
        //     // options.eResampleAlg = resample_alg;
        //     options.hSrcDS = source_dataset.c_dataset();
        //     options.hDstDS = face_dataset.c_dataset();
        //     // options.pfnTransformer = Some(GDALReprojectionTransform);
        //     // options.pTransformerArg = GDALCreateReprojectionTransformer(
        //     //     source_wkt.as_ptr(),
        //     //     dest_wkt.as_ptr()
        //     // );
        //     options.pfnTransformer = Some(GDALGenImgProjTransform);
        //     options.pTransformerArg = GDALCreateGenImgProjTransformer(source_dataset.c_dataset(), source_wkt.as_ptr(), face_dataset.c_dataset(), dest_wkt.as_ptr(), 0, 0.0, 0);

        //     options
        // };
        // println!("Reprojecting");

        // unsafe {
        //     let warper = GDALCreateWarpOperation(&options as *const _);

        //     let warp_result =
        //         GDALChunkAndWarpImage(warper, 0, 0, CUBEMAP_WIDTH as i32, CUBEMAP_WIDTH as i32);
        //     if warp_result != CPLErr::CE_None {
        //         panic!("Warp failed with: {:}", warp_result);
        //     }
        // }

        let name_string = CString::new(format!("/vsimem/Face{:}.tif", face)).unwrap();
        let face_dataset = unsafe {
            let arguments = &mut [
                CString::new("-t_srs").unwrap().into_raw(),
                CString::new(PROJ4_FACES[face]).unwrap().into_raw(),
                // CString::new("-te").unwrap().into_raw(),
                // CString::new("-6378137").unwrap().into_raw(),
                // CString::new("-6378137").unwrap().into_raw(),
                // CString::new("6378137").unwrap().into_raw(),
                // CString::new("6378137").unwrap().into_raw(),
                // CString::new("-te").unwrap().into_raw(),
                // CString::new("-4510023.92").unwrap().into_raw(),
                // CString::new("-4510023.92").unwrap().into_raw(),
                // CString::new("4510023.92").unwrap().into_raw(),
                // CString::new("4510023.92").unwrap().into_raw(),
                // CString::new("-te").unwrap().into_raw(),
                // CString::new("-3189068.5").unwrap().into_raw(),
                // CString::new("-3189068.5").unwrap().into_raw(),
                // CString::new("3189068.5").unwrap().into_raw(),
                // CString::new("3189068.5").unwrap().into_raw(),
                // CString::new("-te").unwrap().into_raw(),
                // CString::new("-797267.125").unwrap().into_raw(),
                // CString::new("-797267.125").unwrap().into_raw(),
                // CString::new("797267.125").unwrap().into_raw(),
                // CString::new("797267.125").unwrap().into_raw(),
                // CString::new("-te").unwrap().into_raw(),
                // CString::new("-1594534.25").unwrap().into_raw(),
                // CString::new("-1594534.25").unwrap().into_raw(),
                // CString::new("1594534.25").unwrap().into_raw(),
                // CString::new("1594534.25").unwrap().into_raw(),
                CString::new("-te").unwrap().into_raw(),
                CString::new("-5009375").unwrap().into_raw(),
                CString::new("-5009375").unwrap().into_raw(),
                CString::new("5009375").unwrap().into_raw(),
                CString::new("5009375").unwrap().into_raw(),
                // CString::new("-te").unwrap().into_raw(),
                // CString::new("-2588300").unwrap().into_raw(),
                // CString::new("-2588300").unwrap().into_raw(),
                // CString::new("2588300").unwrap().into_raw(),
                // CString::new("2588300").unwrap().into_raw(),

                CString::new("-r").unwrap().into_raw(),
                CString::new(resample_alg).unwrap().into_raw(),
                CString::new("-ts").unwrap().into_raw(),
                CString::new(CUBEMAP_WIDTH.to_string()).unwrap().into_raw(),
                CString::new(CUBEMAP_WIDTH.to_string()).unwrap().into_raw(),
                CString::new("-wo").unwrap().into_raw(),
                CString::new("SOURCE_EXTRA=100").unwrap().into_raw(),
                CString::new("-wo").unwrap().into_raw(),
                CString::new("SAMPLE_GRID=YES").unwrap().into_raw(),

                null_mut()
            ];

            let app_options = GDALWarpAppOptionsNew(arguments.as_mut_ptr(), null_mut());

            let mut usage_error = 69420;
            let out = gdal_sys::GDALWarp(
                name_string.into_raw(),
                null_mut(),
                1,
                [source_dataset.c_dataset()].as_mut_ptr(),
                app_options,
                (&mut usage_error) as *mut _,
            );
            // dbg!(usage_error);
            out
        };

        // unsafe {
        //     let result = gdal_sys::GDALReprojectImage(
        //         source_dataset.c_dataset(),
        //         null(),
        //         face_dataset.c_dataset(),
        //         null(),
        //         resample_alg,
        //         0.0,
        //         0.0,
        //         Some(GDALTermProgress),
        //         null_mut(),
        //         &mut options as *mut _
        //     );
        //     match result {
        //         CPLErr::CE_None => {}
        //         CPLErr::CE_Failure => {
        //             println!("Failure to reproject image with CE_Failure")
        //         }
        //         _ => {
        //             println!("Failure to reproject with error {:}", result)
        //         }
        //     }
        // }

        // println!("Reading face dataset");
        unsafe { gdal::Dataset::from_c_dataset(face_dataset) }
            .rasterband(1)
            .unwrap()
            .read_into_slice(
                (0, 0),
                (CUBEMAP_WIDTH, CUBEMAP_WIDTH),
                (CUBEMAP_WIDTH, CUBEMAP_WIDTH),
                &mut slice[CUBEMAP_WIDTH * CUBEMAP_WIDTH * face..][..CUBEMAP_WIDTH * CUBEMAP_WIDTH],
                None,
            )
            .unwrap();
    }

    // for (index, pixel) in slice.iter().enumerate() {
    //     let time_elapsed = instant.elapsed().as_secs_f64();
    //     if time_elapsed > 1.0 {
    //         instant = std::time::Instant::now();
    //         println!("Percent done file: {:}", index as f64 / num_pixels as f64);
    //     };
    //     let longitude = index % raster_band.size().0;
    //     let latitude = index / raster_band.size().0;
    //     // dbg!(longitude);
    //     // dbg!(latitude);
    //     let transformed_longitude = (transform[0]
    //         + longitude as f64 * transform[1]
    //         + latitude as f64 * transform[2])
    //         .to_radians();
    //     let transformed_latitude = (transform[3]
    //         + longitude as f64 * transform[4]
    //         + latitude as f64 * transform[5])
    //         .to_radians();
    //     // dbg!(transformed_longitude);
    //     // dbg!(transformed_latitude);
    //     let raster_coordinate = Vector3::new(
    //         transformed_latitude.cos() * transformed_longitude.cos(),
    //         transformed_latitude.sin(),
    //         transformed_latitude.cos() * transformed_longitude.sin(),
    //     );
    //     // dbg!(raster_coordinate);
    //     // let closest_index = samples.iter().enumerate().min_by_key(|&(index, _)|{
    //     //     let coordinate = index_to_coordinate(index).normalize().cast();
    //     //     FloatOrd((raster_coordinate - coordinate).magnitude())
    //     // }).unwrap().0;
    //     let mut closest_index = coordinate_to_index(raster_coordinate);
    //     if closest_index >= CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6 {
    //         dbg!(closest_index);
    //         closest_index = CUBEMAP_WIDTH * CUBEMAP_WIDTH * 6 - 1;
    //     }
    //     // dbg!(closest_index);
    //     // dbg!(index_to_coordinate(closest_index).normalize());
    //     // dbg!(pixel);
    //     let num_samples = samples[closest_index].1;
    //     match raster_band.no_data_value() {
    //         Some(no_data_value) => {
    //             if *pixel != no_data_value.convert() {
    //                 samples[closest_index].0 += additional_processing(*pixel, num_samples);
    //                 samples[closest_index].1 += 1;
    //             }
    //         }
    //         None => {
    //             samples[closest_index].0 += additional_processing(*pixel, num_samples);
    //             samples[closest_index].1 += 1;
    //         }
    //     }
    // }

    // let num_samples = samples.len();
    // for (i, (sample, sample_count)) in samples.iter_mut().enumerate() {
    //     // if *sample_count != 0{
    //     //     continue;
    //     // }
    //     let time_elapsed = instant.elapsed().as_secs_f64();
    //     if time_elapsed > 1.0 {
    //         instant = std::time::Instant::now();
    //         println!("Percent done file: {:}", i as f64 / num_samples as f64);
    //     };
    //     let coordinate = index_to_coordinate(i).cast() / World::RADIUS;
    //     // dbg!(coordinate);

    //     let longitude = coordinate.z.atan2(coordinate.x).to_degrees();
    //     let latitude = coordinate.y.asin().to_degrees();
    //     if latitude.abs() > 90.0 || longitude.abs() > 180.0 {
    //         dbg!(longitude, latitude);
    //     }

    //     // let transformed_coords = if spatial_ref.name().unwrap() != wgs_84.name().unwrap() {
    //     //     let coord_transform =
    //     //         gdal::spatial_ref::CoordTransform::new(&wgs_84, &spatial_ref).unwrap();
    //     //     let mut x = [longitude];
    //     //     let mut y = [latitude];
    //     //     let mut z = [];
    //     //     match coord_transform.transform_coords(&mut x, &mut y, &mut z) {
    //     //         Ok(_) => {}
    //     //         Err(error) => {
    //     //             match error {
    //     //                 GdalError::InvalidCoordinateRange { from, to, msg } => {dbg!(from,to, msg);}
    //     //                 _ => {
    //     //                     println!("Unknown Transform Coords error: {:?}", error);
    //     //                 }
    //     //             }
    //     //             continue;
    //     //         }
    //     //     }
    //     //     (x[0], y[0])
    //     // } else {
    //     //     (longitude, latitude)
    //     // };
    //     let transformed_coords = (longitude, latitude);
    //     // dbg!(transformed_coords);

    //     let x = inv_transform[0]
    //         + transformed_coords.0 * inv_transform[1]
    //         + transformed_coords.1 * inv_transform[2];
    //     let y = inv_transform[3]
    //         + transformed_coords.0 * inv_transform[4]
    //         + transformed_coords.1 * inv_transform[5];

    //     // dbg!((x,y));
    //     // let x = ((x.to_radians() / (2.0 * PI) + 0.5) * (raster_band.size().0 - 1) as f64) as usize;
    //     // let y = (((y.to_radians() / PI) + 0.5) * (raster_band.size().1 - 1) as f64) as usize;
    //     // dbg!(x,y);

    //     let old_x = x;
    //     let old_y = y;
    //     let x = (x as usize).min(raster_band.size().0 - 1);
    //     let y = (y as usize).min(raster_band.size().1 - 1);
    //     // dbg!((x,y));
    //     let raster_index = x + y * (raster_band.size().0);
    //     if raster_index > slice.len() {
    //         dbg!(x, y);
    //         dbg!(old_x, old_y);
    //         dbg!(coordinate);
    //         dbg!(
    //             latlong_to_vector(transformed_coords.1, transformed_coords.0)
    //                 / World::RADIUS
    //         );
    //     }
    //     let raster_value = slice[raster_index];
    //     let mut no_data = false;
    //     match raster_band.no_data_value() {
    //         Some(no_data_value) => {
    //             no_data = raster_value == no_data_value.convert();
    //         }
    //         None => {}
    //     }
    //     if !no_data {
    //         *sample += additional_processing(raster_value, *sample_count);
    //         *sample_count += 1;
    //     }
    // }
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
