use crate::{support::Vertex, world::World};
use float_ord::FloatOrd;
use gdal::errors::GdalError;
use genmesh::generators::{IcoSphere, IndexedPolygon, SharedVertex};
use nalgebra::{Vector2, Vector3, Vector4};
use std::{
    collections::{HashMap, HashSet},
    fs::read_dir, ops::{Add, Mul}, fmt::Debug, time::Instant,
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
// //https://answers.unity.com/questions/383804/calculate-uv-coordinates-of-3d-point-on-plane-of-m.html
// pub fn triangle_interpolate<T>(target_point: Vector3<f32>,first_point: (Vector3<f32>, T),second_point: (Vector3<f32>, T),third_point: (Vector3<f32>, T)) -> Option<T>
// where T:  Mul<f32, Output = T> + Add<Output = T> + Copy
// {
//     let f1 = first_point.0 - target_point;
//     let f2 = second_point.0 - target_point;
//     let f3 = third_point.0 - target_point;

//     let total_area = dbg!((first_point.0 - second_point.0).cross(&(first_point.0 - third_point.0)).magnitude());

//     let ratio1 = dbg!(f2.cross(&f3).magnitude()/total_area);
//     let ratio2 = dbg!(f3.cross(&f1).magnitude()/total_area);
//     let ratio3 = dbg!(f1.cross(&f2).magnitude()/total_area);


//     if ratio1 > 0.0 && ratio2 > 0.0 && ratio3 > 0.0{
//         Some(first_point.1 * ratio1 + second_point.1 * ratio2 + third_point.1 * ratio3)
//     }else{
//         None
//     }

// }

//https://answers.unity.com/questions/383804/calculate-uv-coordinates-of-3d-point-on-plane-of-m.html
pub fn triangle_interpolate<T>(target_point: Vector3<f32>,first_point: (Vector3<f32>, T),second_point: (Vector3<f32>, T),third_point: (Vector3<f32>, T)) -> Option<T>
where T:  Mul<f32, Output = T> + Add<Output = T> + Copy
{
    let f1 = target_point - first_point.0;
    let f2 = target_point - second_point.0;
    let f3 = target_point - third_point.0;


    let main_cross = (first_point.0 - second_point.0).cross(&(first_point.0 - third_point.0));
    let cross1 = f2.cross(&f3);
    let cross2 = f3.cross(&f1);
    let cross3 = f1.cross(&f2);
    
    let total_area = main_cross.magnitude();
    let ratio1 = cross1.magnitude()/total_area * main_cross.dot(&cross1).signum();
    let ratio2 = cross2.magnitude()/total_area * main_cross.dot(&cross2).signum();
    let ratio3 = cross3.magnitude()/total_area * main_cross.dot(&cross3).signum();

    if ratio1 >= -0.1 && ratio2 >= -0.1 && ratio3 >= -0.1 && ratio1 <= 1.0 && ratio2 <= 1.0 && ratio3 <= 1.0 {
        Some(first_point.1 * ratio1 + second_point.1 * ratio2 + third_point.1 * ratio3)
    }else{
        None
    }

}

pub fn interoplate_on_mesh<T>(point: Vector3<f32>, vertices: &[(Vector3<f32>, T)], indices: &[usize]) -> T
where T:  Mul<f32, Output = T> + Add<Output = T> + Copy + Debug
{
    for triangle in indices.chunks(3){
        let first_point = vertices[triangle[0]];
        let second_point = vertices[triangle[1]];
        let third_point = vertices[triangle[2]];
        
        if let Some(value) = triangle_interpolate(point,first_point, second_point, third_point){
            return value;
        }
    }
    panic!("Failed to find point on mesh: point: {point}")
}

// pub fn are_points_contiguous(point1: Vector3<f32>, point2: Vector3<f32>, vertices: &[(Vector3<f32>, f32)], indices: &[usize]) -> bool{
//     const SAMPLES_PER_RADIAN: usize = 100;
//     let angle = point1.angle(&point2);
//     let num_samples = ((angle * SAMPLES_PER_RADIAN as f32) as usize).max(1);

//     for sample in 0..num_samples{
//         let progress = sample as f32 / num_samples as f32;

//         let sample_vector = point1.slerp(&point2, progress) * World::RADIUS as f32;
//         let interpolate = interoplate_on_mesh(sample_vector, vertices, indices);

        
//         if interpolate < 0.0{
//             return false;
//         }
//     }
//     true
// }

pub fn are_points_contiguous(point1: Vector3<f32>, point2: Vector3<f32>, vertices: &[(Vector3<f32>, f32)], indices: &[usize]) -> bool{
    const NUM_SAMPLES: usize = 100;
    for sample in 0..NUM_SAMPLES{
        let progress = (sample as f32 + 0.5) / NUM_SAMPLES as f32;

        let sample_vector = point1.slerp(&point2, progress) * World::RADIUS as f32;
        let interpolate = interoplate_on_mesh(sample_vector, vertices, indices);

        
        if interpolate < 0.0{
            return false;
        }
    }
    true
}

const NUM_BUCKETS: f32 = 8.0; //Number of buckets to quantize elevations into

fn quantized_to_actual(quantized: i8, range: f32) -> f32{
    (quantized as f32 / NUM_BUCKETS) * range

}

pub fn get_planet(radius: f32) -> MeshOutput {
    let sphere = IcoSphere::subdivide(3);

    let _rng = fastrand::Rng::new();
    let mut vertices: Vec<_> = sphere
        .shared_vertex_iter()
        .map(|vertex| Vertex {
            position: Vector3::new(vertex.pos.x, vertex.pos.y, vertex.pos.z),
            normal: Vector3::new(vertex.pos.x, vertex.pos.y, vertex.pos.z).normalize(),
            tangent: Vector4::zeros(),
            texture_coordinate: Vector2::zeros(),
            texture_type: 0,
            bone_indices: Vector4::zeros(),
            bone_weights: Vector4::zeros(),
            elevation: 0.0,
        })
        .collect();

    let paths = std::fs::read_dir("../GSG/elevation_gebco").unwrap();

    let wgs_84 = gdal::spatial_ref::SpatialRef::from_epsg(4326).unwrap();

    println!("Loading elevation data");
    let mut raw_vertex_data = vec![
        RawVertexData {
            sum: 0.0,
            sample_count: 0,
            data_priority: 0
        };
        vertices.len()
    ];
    for folder in paths.filter_map(|s| s.ok()) {
        let data_priority = if folder.file_name().to_str().unwrap() == "arctic" {
            1
        } else {
            0
        };

        let image_files = read_dir(folder.path()).unwrap();
        let path = image_files
            .filter_map(|s| s.ok())
            .find(|image_file| {
                if let Ok(file_name) = image_file.file_name().into_string() {
                    file_name.to_lowercase().contains("mea")
                } else {
                    false
                }
            })
            .unwrap()
            .path();

        let dataset = gdal::Dataset::open(path).unwrap();
        let mut transform = dataset.geo_transform().unwrap();
        let mut inv_transform = [0.0f64; 6];
        unsafe {
            assert_eq!(
                gdal_sys::GDALInvGeoTransform(transform.as_mut_ptr(), inv_transform.as_mut_ptr(),),
                1,
                "InvGeoTransform failed"
            );
        }
        let rasterband = dataset.rasterband(1).unwrap();
        let no_data_value = rasterband.no_data_value().unwrap();
        let spatial_ref = dataset.spatial_ref().unwrap();
        let mut instant = std::time::Instant::now();

        for (i, (vertex, raw_vertex)) in vertices.iter().zip(raw_vertex_data.iter_mut()).enumerate()
        {
            let time_elapsed = instant.elapsed().as_secs_f64();
            if time_elapsed > 1.0 {
                instant = std::time::Instant::now();
                println!("Percent done file: {:}", i as f64 / vertices.len() as f64);
            };

            let latitude = (vertex.position.y.asin() as f64).to_degrees();

            let longitude = (vertex.position.z.atan2(vertex.position.x) as f64).to_degrees();

            let transformed_coords = if spatial_ref.name().unwrap() != wgs_84.name().unwrap() {
                let coord_transform =
                    gdal::spatial_ref::CoordTransform::new(&wgs_84, &spatial_ref).unwrap();
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
                        if slice[0] != no_data_value && data_priority >= raw_vertex.data_priority {
                            if data_priority > raw_vertex.data_priority {
                                raw_vertex.sum = 0.0;
                                raw_vertex.sample_count = 0;
                                raw_vertex.data_priority = data_priority;
                            }
                            raw_vertex.sum += slice[0];
                            raw_vertex.sample_count += 1;
                        }
                    }
                }
            }
        }
    }

    for (vertex, raw_vertex) in vertices.iter_mut().zip(raw_vertex_data.iter()) {
        if raw_vertex.sample_count > 0 {
            vertex.elevation = (raw_vertex.sum / raw_vertex.sample_count as f64) as f32;
        }
        // vertex.elevation = 1000.0;
        // vertex.position *= radius;
    }

    let indices: Vec<_> = sphere
        .indexed_polygon_iter()
        .map(|triangle| [triangle.x, triangle.y, triangle.z].into_iter())
        .flatten()
        .collect();

    let mut vertex_data: Vec<_> = vertices
        .iter()
        .enumerate()
        .map(|(_index, vertex)| {
            const NUM_NEIGHBOURS: usize = 6;
            let mut closest_vertices = vertices.iter().enumerate().collect::<Vec<_>>();

            closest_vertices.sort_by_key(|potential_closest| {
                FloatOrd((potential_closest.1.position - vertex.position).magnitude())
            });

            VertexData {
                elevation: vertex.elevation,
                quantized_elevation: 0,
                neighbours: closest_vertices
                    .iter()
                    .map(|(index, _)| *index)
                    .take(NUM_NEIGHBOURS)
                    .collect(),
            }
        })
        .collect();

    dbg!(&vertex_data[5]);

    for vertex in &mut vertex_data{
        let divisor = 2500.0;
        if vertex.elevation < 0.0{
            vertex.elevation = (vertex.elevation/divisor).ceil()*divisor;
        }
    }

    let min_elevation = vertex_data
        .iter()
        .fold(f32::INFINITY, |a, b| a.min(b.elevation));
    let max_elevation = vertex_data
        .iter()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b.elevation));

    for vertex in &mut vertex_data {
        vertex.quantized_elevation =
            ((vertex.elevation / max_elevation.max(min_elevation.abs())) * NUM_BUCKETS) as i8
    }

    let mut quantized_elevations = HashSet::new();
    for vertex in &vertex_data{
        quantized_elevations.insert(vertex.quantized_elevation);
    }
    let mut quantized_elevations: Vec<_> = quantized_elevations.iter().collect();
    quantized_elevations.sort();
    for elevation in quantized_elevations{
        let actual = quantized_to_actual(*elevation, max_elevation.max(min_elevation.abs()));
        println!("Quantized: {elevation} Actual: {actual}");
    }

    let mut start_instant = std::time::Instant::now();
    let mut current_try = 0;
    'wave_function: loop {
        current_try += 1; 
        println!("Current try: {:}", current_try);
        let mut wave_function_collapse = ElevationWaveFunctionCollapse::new(&vertex_data);
    
        let mut num_collapses = 0;
        loop {

            let mut lowest_entropy_cells = vec![];
            let mut lowest_entropy = wave_function_collapse.get_cell_entropy(0);
            let mut unfinished_cell_indices = HashSet::new();
            for (index, (vertex, cell)) in vertices
                .iter_mut()
                .zip(wave_function_collapse.cells.iter())
                .enumerate()
            {
                let cell_entropy = wave_function_collapse.get_cell_entropy(index);
                if cell_entropy > lowest_entropy{
                    lowest_entropy = cell_entropy;
                    lowest_entropy_cells = vec![index];
                }else if cell_entropy == lowest_entropy{
                    lowest_entropy_cells.push(index);
                }

                match cell.cell_state {
                    CellState::Collapsed(quantized_elevation) => {
                        // vertex.elevation = (quantized_elevation as f32 / 4.0)
                        //     * max_elevation.max(min_elevation.abs());
                        vertex.elevation = quantized_to_actual(quantized_elevation, max_elevation.max(min_elevation.abs()));
                        // vertex.position += vertex.position.normalize() * vertex.elevation * 1.0;
                    }
                    CellState::Superposition(_) => {
                        unfinished_cell_indices.insert(index);
                    }
                }
            }
            if start_instant.elapsed().as_secs_f64() > 1.0{
                start_instant = std::time::Instant::now();
                println!("{:} collapses to go", unfinished_cell_indices.len())
            }
            if unfinished_cell_indices.len() > 0 {
                let rand_index = fastrand::usize(0..lowest_entropy_cells.len());
                match wave_function_collapse
                    .collapse_cell(*lowest_entropy_cells.iter().nth(rand_index).unwrap()){
                        Ok(_) => {
                            num_collapses += 1;
                        },
                        Err(_) => {
                            println!("Wave function collapse failed");
                            println!("Made it {:} collapses", num_collapses);
                            continue 'wave_function;
                        },
                    }
            } else {
                break 'wave_function;
            }
        }
    }

    for vertex in &mut vertices {
        vertex.normal = Vector3::zeros();
        vertex.position *= radius + vertex.elevation * 1.0;
    }

    for triangle_index in (0..indices.len()).step_by(3) {
        let edge1 = vertices[indices[triangle_index] as usize].position
            - vertices[indices[triangle_index + 1] as usize].position;
        let edge2 = vertices[indices[triangle_index] as usize].position
            - vertices[indices[triangle_index + 2] as usize].position;
        let weighted_normal = edge1.cross(&edge2);
        for i in 0..3 {
            vertices[indices[triangle_index + i] as usize].normal += weighted_normal
        }
    }
    for vertex in &mut vertices {
        vertex.normal = vertex.normal.normalize();
    }

    return MeshOutput { vertices, indices };
}

#[derive(Clone, Copy, Debug)]
enum Error{
    PropogationFailed
}


#[derive(Clone, Debug)]
struct Cell {
    cell_state: CellState,
    neighbours: Vec<usize>,
}

#[derive(Clone, Debug)]
enum CellState {
    Collapsed(i8),              //Collapsed cell has only one value
    Superposition(HashMap<i8, f32>), //Cell is in a superposition of all possible values
}

#[derive(Clone, Copy, Debug)]
enum Ratio{
    RawSamples{
        sample_count: u32
    },
    Data(f32)
}
impl Ratio{
    fn add_sample(&mut self){
        match self{
            Ratio::RawSamples {sample_count } => {
                *sample_count += 1;
            },
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

fn balance_ratios(hashmap: &mut HashMap<i8, f32>){
    
    let ratio_sum: f32 = hashmap.values().sum();
    assert_ne!(ratio_sum, 0.0);

    for ratio in hashmap.values_mut(){
        *ratio /= ratio_sum;
    }
}

#[derive(Clone, Debug)]
struct ElevationWaveFunctionCollapse {
    cells: Vec<Cell>,
    elevation_mapping: HashMap<i8, HashMap<i8, Ratio>>, //given quantized elevation it gives you the possible states and their probabilities
}

impl ElevationWaveFunctionCollapse {
    fn new(vertex_data: &[VertexData]) -> Self {
        let mut elevation_mapping = HashMap::new();
        for i in 0..vertex_data.len() {
            let inner_map = elevation_mapping
                .entry(vertex_data[i].quantized_elevation)
                .or_insert(HashMap::new());
            for neighbour in &vertex_data[i].neighbours {
                let entry = inner_map.entry(vertex_data[*neighbour].quantized_elevation).or_insert(Ratio::RawSamples {sample_count: 0 });
                entry.add_sample();
                // if !inner_map.contains(&vertex_data[*neighbour].quantized_elevation) {
                //     inner_map.insert(vertex_data[*neighbour].quantized_elevation);
                // }
            }
        }

        for raw_samples in elevation_mapping.values_mut(){
            let total: f32 = raw_samples.values().map(|sample_set|{
                match sample_set{
                    Ratio::RawSamples {sample_count } => {
                        *sample_count as f32
                    },
                    Ratio::Data(ratio) => {
                        *ratio
                    },
                }
            }).sum();

            for raw_sample in raw_samples.values_mut(){
                let new_value = match raw_sample{
                    Ratio::RawSamples {sample_count } => {
                        (*sample_count as f32)/total
                    },
                    Ratio::Data(_) => {
                        panic!("Data already averaged")
                    },
                };

                *raw_sample = Ratio::Data(new_value);
            }
        }

        // dbg!(raw_samp)
        
        let cells: Vec<_> = vertex_data
            .iter()
            .map(|vertex| {
                let mut superpositions = HashMap::new();

                for ratio_map in elevation_mapping.values() {
                    for (potential_elevation, ratio) in ratio_map{
                        if let Ratio::Data(ratio) = ratio{
                            let entry = superpositions.entry(*potential_elevation).or_insert(0.0);
                            *entry += ratio;
                        }else{
                            panic!("Data not averaged")
                        }
                    }
                }

                balance_ratios(&mut superpositions);

                Cell {
                    cell_state: CellState::Superposition(superpositions),
                    neighbours: vertex.neighbours.clone(),
                }
            })
            .collect();

        let out =Self {
            cells,
            elevation_mapping,
        };
        out    

    }


    //will be unbalanced
    fn get_possible_states(&self, index: usize) -> HashMap<i8, f32> {
        match &self.cells[index].cell_state {
            CellState::Collapsed(state) => {
                HashMap::from_iter(self.elevation_mapping[&state].iter().map(|(key, value)|{
                    match value{
                        Ratio::RawSamples{..} => panic!("Unfinished ratio map"),
                        Ratio::Data(new_value) => (*key, *new_value),
                    }
                }))
            },
            CellState::Superposition(states) => {
                let mut output = HashMap::new();
                for (quantized_elevation, ratio) in states{
                    for (possible_quantized_elevation, possible_ratio) in &self.elevation_mapping[quantized_elevation]{
                        let entry = output.entry(*possible_quantized_elevation).or_insert(0.0);
                        if let Ratio::Data(possible_ratio) = possible_ratio{
                            *entry += ratio * possible_ratio;  
                        }
                    }
                }
                // balance_ratios(&mut output);
                output
            }
        }
    }

    fn propogate_changes(&mut self) -> Result<(), Error> {
        let mut done = true;
        loop {
            // println!("propogate changes loop");
            // dbg!(self.cells.len());
            for i in 0..self.cells.len() {
                let new_states = match &self.cells[i].cell_state {
                    CellState::Collapsed(_) => continue,
                    CellState::Superposition(states) => {
                        // dbg!(states.len());
                        let mut neighbour_states = HashMap::new();
                        for neighbour_index in &self.cells[i].neighbours {
                            for (neighbour_state_elevation, neighbour_state_ratio) in self.get_possible_states(*neighbour_index).iter(){
                                let entry = neighbour_states.entry(*neighbour_state_elevation).or_insert(0.0);
                                *entry += neighbour_state_ratio * states[neighbour_state_elevation];
                            }
                        }
                        balance_ratios(&mut neighbour_states);
                        const EPSILON: f32 = 0.001;

                        // dbg!(&neighbour_states);
                        // dbg!(&states);
                        for (neighbour_quantized_elevation, neighbour_state_ratio) in &neighbour_states{
                            let state_ratio = states[neighbour_quantized_elevation];
                            if (state_ratio - *neighbour_state_ratio).abs() > EPSILON{
                                done = false;
                            }
                        }

                        

                        // let out: HashSet<_> = neighbour_states
                        //     .intersection(states)
                        //     .map(|&state| state)
                        //     .collect();
                        // if &out != states {
                        //     done = false;
                        // }
                        neighbour_states
                    }
                };

                // dbg!(&new_states);

                if new_states.len() == 0 {
                    return Err(Error::PropogationFailed)
                } else if new_states.len() == 1 {
                    self.cells[i].cell_state =
                        CellState::Collapsed(*new_states.iter().next().unwrap().0)
                } else {
                    self.cells[i].cell_state = CellState::Superposition(new_states)
                }
            }
            if done {
                break;
            }
            done = true;
        }
        Ok(())
    }

    fn get_cell_entropy(&self, index: usize) -> usize{
        match &self.cells[index].cell_state{
            CellState::Collapsed(_) => {
                1
            },
            CellState::Superposition(states) => {
                states.iter().count()
            },
        }
    }

    fn collapse_cell(&mut self, index: usize) -> Result<(), Error> {
        let mut collapsed_value = None;
        match &self.cells[index].cell_state {
            CellState::Collapsed(_) => return Ok(()),
            CellState::Superposition(superpositions) => {
                let random = fastrand::f32();
                let mut probability_sum = 0.0;
                for (quantized_elevation, probability) in superpositions{
                    probability_sum += probability;
                    if random <= probability_sum{
                        collapsed_value = Some(*quantized_elevation)
                    }
                }
            }
        }
        self.cells[index].cell_state = CellState::Collapsed(collapsed_value.expect("Cell Collapse failed. This probably shouldn't happen... maybe"));
        self.propogate_changes()
    }
    fn collapse_cell_to_value(&mut self, index: usize, value: i8) -> Result<(), Error> {
        self.cells[index].cell_state = CellState::Collapsed(value);
        self.propogate_changes()
    }
}
