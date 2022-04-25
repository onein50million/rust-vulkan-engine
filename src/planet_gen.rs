use crate::support::Vertex;
use float_ord::FloatOrd;
use gdal::errors::GdalError;
use genmesh::generators::{IcoSphere, IndexedPolygon, SharedVertex};
use nalgebra::{Vector2, Vector3, Vector4};
use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    fs::read_dir,
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

pub fn get_planet(radius: f32) -> MeshOutput {
    let sphere = IcoSphere::subdivide(3);

    let rng = fastrand::Rng::new();
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
        .map(|(index, vertex)| {
            const NUM_NEIGHBOURS: usize = 3;
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

    let min_elevation = vertices
        .iter()
        .fold(f32::INFINITY, |a, &b| a.min(b.elevation));
    let max_elevation = vertices
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b.elevation));

    for vertex in &mut vertex_data {
        vertex.quantized_elevation =
            ((vertex.elevation / max_elevation.max(min_elevation.abs())) * 128.0) as i8
    }

    let mut wave_function_collapse = ElevationWaveFunctionCollapse::new(&vertex_data);

    wave_function_collapse.collapse_cell(0);

    loop {
        let mut unfinished_cell_indices = HashSet::new();
        for (index, (vertex, cell)) in vertices
            .iter_mut()
            .zip(wave_function_collapse.cells.iter())
            .enumerate()
        {
            match cell.cell_state {
                CellState::Collapsed(quantized_elevation) => {
                    vertex.elevation = (quantized_elevation as f32 / 128.0)
                        * max_elevation.max(min_elevation.abs());
                    // vertex.position += vertex.position.normalize() * vertex.elevation * 1.0;
                }
                CellState::Superposition(_) => {
                    unfinished_cell_indices.insert(index);
                }
            }
        }
        if unfinished_cell_indices.len() > 0 {
            let rand_index = fastrand::usize(0..unfinished_cell_indices.len());
            wave_function_collapse
                .collapse_cell(*unfinished_cell_indices.iter().nth(rand_index).unwrap());
        } else {
            break;
        }
    }

    for vertex in &mut vertices {
        vertex.normal = Vector3::zeros();
        vertex.position *= radius + vertex.elevation * 100.0;
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

#[derive(Clone, Debug)]
struct Cell {
    cell_state: CellState,
    neighbours: Vec<usize>,
}

#[derive(Clone, Debug)]
enum CellState {
    Collapsed(i8),              //Collapsed cell has only one value
    Superposition(HashSet<i8>), //Cell is in a superposition of all possible values
}

#[derive(Clone, Debug)]
struct ElevationWaveFunctionCollapse {
    cells: Vec<Cell>,
    elevation_mapping: HashMap<i8, HashSet<i8>>, //given quantized elevation it gives you the possible states
}

impl ElevationWaveFunctionCollapse {
    fn new(vertex_data: &[VertexData]) -> Self {
        let mut elevation_mapping = HashMap::new();
        for i in 0..vertex_data.len() {
            let inner_map = elevation_mapping
                .entry(vertex_data[i].quantized_elevation)
                .or_insert(HashSet::new());
            for neighbour in &vertex_data[i].neighbours {
                if !inner_map.contains(&vertex_data[*neighbour].quantized_elevation) {
                    inner_map.insert(vertex_data[*neighbour].quantized_elevation);
                }
            }
        }
        let cells: Vec<_> = vertex_data
            .iter()
            .map(|vertex| {
                let mut superpositions = HashSet::new();

                for quantized_elevation in elevation_mapping.keys() {
                    superpositions.insert(*quantized_elevation);
                }

                Cell {
                    cell_state: CellState::Superposition(superpositions),
                    neighbours: vertex.neighbours.clone(),
                }
            })
            .collect();

        Self {
            cells,
            elevation_mapping,
        }
    }

    fn get_possible_states(&self, index: usize) -> HashSet<i8> {
        match &self.cells[index].cell_state {
            CellState::Collapsed(state) => self.elevation_mapping[&state].clone(),
            CellState::Superposition(states) => states
                .iter()
                .map(|state| self.elevation_mapping[state].clone())
                .flatten()
                .collect(),
        }
    }

    fn propogate_changes(&mut self) {
        let mut done = true;
        loop {
            // println!("propogate changes loop");
            // dbg!(self.cells.len());
            for i in 0..self.cells.len() {
                let new_states = match &self.cells[i].cell_state {
                    CellState::Collapsed(_) => continue,
                    CellState::Superposition(states) => {
                        // dbg!(states.len());
                        let mut new_states = HashSet::new();
                        for neighbour_index in &self.cells[i].neighbours {
                            new_states.extend(self.get_possible_states(*neighbour_index).iter());
                        }
                        let out: HashSet<_> = new_states
                            .intersection(states)
                            .map(|&state| state)
                            .collect();
                        if &out != states {
                            done = false;
                        }
                        out
                    }
                };

                // dbg!(&new_states);

                if new_states.len() == 0 {
                    panic!("Wave function propogation failed")
                } else if new_states.len() == 1 {
                    self.cells[i].cell_state =
                        CellState::Collapsed(*new_states.iter().next().unwrap())
                } else {
                    self.cells[i].cell_state = CellState::Superposition(new_states)
                }
            }
            if done {
                break;
            }
            done = true;
        }
    }
    fn collapse_cell(&mut self, index: usize) {
        match &self.cells[index].cell_state {
            CellState::Collapsed(_) => return,
            CellState::Superposition(superpositions) => {
                let random = fastrand::usize(0..superpositions.len());
                self.cells[index].cell_state =
                    CellState::Collapsed(*superpositions.iter().nth(random).unwrap());
                self.propogate_changes();
            }
        }
    }
}
