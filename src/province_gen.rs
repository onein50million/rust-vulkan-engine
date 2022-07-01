// use std::collections::HashSet;

// use nalgebra::Vector3;

// use crate::planet_gen::{are_points_contiguous, MeshOutput};

// // pub fn print_duration(last_instant: &mut Instant, message: &str){
// //     println!("{:}{:}", last_instant.elapsed().as_secs_f64(), message);
// //     *last_instant = Instant::now();
// // }

// pub struct Islands<'a> {
//     pub islands: Vec<HashSet<usize>>,
//     points: &'a [(Vector3<f32>, f32)],
//     planet_mesh_vertices: Vec<(Vector3<f32>, f32)>,
//     planet_mesh_indices: Vec<usize>,
//     completed_up_to: usize,
// }

// impl<'a> Islands<'a> {
//     pub fn new(province_points: &'a [(Vector3<f32>, f32)], planet_mesh: &MeshOutput) -> Self {
//         let islands: Vec<_> = province_points
//             .iter()
//             .enumerate()
//             .map(|(index, _value)| {
//                 let mut h = HashSet::new();
//                 h.insert(index);
//                 h
//             })
//             .collect();
//         Self {
//             islands,
//             points: province_points,
//             planet_mesh_vertices: planet_mesh
//                 .vertices
//                 .iter()
//                 .map(|&vertex| (vertex.position, vertex.elevation))
//                 .collect(),
//             planet_mesh_indices: planet_mesh.indices.clone(),
//             completed_up_to: 0,
//         }
//     }
//     pub fn collapse_islands(&mut self) -> bool {
//         dbg!(&self.islands.len());
//         for island_index in self.completed_up_to..self.islands.len() {
//             // for other_island_index in self.islands.iter().enumerate().filter(|(other_island_index, _)|{
//             //     *other_island_index != island_index
//             // }).map(|(other_island_index, _)|other_island_index){
//             for other_island_index in (island_index + 1)..self.islands.len() {
//                 let point_index = *self.islands[island_index].iter().next().unwrap();
//                 // for &point_index in &self.islands[island_index]{
//                 for &other_point_index in &self.islands[other_island_index] {
//                     let contiguous = are_points_contiguous(
//                         self.points[point_index].0,
//                         self.points[other_point_index].0,
//                         &self.planet_mesh_vertices,
//                         &self.planet_mesh_indices,
//                     );

//                     if contiguous {
//                         // dbg!(other_island_index);
//                         self.combine_two_islands(island_index, other_island_index);
//                         return false;
//                     }
//                 }
//                 // }
//             }
//             self.completed_up_to += 1;
//         }

//         true
//     }

//     fn combine_two_islands(&mut self, first_index: usize, second_index: usize) {
//         assert_ne!(first_index, second_index);

//         let (first_island, second_island) = self.get_two_islands(first_index, second_index);

//         first_island.extend(second_island.iter());
//         self.islands.remove(second_index);
//     }

//     fn get_two_islands(
//         &mut self,
//         first_index: usize,
//         second_index: usize,
//     ) -> (&mut HashSet<usize>, &mut HashSet<usize>) {
//         unsafe {
//             return (
//                 &mut *(&mut self.islands[first_index] as *mut _),
//                 &mut *(&mut self.islands[second_index] as *mut _),
//             );
//         }
//     }

// }
