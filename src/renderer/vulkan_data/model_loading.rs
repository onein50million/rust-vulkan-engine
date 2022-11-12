use std::{path::{PathBuf, Path}, f32::consts::PI};

use erupt::vk;
use gltf::animation::util::{ReadOutputs, Rotations};
use nalgebra::{Matrix4, UnitQuaternion, Vector3, Point3, Vector4, Vector2, Translation3, Quaternion, Scale3};

use crate::{renderer::{combination_types::{CombinedSampledImage, TextureSet}, animation::{AnimationObject, AnimationKeyframes, Keyframe}, drawables::RenderObject}, support::{Vertex, Bone}};

use super::VulkanData;

impl VulkanData{
    pub fn load_folder(&mut self, folder: PathBuf) -> usize {
        let albedo_path = folder.join("albedo.png");
        let albedo = CombinedSampledImage::new(
            self,
            albedo_path,
            vk::ImageViewType::_2D,
            vk::Format::R8G8B8A8_SRGB,
            false,
        );

        let normal_path = folder.join("normal.png");
        let normal = CombinedSampledImage::new(
            self,
            normal_path,
            vk::ImageViewType::_2D,
            vk::Format::R8G8B8A8_UNORM,
            false,
        );

        let roughness_path = folder.join("rough_metal_ao.png");
        let rough_metal_ao = CombinedSampledImage::new(
            self,
            roughness_path,
            vk::ImageViewType::_2D,
            vk::Format::R8G8B8A8_UNORM,
            false,
        );

        let texture = TextureSet {
            albedo,
            normal,
            roughness_metalness_ao: rough_metal_ao,
        };
        let (vertices, indices, animations) = if folder.join("model.glb").is_file() {
            Self::load_gltf_model(folder.join("model.glb"))
        } else {
            (vec![], vec![], vec![])
        };
        let mut out_animations = vec![];
        for animation in animations {
            let num_bone_sets = animation.len();
            let frame_start = self.current_boneset;
            for (bone_set_index, bones) in animation.into_iter().enumerate() {
                for (bone_index, bone) in bones.into_iter().enumerate() {
                    self.storage_buffer_object.bone_sets[self.current_boneset + bone_set_index]
                        .bones[bone_index] = bone;
                }
            }
            self.current_boneset += num_bone_sets;
            out_animations.push(AnimationObject {
                frame_start,
                frame_count: num_bone_sets,
            })
        }
        let render_object =
            RenderObject::new(self, vertices, indices, out_animations, texture, false);
        let output = self.objects.len();
        self.objects.push(render_object);

        return output;
    }

    pub fn load_vertices_and_indices(
        &mut self,
        vertices: Vec<Vertex>,
        indices: Vec<usize>,
        is_globe: bool,
    ) -> usize {
        let indices = indices.iter().map(|index| *index as u32).collect();
        let mut render_object = RenderObject::new(
            self,
            vertices,
            indices,
            vec![],
            TextureSet::new_empty(),
            false,
        );
        render_object.is_globe = is_globe;
        let output = self.objects.len();
        self.objects.push(render_object);
        return output;
    }

    pub(crate) fn load_gltf_model(
        path: std::path::PathBuf,
    ) -> (Vec<Vertex>, Vec<u32>, Vec<Vec<Vec<Bone>>>) {
        println!("loading {:}", path.to_str().unwrap());
        let (gltf, buffers, _) = gltf::import(path).unwrap();
        // let materials = material_result.unwrap();

        let mut out_vertices = vec![];
        let mut out_indices = vec![];
        let mut out_animations = vec![];
        // let mut positions = vec![];

        let root_node = gltf.scenes().nth(0).unwrap().nodes().nth(0).unwrap();

        let mut mesh_transform = Matrix4::identity();
        // let vulkan_correction_transform =
        //     Matrix4::from_axis_angle(&Vector3::x_axis(), std::f32::consts::PI);
        let vulkan_correction_transform =
            Matrix4::from(UnitQuaternion::from_euler_angles(0.0, 0.0, PI));
        // let vulkan_correction_transform = Matrix4::identity();
        // let vulkan_correction_transform = Scale3::new(1.0,-1.0,1.0).to_homogeneous();
        // let vulkan_correction_transform = Scale3::new(1.0,-1.0,1.0).to_homogeneous();
        for node in gltf.nodes() {
            // let transformation_matrix = Matrix4::from(node.transform().matrix())
            //     * Matrix4::from_axis_angle(&Vector3::x_axis(), std::f32::consts::PI);
            // let transformation_matrix = Matrix4::from(node.transform().matrix());
            // let transformation_matrix = Matrix4::identity();
            let transformation_matrix = Matrix4::identity();
            // * vulkan_correction_transform;

            let _transformation_position = transformation_matrix.column(3).xyz();
            match node.mesh() {
                None => {}
                Some(mesh) => {
                    mesh_transform = Matrix4::from(node.transform().matrix());
                    let mut texture_type = 0; //I think primitve index should line up with texture type... maybe?
                    for primitive in mesh.primitives() {
                        println!("Texture type: {:}", texture_type);
                        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                        let indices = reader
                            .read_indices()
                            .unwrap()
                            .into_u32()
                            .collect::<Vec<_>>();
                        let positions = reader.read_positions().unwrap().collect::<Vec<_>>();
                        let normals = reader.read_normals().unwrap().collect::<Vec<_>>();
                        let texture_coordinates = reader
                            .read_tex_coords(0)
                            .unwrap()
                            .into_f32()
                            .collect::<Vec<_>>();

                        let weights = match reader.read_weights(0) {
                            None => None,
                            Some(weights) => Some(weights.into_f32().collect::<Vec<_>>()),
                        };
                        let joints = match reader.read_joints(0) {
                            None => None,
                            Some(joints) => Some(joints.into_u16().collect::<Vec<_>>()),
                        };

                        out_indices.extend_from_slice(&indices);
                        for i in 0..positions.len() {
                            let position = Vector3::from(positions[i]);
                            let position = transformation_matrix
                                .transform_point(&Point3::from(position))
                                .coords;
                            out_vertices.push(Vertex {
                                position,
                                normal: Vector3::zeros(),
                                tangent: Vector4::zeros(),
                                texture_coordinate: Vector2::zeros(),
                                texture_type,
                                bone_indices: Vector4::new(0, 0, 0, 0),
                                bone_weights: Vector4::new(0.0, 0.0, 0.0, 0.0),
                            });
                        }

                        for triangle in indices.chunks(3) {
                            for i in 0..3 {
                                let normal = normals[triangle[i] as usize];
                                let normal = -Vector3::from(normal);
                                let normal = transformation_matrix.transform_vector(&normal);

                                let texture_coordinate = texture_coordinates[triangle[i] as usize];
                                let texture_coordinate = Vector2::from(texture_coordinate);

                                let index1 = triangle[(i + 0) % 3] as usize;
                                let index2 = triangle[(i + 1) % 3] as usize;
                                let index3 = triangle[(i + 2) % 3] as usize;

                                let position3 = transformation_matrix
                                    .transform_point(&Point3::from(Vector3::from(
                                        positions[index3],
                                    )))
                                    .coords;
                                let position1 = transformation_matrix
                                    .transform_point(&Point3::from(Vector3::from(
                                        positions[index1],
                                    )))
                                    .coords;
                                let position2 = transformation_matrix
                                    .transform_point(&Point3::from(Vector3::from(
                                        positions[index2],
                                    )))
                                    .coords;

                                let edge1 = position2 - position1;
                                let edge2 = position3 - position1;
                                let delta_uv1 = Vector2::from(texture_coordinates[index2])
                                    - Vector2::from(texture_coordinates[index1]);
                                let delta_uv2 = Vector2::from(texture_coordinates[index3])
                                    - Vector2::from(texture_coordinates[index1]);

                                let f =
                                    1.0 / (delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y);

                                let mut tangent = Vector4::new(0.0, 0.0, 0.0, -1.0);

                                tangent.x = f * (delta_uv2.y * edge1.x - delta_uv1.y * edge2.x);
                                tangent.y = f * (delta_uv2.y * edge1.y - delta_uv1.y * edge2.y);
                                tangent.z = f * (delta_uv2.y * edge1.z - delta_uv1.y * edge2.z);

                                let normal_tangent = tangent.xyz().normalize();

                                tangent = Vector4::new(
                                    normal_tangent.x,
                                    normal_tangent.y,
                                    normal_tangent.z,
                                    tangent.w,
                                );
                                out_vertices[index1].normal = normal;
                                out_vertices[index1].texture_coordinate = texture_coordinate;
                                out_vertices[index1].tangent = tangent;

                                match (joints.as_ref(), weights.as_ref()) {
                                    (Some(joints), Some(weights)) => {
                                        out_vertices[index1].bone_indices =
                                            Vector4::from(joints[index1]).cast();
                                        out_vertices[index1].bone_weights =
                                            Vector4::from(weights[index1]);
                                    }
                                    (_, _) => {}
                                }
                            }
                        }
                        texture_type += 1;
                    }
                }
            }
        }
        match gltf.skins().nth(0) {
            None => {}
            Some(skin) => {
                let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
                let inverse_bind_matrices: Vec<_> = reader
                    .read_inverse_bind_matrices()
                    .unwrap()
                    .into_iter()
                    .map(|matrix| Matrix4::from(matrix))
                    .collect();

                for (_animation_index, animation) in gltf.animations().enumerate() {
                    let mut bone_sets = vec![];
                    let animation_end = {
                        animation
                            .channels()
                            .map(|channel| {
                                let reader =
                                    channel.reader(|buffer| Some(&buffers[buffer.index()]));
                                reader.read_inputs().unwrap().reduce(f32::max).unwrap()
                            })
                            .reduce(f32::max)
                            .unwrap()
                    };

                    let num_frames = {
                        animation
                            .channels()
                            .map(|channel| {
                                let reader =
                                    channel.reader(|buffer| Some(&buffers[buffer.index()]));
                                reader.read_inputs().unwrap().count()
                            })
                            .max()
                            .unwrap()
                    };

                    let mut animation_frames = vec![
                        AnimationKeyframes {
                            keyframes: vec![],
                            end_time: animation_end
                        };
                        gltf.nodes().len()
                    ];

                    animation.channels().for_each(|channel| {
                        let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));

                        let keyframes = match reader.read_outputs().unwrap() {
                            ReadOutputs::Translations(translations) => {
                                let out_translations = translations.map(|translation| Keyframe {
                                    frame_time: f32::NAN,
                                    translation: Some(Translation3::from(translation)),
                                    rotation: None,
                                    scale: None,
                                });
                                out_translations.collect::<Vec<_>>()
                            }
                            ReadOutputs::Rotations(rotations) => {
                                match rotations.into_f32().unwrap() {
                                    Rotations::F32(inner_rotations) => {
                                        let out_rotations =
                                            inner_rotations.map(|rotation| Keyframe {
                                                frame_time: f32::NAN,
                                                translation: None,
                                                rotation: Some(UnitQuaternion::from_quaternion(
                                                    Quaternion::from(rotation),
                                                )),
                                                scale: None,
                                            });
                                        out_rotations.collect::<Vec<_>>()
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            ReadOutputs::Scales(scales) => {
                                let out_scales = scales.map(|scale| Keyframe {
                                    frame_time: f32::NAN,
                                    translation: None,
                                    rotation: None,
                                    scale: Some(Scale3::from(scale)),
                                });
                                out_scales.collect::<Vec<_>>()
                            }
                            ReadOutputs::MorphTargetWeights(_) => unimplemented!(),
                        };

                        let node_index = channel.target().node().index();

                        reader
                            .read_inputs()
                            .unwrap()
                            .zip(keyframes.into_iter())
                            .for_each(|(frame_time, sampler_output)| {
                                animation_frames[node_index].add_sample(Keyframe {
                                    frame_time,
                                    translation: sampler_output.translation,
                                    rotation: sampler_output.rotation,
                                    scale: sampler_output.scale,
                                });
                            })
                    });

                    for keyframe_index in 0..num_frames {
                        let current_frame_time =
                            (keyframe_index as f32 / num_frames as f32) * animation_end;

                        let mut node_global_transforms = gltf
                            .nodes()
                            .map(|node| animation_frames[node.index()].sample(current_frame_time))
                            .collect::<Vec<_>>();

                        let nodes = gltf.nodes().collect::<Vec<_>>();
                        let mut current_indices: Vec<_> = vec![root_node.index()];

                        let mut next_indices = vec![];
                        loop {
                            for index in current_indices {
                                let node = &nodes[index];
                                let matrix = node_global_transforms[node.index()];
                                node.children().for_each(|child| {
                                    node_global_transforms[child.index()] =
                                        matrix * node_global_transforms[child.index()];
                                });

                                next_indices
                                    .extend(nodes[index].children().map(|node| node.index()));
                            }
                            if next_indices.len() > 0 {
                                current_indices = vec![];
                                current_indices.append(&mut next_indices);
                            } else {
                                break;
                            }
                        }

                        let mut bones = vec![];
                        for (joint_index, joint) in skin.joints().enumerate() {
                            let new_bone = Bone {
                                matrix: vulkan_correction_transform
                                    * (mesh_transform.try_inverse().unwrap()
                                        * node_global_transforms[joint.index()]
                                        * inverse_bind_matrices[joint_index]),
                            };
                            bones.push(new_bone);
                        }
                        bone_sets.push(bones);
                    }
                    out_animations.push(bone_sets)
                }
            }
        }

        return (out_vertices, out_indices, out_animations);
    }
    pub(crate) fn load_image_sequence(&self, folder: &Path) -> CombinedSampledImage {
        let mut images = vec![];
        for file in folder.read_dir().expect("Failed to read_dir") {
            let dynamic_image = image::io::Reader::open(file.unwrap().path())
                .unwrap()
                .decode()
                .unwrap();
            images.push(dynamic_image.into_rgba8());
        }
        assert_ne!(images.len(), 0);
        for image in &images {
            assert_eq!(
                image.width() * image.height(),
                images[0].width() * images[0].height()
            );
        }

        let width = images[0].width();
        let height = images[0].height();
        let depth = images.len() as u32;

        let image_info = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_3D)
            .flags(vk::ImageCreateFlags::empty())
            .extent(vk::Extent3D {
                width,
                height,
                depth,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_SRGB)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .samples(vk::SampleCountFlagBits::_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_info = vk_mem_erupt::AllocationCreateInfo {
            usage: vk_mem_erupt::MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let (image, allocation, _) = self
            .allocator
            .as_ref()
            .unwrap()
            .create_image(&image_info, &allocation_info)
            .unwrap();

        let image_view_create_info = vk::ImageViewCreateInfoBuilder::new()
            .image(image)
            .view_type(vk::ImageViewType::_3D)
            .format(image_info.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let image_view = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_image_view(&image_view_create_info, None)
        }
        .unwrap();

        let sampler_create_info = vk::SamplerCreateInfoBuilder::new()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(1.0)
            .border_color(vk::BorderColor::INT_OPAQUE_WHITE)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE);

        let sampler = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_sampler(&sampler_create_info, None)
        }
        .unwrap();

        let (transfer_buffer, transfer_allocation, transfer_allocation_info) = {
            let buffer_info = vk::BufferCreateInfoBuilder::new()
                .size((width * height * depth * 4) as u64)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC);
            let allocation_info = vk_mem_erupt::AllocationCreateInfo {
                usage: vk_mem_erupt::MemoryUsage::CpuToGpu,
                flags: vk_mem_erupt::AllocationCreateFlags::MAPPED,
                required_flags: vk::MemoryPropertyFlags::empty(),
                preferred_flags: vk::MemoryPropertyFlags::empty(),
                memory_type_bits: u32::MAX,
                pool: None,
                user_data: None,
            };

            self.allocator
                .as_ref()
                .unwrap()
                .create_buffer(&buffer_info, &allocation_info)
                .expect("Transfer buffer failed")
        };

        for (index, image) in images.into_iter().enumerate() {
            unsafe {
                (transfer_allocation_info
                    .get_mapped_data()
                    .offset((index * image.len()) as isize))
                .copy_from_nonoverlapping(image.as_ptr(), image.len());
            }
        }

        {
            let subresource_range = vk::ImageSubresourceRangeBuilder::new()
                .level_count(1)
                .layer_count(1)
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .base_array_layer(0);
            let barrier = vk::ImageMemoryBarrierBuilder::new()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(*subresource_range);

            let subresource = vk::ImageSubresourceLayersBuilder::new()
                .base_array_layer(0)
                .layer_count(1)
                .mip_level(0)
                .aspect_mask(vk::ImageAspectFlags::COLOR);
            let region = vk::BufferImageCopyBuilder::new()
                .buffer_image_height(0)
                .buffer_row_length(0)
                .image_subresource(subresource.build())
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth,
                });

            let command_buffer = self.begin_single_time_commands();
            unsafe {
                self.device.as_ref().unwrap().cmd_pipeline_barrier(
                    command_buffer,
                    Some(vk::PipelineStageFlags::ALL_COMMANDS),
                    Some(vk::PipelineStageFlags::ALL_COMMANDS),
                    None,
                    &[],
                    &[],
                    &[barrier],
                );
                self.device.as_ref().unwrap().cmd_copy_buffer_to_image(
                    command_buffer,
                    transfer_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                )
            }
            let barrier = vk::ImageMemoryBarrierBuilder::new()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(*subresource_range);
            unsafe {
                self.device.as_ref().unwrap().cmd_pipeline_barrier(
                    command_buffer,
                    Some(vk::PipelineStageFlags::ALL_COMMANDS),
                    Some(vk::PipelineStageFlags::ALL_COMMANDS),
                    None,
                    &[],
                    &[],
                    &[barrier],
                );
            }

            self.end_single_time_commands(command_buffer);
        }
        self.allocator
            .as_ref()
            .unwrap()
            .destroy_buffer(transfer_buffer, &transfer_allocation);

        return CombinedSampledImage {
            image,
            image_view,
            sampler,
            allocation,
            width,
            height,
        };
    }

}