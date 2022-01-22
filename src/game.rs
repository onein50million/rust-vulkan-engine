use crate::marching_cubes::{World, WORLD_SIZE_X, WORLD_SIZE_Y, WORLD_SIZE_Z};
use crate::renderer::*;
use nalgebra::{
    Isometry3, Matrix4, Point3, Rotation3, Translation3, UnitQuaternion, Vector2, Vector3,
};
use parry3d_f64::query::contact;
use parry3d_f64::shape::{Capsule, TriMesh};
use std::convert::TryInto;
use std::mem::size_of;
use std::path::Path;
use std::time::Instant;
use winit::window::Window;

mod directions {
    use nalgebra::Vector3;
    use std::f64::consts::FRAC_1_SQRT_2;

    pub(crate) const UP: Vector3<f64> = Vector3::new(0.0, -1.0, 0.0);
    pub(crate) const DOWN: Vector3<f64> = Vector3::new(0.0, 1.0, 0.0);
    pub(crate) const LEFT: Vector3<f64> = Vector3::new(-1.0, 0.0, 0.0);
    pub(crate) const RIGHT: Vector3<f64> = Vector3::new(1.0, 0.0, 0.0);
    pub(crate) const FORWARDS: Vector3<f64> = Vector3::new(0.0, 0.0, 1.0);
    pub(crate) const BACKWARDS: Vector3<f64> = Vector3::new(0.0, 0.0, -1.0);

    pub(crate) const ISOMETRIC_DOWN: Vector3<f64> =
        Vector3::new(-FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2);
    pub(crate) const ISOMETRIC_UP: Vector3<f64> = Vector3::new(FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2);
    pub(crate) const ISOMETRIC_RIGHT: Vector3<f64> =
        Vector3::new(-FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2);
    pub(crate) const ISOMETRIC_LEFT: Vector3<f64> =
        Vector3::new(FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2);
}

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
    pub(crate) exposure: f64,
    pub(crate) angle: f64,
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
            exposure: 1.0,
            angle: 0.0,
            panning: false,
            left_click: false,
        };
    }
}
struct Camera {
    position: Vector3<f64>,
    rotation: UnitQuaternion<f64>,
}
impl Camera {
    fn new() -> Self {
        Self {
            position: Vector3::zeros(),
            rotation: UnitQuaternion::face_towards(
                &Vector3::new(1.0, 1.0, 1.0),
                &Vector3::new(0.0, -1.0, 0.0),
            ),
        }
    }
    fn get_rotation(&self) -> UnitQuaternion<f64> {
        return self.rotation;
    }
    fn get_position(&self) -> Vector3<f64> {
        return self.position;
    }
}

struct Player {
    position: Vector3<f64>,
    velocity: Vector3<f64>,
    angle: f64,
    render_object_index: usize,
}
impl Player {
    const HEIGHT: Vector3<f64> = Vector3::new(0.0, 1.7, 0.0);
    fn get_rotation(&self) -> UnitQuaternion<f64> {
        return UnitQuaternion::face_towards(
            &Vector3::new(1.0, 0.0, 1.0),
            &Vector3::new(0.0, 1.0, 0.0),
        ) * UnitQuaternion::from_euler_angles(0.0, 0.0, self.angle);
    }
    fn get_position(&self) -> Vector3<f64> {
        return self.position;
    }

    fn process(
        &mut self,
        delta_time: f64,
        inputs: &mut Inputs,
        world: &World,
        world_isometry: &Isometry3<f64>,
    ) {
        let friction = Vector3::new(10.0, 0.1, 10.0);
        let acceleration = 100.0;
        if inputs.left_click {
            self.position.y = -2.0 * (WORLD_SIZE_Y as f64);
            self.velocity.y = 0.0;
            inputs.left_click = false;
        }

        self.velocity += (inputs.up * directions::ISOMETRIC_UP
            + inputs.down * directions::ISOMETRIC_DOWN
            + inputs.left * directions::ISOMETRIC_LEFT
            + inputs.right * directions::ISOMETRIC_RIGHT)
            .try_normalize(0.1)
            .unwrap_or(Vector3::zeros())
            * acceleration
            * delta_time;

        self.velocity -= self.velocity.component_mul(&friction) * delta_time.min(1.0);

        self.position += self.velocity * delta_time;

        // let ray =  Ray::new(
        //     Point3::from(self.position),
        //     Vector3::new(0.0,1.0,0.0));
        //
        // let ray_result = world.collision.as_ref().unwrap().cast_ray(world_isometry, &ray, 0.1,true);

        // let player_capsule = Capsule::new(
        //     Point3::from(self.position),
        //     Point3::from(self.position + Self::HEIGHT),
        //     0.5
        // );
        let player_capsule = Capsule::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, -Self::HEIGHT.y, 0.0),
            0.5,
        );

        let player_isometry =
            Isometry3::from_parts(Translation3::from(self.position), self.get_rotation());

        let contact_result = contact(
            world_isometry,
            world.collision.as_ref().unwrap(),
            &player_isometry,
            &player_capsule,
            10.0,
        )
        .unwrap();

        if contact_result.is_none() || contact_result.unwrap().dist > 0.1 {
            self.velocity.y += 9.8 * delta_time
        } else {
            self.velocity.y = 0.0;
            self.position.y = contact_result.unwrap().point1.y;
        }
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
    player: Player,
    world: World,
    world_index: usize,
}

impl Game {
    pub(crate) fn new(window: &Window) -> Self {
        let mut objects = vec![];

        let mut world = World::new_random();

        dbg!(size_of::<VulkanData>());

        let mut vulkan_data = VulkanData::new();
        println!("Vulkan Data created");
        vulkan_data.init_vulkan(&window);

        println!("Vulkan Data initialized");

        let world_index = objects.len();
        let mut world_object = GameObject::new(vulkan_data.objects.len());

        world_object.position =
            Vector3::new(WORLD_SIZE_X as f64 / -2.0, 0.0, WORLD_SIZE_Z as f64 / -2.0);

        objects.push(world_object);
        let mesh = world.generate_mesh();
        world.collision = Some(TriMesh::new(
            mesh.iter()
                .map(|vector| Point3::from(vector.position).cast())
                .collect(),
            (0..mesh.len())
                .step_by(3)
                .map(|index| {
                    (index..(index + 3))
                        .map(|index| index as u32)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect(),
        ));
        let mut voxel_render_object = RenderObject::new(
            &mut vulkan_data,
            mesh,
            vec![],
            TextureSet::new_empty(),
            false,
        );
        voxel_render_object.is_globe = true;

        vulkan_data.objects.push(voxel_render_object);

        vulkan_data.load_folder("models/planet/deep_water".parse().unwrap()); //Dummy objects to fill offsets
        vulkan_data.load_folder("models/planet/shallow_water".parse().unwrap()); //Dummy objects to fill offsets
        vulkan_data.load_folder("models/planet/foliage".parse().unwrap());
        vulkan_data.load_folder("models/planet/desert".parse().unwrap());
        vulkan_data.load_folder("models/planet/mountain".parse().unwrap());
        vulkan_data.load_folder("models/planet/snow".parse().unwrap());

        let mut player = Player {
            position: Vector3::new(0.0, -2.0 * (WORLD_SIZE_Y as f64), 0.0),
            velocity: Vector3::zeros(),
            angle: 0.0,
            render_object_index: vulkan_data.load_folder("models/person".parse().unwrap()),
        };
        // player.rotation = UnitQuaternion::face_towards(
        //     &Vector3::new(1.0, 0.0, 1.0),
        //     &Vector3::new(0.0, -1.0, 0.0),
        // );

        objects.push(GameObject::new(
            vulkan_data.load_folder("models/test_ball".parse().unwrap()),
        ));

        vulkan_data.update_vertex_and_index_buffers();

        let game = Game {
            game_start: std::time::Instant::now(),
            objects,
            mouse_position: Vector2::new(0.0, 0.0),
            last_mouse_position: Vector2::new(0.0, 0.0),
            inputs: Inputs::new(),
            focused: false,
            last_frame_instant: Instant::now(),
            vulkan_data,
            camera: Camera::new(),
            player,
            world,
            world_index,
        };
        return game;
    }

    pub(crate) fn process(&mut self) {
        let delta_time = self.last_frame_instant.elapsed().as_secs_f64();
        self.last_frame_instant = std::time::Instant::now();
        let delta_mouse = self.last_mouse_position - self.mouse_position;
        self.last_mouse_position = self.mouse_position;

        for i in 0..self.objects.len() {
            self.objects[i].process(delta_time);
        }

        // self.camera.rotation = UnitQuaternion::face_towards(&(self.camera.position - self.objects[0].position), &Vector3::new(0.0,-1.0,0.0));

        // if self.inputs.panning {
        //     self.objects[self.player_index].position.z += delta_mouse.x * delta_time * 5.0;
        //     self.objects[self.player_index].position.x -= delta_mouse.x * delta_time * 5.0;
        //
        //     self.objects[self.player_index].position.z -= delta_mouse.y * delta_time * 5.0;
        //     self.objects[self.player_index].position.x -= delta_mouse.y * delta_time * 5.0;
        // }

        // self.objects[self.player_index].position

        let world_isometry = Isometry3::from_parts(
            Translation3::from(self.objects[self.world_index].position),
            self.objects[self.world_index].rotation,
        );
        self.player
            .process(delta_time, &mut self.inputs, &self.world, &world_isometry);

        self.camera.position =
            self.player.get_position() + Vector3::new(1.0, 1.0, 1.0).normalize() * 50.0;

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

        let view_matrix = (Rotation3::from_euler_angles(0.0, self.inputs.angle, 0.0)
            .to_homogeneous()
            * (Matrix4::from(Translation3::from(self.camera.get_position()))
                * self.camera.get_rotation().to_homogeneous()))
        .try_inverse()
        .unwrap();

        for i in 0..self.objects.len() {
            let model_matrix = (Matrix4::from(Translation3::from(self.objects[i].position))
                * Matrix4::from(Rotation3::from(self.objects[i].rotation)))
            .cast();
            self.vulkan_data.objects[self.objects[i].render_object_index].model = model_matrix;
        }

        let model_matrix = (Matrix4::from(Translation3::from(self.player.get_position()))
            * Matrix4::from(Rotation3::from(self.player.get_rotation())))
        .cast();
        self.vulkan_data.objects[self.player.render_object_index].model = model_matrix;
        self.vulkan_data.objects[self.player.render_object_index].previous_frame = 0;
        self.vulkan_data.objects[self.player.render_object_index].next_frame = 1;
        self.vulkan_data.objects[self.player.render_object_index].animation_progress = self.game_start.elapsed().as_secs_f64() % 1.0;


        self.vulkan_data.uniform_buffer_object.view = view_matrix.cast();
        self.vulkan_data.uniform_buffer_object.proj = projection_matrix.cast();

        self.vulkan_data.uniform_buffer_object.time = self.game_start.elapsed().as_secs_f32();
        self.vulkan_data.uniform_buffer_object.player_position = self.player.position.cast();
        self.vulkan_data.uniform_buffer_object.exposure = self.inputs.exposure as f32;
        // self.vulkan_data.uniform_buffer_object.time = 0.5;
        self.vulkan_data.uniform_buffer_object.mouse_position = Vector2::new(
            (self.mouse_position.x) as f32,
            (self.mouse_position.y) as f32,
        );
    }
}
