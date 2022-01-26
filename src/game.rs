use std::borrow::BorrowMut;
use crate::marching_cubes::{World, WORLD_SIZE_X, WORLD_SIZE_Y, WORLD_SIZE_Z};
use crate::renderer::*;
use nalgebra::{
    Isometry3, Matrix4, Point3, Rotation3, Translation3, Unit, UnitQuaternion, Vector2, Vector3,
};
use parry3d_f64::query::contact;
use parry3d_f64::shape::{Capsule, TriMesh};
use std::convert::TryInto;
use std::mem::size_of;
use std::option::Option::None;
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

enum ObjectType {
    None,
    Grip(bool),
    Player(Player),
}

pub(crate) struct GameObject {
    position: Vector3<f64>,
    rotation: UnitQuaternion<f64>,
    render_object_index: usize,
    animation_handler: Option<AnimationHandler>,
    object_type: ObjectType
}
impl GameObject {
    fn new(render_object_index: usize, animation_handler: Option<AnimationHandler>, object_type: ObjectType) -> Self {
        return Self {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.0),
            render_object_index,
            animation_handler,
            object_type
        };
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

struct AnimationHandler {
    frame_start: usize,
    previous_frame: usize,
    next_frame: usize,
    frame_count: usize,
    frame_rate: f64,
    progress: f64,
}
impl AnimationHandler {
    fn new(frame_count: usize, frame_start: usize) -> Self {
        Self {
            frame_start,
            previous_frame: 0,
            next_frame: 1,
            frame_count,
            frame_rate: 60.0,
            progress: 0.0,
        }
    }
    fn process(&mut self, delta_time: f64) {
        self.progress += delta_time * self.frame_rate;
        if self.progress > 1.0 {
            // dbg!(self.progress);
            self.progress = 0.0;
            self.previous_frame = (self.previous_frame + 1) % self.frame_count;
            self.next_frame = (self.next_frame + 1) % self.frame_count;
        }
    }
}

struct Player {
    velocity: Vector3<f64>,
    last_rotation: UnitQuaternion<f64>,
    angle: f64,
}
impl Player {
    const HEIGHT: Vector3<f64> = Vector3::new(0.0, 1.7, 0.0);
    fn process(
        &mut self,
        delta_time: f64,
        inputs: &mut Inputs,
        world: &World,
        world_isometry: &Isometry3<f64>,
    ) {
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
    player_index: usize,
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
        let mut world_object = GameObject::new(vulkan_data.objects.len(), None, ObjectType::None);

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

        vulkan_data
            .load_folder("models/planet/deep_water".parse().unwrap())
            .0; //Dummy objects to fill offsets
        vulkan_data
            .load_folder("models/planet/shallow_water".parse().unwrap())
            .0; //Dummy objects to fill offsets
        vulkan_data
            .load_folder("models/planet/foliage".parse().unwrap())
            .0;
        vulkan_data
            .load_folder("models/planet/desert".parse().unwrap())
            .0;
        vulkan_data
            .load_folder("models/planet/mountain".parse().unwrap())
            .0;
        vulkan_data
            .load_folder("models/planet/snow".parse().unwrap())
            .0;

        let player_frame_start = vulkan_data.current_boneset;
        let (render_object_index, frame_count) =
            vulkan_data.load_folder("models/person".parse().unwrap());
        let player = Player {
            velocity: Vector3::zeros(),
            last_rotation: UnitQuaternion::identity(),
            angle: 0.0,
        };
        let player_index = objects.len();
        objects.push(GameObject::new(
            render_object_index,
            Some(AnimationHandler::new(frame_count, player_frame_start)),
            ObjectType::Player(player)
        ));
        objects[player_index].position = Vector3::new(0.0, -2.0 * (WORLD_SIZE_Y as f64), 0.0);
        // player.rotation = UnitQuaternion::face_towards(
        //     &Vector3::new(1.0, 0.0, 1.0),
        //     &Vector3::new(0.0, -1.0, 0.0),
        // );

        objects.push(GameObject::new(vulkan_data
                                         .load_folder("models/test_ball".parse().unwrap())
                                         .0, None, ObjectType::None));

        let frame_start = vulkan_data.current_boneset;
        let (render_index, frame_count) = vulkan_data.load_folder("models/cube".parse().unwrap());
        objects.push(GameObject::new(render_index, Some(AnimationHandler::new(frame_count, frame_start)), ObjectType::None));

        let frame_start = vulkan_data.current_boneset;
        let (render_index, frame_count) = vulkan_data.load_folder("models/shotgun".parse().unwrap());
        objects.push(GameObject::new(
            render_index,
            Some(AnimationHandler{
                frame_start,
                previous_frame: 0,
                next_frame: 0,
                frame_count,
                frame_rate: 0.0,
                progress: 0.0
            }),
            ObjectType::Grip(true)
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
            player_index,
            world,
            world_index,
        };
        return game;
    }

    pub(crate) fn process(&mut self) -> Result<(), DanielError> {
        let delta_time = self.last_frame_instant.elapsed().as_secs_f64();
        self.last_frame_instant = std::time::Instant::now();
        let delta_mouse = self.last_mouse_position - self.mouse_position;
        self.last_mouse_position = self.mouse_position;

        let world_isometry = Isometry3::from_parts(
            Translation3::from(self.objects[self.world_index].position),
            self.objects[self.world_index].rotation,
        );


        for i in 0..self.objects.len() {
            let mut position = self.objects[i].position;
            let mut rotation = self.objects[i].rotation;
            let mut animation_change = delta_time;
            match self.objects[i].object_type.borrow_mut(){
                ObjectType::None => {}
                ObjectType::Grip(is_gripped) => {}
                ObjectType::Player(player) =>{
                    let friction = Vector3::new(10.0, 0.1, 10.0);
                    let acceleration = 100.0;
                    if self.inputs.left_click {
                        position.y = -2.0 * (WORLD_SIZE_Y as f64);
                        player.velocity.y = 0.0;
                        self.inputs.left_click = false;
                    }

                    player.velocity += (self.inputs.up * directions::ISOMETRIC_UP
                        + self.inputs.down * directions::ISOMETRIC_DOWN
                        + self.inputs.left * directions::ISOMETRIC_LEFT
                        + self.inputs.right * directions::ISOMETRIC_RIGHT)
                        .try_normalize(0.1)
                        .unwrap_or(Vector3::zeros())
                        * acceleration
                        * delta_time;

                    rotation = if player.velocity.magnitude() > 0.1 {
                        UnitQuaternion::face_towards(&player.velocity, &Vector3::new(0.0, 1.0, 0.0))
                    } else {
                        player.last_rotation
                    };

                    player.velocity -= player.velocity.component_mul(&friction) * delta_time.min(1.0);

                    position += player.velocity * delta_time;

                    let player_capsule = Capsule::new(
                        Point3::new(0.0, 0.0, 0.0),
                        Point3::new(0.0, -Player::HEIGHT.y, 0.0),
                        0.5,
                    );

                    let player_isometry =
                        Isometry3::from_parts(Translation3::from(position), rotation);

                    let contact_result = contact(
                        &world_isometry,
                        self.world.collision.as_ref().unwrap(),
                        &player_isometry,
                        &player_capsule,
                        10.0,
                    )
                        .unwrap();

                    if contact_result.is_none() || contact_result.unwrap().dist > 0.1 {
                        player.velocity.y += 9.8 * delta_time
                    } else {
                        player.velocity.y = 0.0;
                        position.y = contact_result.unwrap().point1.y;
                    }
                    animation_change = if player.velocity.magnitude() > 0.1{
                        player.last_rotation = rotation;
                        delta_time * player.velocity.magnitude().clamp(0.0,1.0)
                    }else{
                        0.0
                    }

                }
            }
            self.objects[i].position = position;
            self.objects[i].rotation = rotation;
            match self.objects[i].animation_handler.as_mut() {
                None => {}
                Some(animation_handler) => animation_handler.process(animation_change),
            }

        }


        self.camera.position =
            self.objects[self.player_index].position - Player::HEIGHT*0.5 + Vector3::new(1.0, 1.0, 1.0).normalize() * 50.0;

        self.update_renderer();
        self.vulkan_data.transfer_data_to_gpu();
        self.vulkan_data.draw_frame()
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
            let render_object = &mut self.vulkan_data.objects[self.objects[i].render_object_index];
            let model_matrix = (Matrix4::from(Translation3::from(self.objects[i].position))
                * Matrix4::from(Rotation3::from(self.objects[i].rotation)))
            .cast();
            match self.objects[i].object_type{
                ObjectType::Grip(is_gripped) => {
                    const HAND_POSE_INDEX:usize = 17;
                    if is_gripped{
                        let player = &self.objects[self.player_index];
                        let previous =  self.vulkan_data.storage_buffer_object.bone_sets[player.animation_handler.as_ref().unwrap().frame_start+player.animation_handler.as_ref().unwrap().previous_frame].bones[HAND_POSE_INDEX].matrix;
                        let next =  self.vulkan_data.storage_buffer_object.bone_sets[player.animation_handler.as_ref().unwrap().frame_start+player.animation_handler.as_ref().unwrap().next_frame].bones[HAND_POSE_INDEX].matrix;
                        let progress = self.objects[self.player_index].animation_handler.as_ref().unwrap().progress;
                        let matrix = previous.cast() * (1.0 - progress) + next.cast() * progress;
                        let matrix = (Translation3::from(player.position).to_homogeneous() * player.rotation.to_homogeneous()) * matrix;
                        render_object.model = matrix.cast();
                    }
                }
                _=>{
                    render_object.model = model_matrix;
                }
            }


            match &self.objects[i].animation_handler {
                None => {}
                Some(animation_handler) => {
                    render_object.previous_frame = animation_handler.frame_start as u8
                        + animation_handler.previous_frame as u8;
                    render_object.next_frame =
                        animation_handler.frame_start as u8 + animation_handler.next_frame as u8;
                    render_object.animation_progress = animation_handler.progress;
                }
            }
        }

        self.vulkan_data.uniform_buffer_object.view = view_matrix.cast();
        self.vulkan_data.uniform_buffer_object.proj = projection_matrix.cast();

        self.vulkan_data.uniform_buffer_object.time = self.game_start.elapsed().as_secs_f32();
        self.vulkan_data.uniform_buffer_object.player_position = self.objects[self.player_index].position.cast();
        self.vulkan_data.uniform_buffer_object.exposure = self.inputs.exposure as f32;
        // self.vulkan_data.uniform_buffer_object.time = 0.5;
        self.vulkan_data.uniform_buffer_object.mouse_position = Vector2::new(
            (self.mouse_position.x) as f32,
            (self.mouse_position.y) as f32,
        );
    }
}
