/*
Some random ideas:

player has states, ie crouching, aiming, maybe a bitfield-like object?

compile time gltf parsing


Game object types:
Server object: calculated purely on server, client just copies data from server
Client predicted: Uses client side prediction to reduce latency
Client only (Animations, particles)
 */
const GAME_SPEEDS: [f64; 5] = [0.0, 1.0, 24.0, 24.0 * 7.0, 24.0 * 30.0];
// const GAME_SPEEDS: [f64; 5] = [0.0, 182.5*24.0, 24.0, 24.0 * 7.0, 24.0 * 30.0];

pub mod directions {
    use nalgebra::Vector3;
    use std::f64::consts::FRAC_1_SQRT_2;

    pub const UP: Vector3<f64> = Vector3::new(0.0, -1.0, 0.0);
    pub const DOWN: Vector3<f64> = Vector3::new(0.0, 1.0, 0.0);
    pub const LEFT: Vector3<f64> = Vector3::new(-1.0, 0.0, 0.0);
    pub const RIGHT: Vector3<f64> = Vector3::new(1.0, 0.0, 0.0);
    pub const FORWARDS: Vector3<f64> = Vector3::new(0.0, 0.0, 1.0);
    pub const BACKWARDS: Vector3<f64> = Vector3::new(0.0, 0.0, -1.0);

    pub const ISOMETRIC_DOWN: Vector3<f64> = Vector3::new(-FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2);
    pub const ISOMETRIC_UP: Vector3<f64> = Vector3::new(FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2);
    pub const ISOMETRIC_RIGHT: Vector3<f64> = Vector3::new(-FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2);
    pub const ISOMETRIC_LEFT: Vector3<f64> = Vector3::new(FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2);
}

pub(crate) mod server {
    use nalgebra::{UnitQuaternion, Vector3};
    use std::time::Instant;
    pub struct GameObject {
        pub position: Vector3<f64>,
        pub rotation: UnitQuaternion<f64>,
    }
    impl GameObject {
        pub fn new() -> Self {
            return Self {
                position: Vector3::new(0.0, 0.0, 0.0),
                rotation: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.0),
            };
        }
    }
    struct Game {
        game_start: Instant,
        last_frame_instant: Instant,
        planet: GameObject,
    }

    impl Game {
        pub fn new() -> Self {
            let game = Self {
                game_start: Instant::now(),
                last_frame_instant: Instant::now(),
                planet: GameObject::new(),
            };
            game
        }

        pub fn process(&mut self) {
            let _delta_time = self.last_frame_instant.elapsed().as_secs_f64();
            self.last_frame_instant = std::time::Instant::now();
        }
    }
}
pub mod client {
    use std::f64::consts::PI;
    use std::time::Instant;

    use crate::support::Inputs;
    use crate::world::agent::{Agent, AgentAction};
    use crate::world::organization::{DecisionCategory, DiplomaticAction};
    use crate::world::{ProvinceKey, World};
    use crate::{renderer::VulkanData, world::organization::OrganizationKey};
    use float_ord::FloatOrd;
    use nalgebra::{Matrix4, Perspective3, Point3, Translation3, UnitQuaternion, Vector2, Vector3};
    use rand::Rng;
    use rand::seq::SliceRandom;

    use super::GAME_SPEEDS;

    pub struct AnimationHandler {
        pub index: usize,
        pub previous_frame: usize,
        pub next_frame: usize,
        frame_count: usize,
        frame_rate: f64,
        pub progress: f64,
    }
    impl AnimationHandler {
        fn new(index: usize, frame_count: usize) -> Self {
            Self {
                index,
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
                self.progress = 0.0;
                self.previous_frame = (self.previous_frame + 1) % self.frame_count;
                self.next_frame = (self.next_frame + 1) % self.frame_count;
            }
        }
        fn switch_animation(
            &mut self,
            vulkan_data: &VulkanData,
            render_object_index: usize,
            animation_index: usize,
        ) {
            self.previous_frame = 0;
            self.next_frame = 1;
            self.frame_count =
                vulkan_data.objects[render_object_index].get_animation_length(animation_index);
            self.progress = 0.0
        }
    }

    pub struct Camera {
        latitude: f64,
        longitude: f64,
    }
    impl Camera {
        fn new() -> Self {
            Self {
                latitude: 0.0,
                longitude: 0.0,
            }
        }
        fn get_rotation(&self) -> UnitQuaternion<f64> {
            return UnitQuaternion::face_towards(
                &(self.get_position()),
                &Vector3::new(0.0, -1.0, 0.0),
            );
        }
        fn get_position(&self) -> Vector3<f64> {
            return UnitQuaternion::from_euler_angles(0.0, self.longitude, 0.0)
                * UnitQuaternion::from_euler_angles(0.0, 0.0, self.latitude)
                * Vector3::new(10_000_000.0, 0.0, 0.0);
        }
        pub fn get_view_matrix(&self, planet_transform: Matrix4<f64>) -> Matrix4<f64> {
            (planet_transform
                * (Matrix4::from(Translation3::from(self.get_position()))
                    * self.get_rotation().to_homogeneous()))
            .try_inverse()
            .unwrap()
        }
    }

    pub struct GameObject {
        pub position: Vector3<f64>,
        pub rotation: UnitQuaternion<f64>,
        pub render_object_index: usize,
        pub animation_handler: Option<AnimationHandler>,
    }
    impl GameObject {
        pub fn get_transform(&self) -> Matrix4<f64> {
            return Matrix4::from(Translation3::from(self.position))
                * self.rotation.to_homogeneous();
        }
    }
    pub struct Game {
        pub world: World,
        pub player_agent: Agent,
        pub ai_agents: Vec<Agent>,
        pub inputs: Inputs,
        pub mouse_position: Vector2<f64>,
        pub last_mouse_position: Vector2<f64>,
        pub planet: GameObject,
        pub start_time: Instant,
        pub camera: Camera,
    }
    impl Game {
        pub fn new(planet_render_index: usize, world: World) -> Self {
            let ai_agents = world
                .organizations
                .iter()
                .map(|(o, _)| Agent::new(Some(o)))
                .collect();
            Self {
                inputs: Inputs::new(),
                mouse_position: Vector2::zeros(),
                planet: GameObject {
                    position: Vector3::zeros(),
                    rotation: UnitQuaternion::identity(),
                    render_object_index: planet_render_index,
                    animation_handler: None,
                },
                start_time: Instant::now(),
                camera: Camera::new(),
                last_mouse_position: Vector2::zeros(),
                world,
                player_agent: Agent::new(None),
                ai_agents,
            }
        }
        pub fn process(
            &mut self,
            delta_time: f64,
            projection: &Perspective3<f64>,
            game_speed: u16,
        ) {
            let speed = GAME_SPEEDS[(game_speed as usize) % GAME_SPEEDS.len()];
            if speed > 0.0 {
                let delta_year = World::STEP_LENGTH * speed;
                self.world.process(delta_year);
                const POLITICAL_POWER_YEARLY_GAIN: f64 = 100.0;
                self.player_agent.political_power += POLITICAL_POWER_YEARLY_GAIN * delta_year;
                for agent in &mut self.ai_agents {
                    agent.political_power += POLITICAL_POWER_YEARLY_GAIN * delta_year;
                }
            }
            // let total_money: f64 = self.world.provinces.iter().map(|a|a.pops.pop_slices.iter().map(|b|b.money).sum::<f64>()).sum();
            // dbg!(total_money);

            // yearly
            let ai_chance: f64 = 0.1;
            let mut rng = rand::thread_rng();

            for ai_agent in &mut self.ai_agents {
                if let Some(ai_org) = ai_agent.controlling_organization {
                    
                    if rng.gen::<f64>() > (1.0 - ai_chance).powf(delta_time) {        

                        if self.world.organizations[ai_org].diplomatic_offers.len() > 0{
                            let action = AgentAction::RespondToDiplomaticOffer(
                                0,
                                rng.gen(),
                            );
                            if action.get_cost(ai_org, &self.world) < ai_agent.political_power{
                                if let Err(err) =
                                    ai_agent.attempt_action(&mut self.world, action)
                                {
                                    println!("AI Error: {err}");
                                }
                            }
                        }

                        let enemy_orgs:Box<[_]> = (0..self.world.organizations.0.len()).map(|o|OrganizationKey(o)).filter(|enemy_org|self.world.relations.get_relations(ai_org, *enemy_org).at_war).collect();
                        let possible_targets:Box<[_]> = (0..self.world.provinces.0.len()).map(|p| ProvinceKey(p))
                            .filter(|&p| enemy_orgs.iter().any(|&o|self.world.organizations[o].province_control[p] > 0.5))
                            .filter(|p|{
                                self.world.provinces[*p].neighbouring_provinces.iter().any(|&p| self.world.organizations[ai_org].province_control[p] > 0.5)
                        }).collect();

                        ai_agent.targeted_province = possible_targets.choose(&mut rng).copied();

                        for province_key in
                            (0..self.world.provinces.0.len()).map(|p| ProvinceKey(p))
                        {
                            let mut new_weight = 0.0;

                            if self.world.organizations[ai_org].province_control[province_key] > 0.5
                            {
                                new_weight += 1.0;
                                for neighbour in self.world.provinces[province_key]
                                    .neighbouring_provinces
                                    .iter()
                                {
                                    if self.world.organizations[ai_org].province_control[*neighbour]
                                        < 0.5
                                    {
                                        new_weight += 1.0;
                                    }
                                }
                            }
                            if self.world.organizations[ai_org].province_control[province_key] > 0.1 && self.world.organizations[ai_org].province_control[province_key] < 0.99{
                                new_weight += 5.0;
                            }
                            if ai_agent.targeted_province == Some(province_key){
                                new_weight += 10.0;
                            }
                            // for &enemy_org in enemy_orgs.iter(){
                            //     if self.world.organizations[enemy_org].province_control[province_key] > 0.5{
                            //         for &neighbour in self.world.provinces[province_key]
                            //             .neighbouring_provinces
                            //             .iter(){
                            //                 if self.world.organizations[ai_org].province_control[neighbour] > 0.5{
                            //                     new_weight += 2.0;
                            //                 }
                            //             }
                            //     }
                            // }

                            if let Err(err) = ai_agent.attempt_action(
                                &mut self.world,
                                AgentAction::SetTroopWeight {
                                    province: province_key,
                                    weight: new_weight,
                                },
                            ) {
                                println!("AI Error: {err}");
                            }
                        }
                        for other_org in
                            (0..self.world.organizations.0.len()).map(|o| OrganizationKey(o))
                        {
                            if other_org == ai_org {
                                continue;
                            }
                            if self.world.relations.get_relations(ai_org, other_org).at_war {
                                let action = AgentAction::DiplomaticAction(
                                    other_org,
                                    DiplomaticAction::OfferPeace,
                                );
                                if action.get_cost(ai_org, &self.world) < ai_agent.political_power && rng.gen::<f64>() > 0.95{
                                    if let Err(err) =
                                        ai_agent.attempt_action(&mut self.world, action)
                                    {
                                        println!("AI Error: {err}");
                                    }
                                }
                            }else{
                                let action = AgentAction::DiplomaticAction(
                                    other_org,
                                    DiplomaticAction::DeclareWar,
                                );
                                if action.get_cost(ai_org, &self.world) < ai_agent.political_power && rng.gen::<f64>() > 0.99{
                                    if let Err(err) =
                                        ai_agent.attempt_action(&mut self.world, action)
                                    {
                                        println!("AI Error: {err}");
                                    }
                                }

                            }
                        }
                    }
                }
            }

            let delta_mouse = self.last_mouse_position - self.mouse_position;
            self.last_mouse_position = self.mouse_position;
            if self.inputs.panning {
                self.camera.latitude =
                    (self.camera.latitude + 1.0 * delta_mouse.y).clamp(-PI / 2.01, PI / 2.01);
                self.camera.longitude += delta_mouse.x * 1.0;
            }

            self.planet.rotation =
                UnitQuaternion::from_euler_angles(23.43644f64.to_radians(), 0.0, 0.0)
                    * UnitQuaternion::from_euler_angles(
                        0.0,
                        -std::f64::consts::PI * 2.0 * 365.0 * self.world.current_year,
                        0.0,
                    );

            if self.inputs.left_click {
                self.world.selected_province = self.get_closest_province(projection);
                self.inputs.left_click = false;
            }
            if self.inputs.right_click {
                self.world.targeted_province = self.get_closest_province(projection);
                self.inputs.right_click = false;
            }
        }

        fn get_closest_province(&mut self, projection: &Perspective3<f64>) -> Option<ProvinceKey> {
            //TODO: Figure out why I have to make these negative. Probably something to do with the inconsistent coordinate system
            let view_matrix_inverse = self
                .camera
                .get_view_matrix(self.planet.get_transform())
                .try_inverse()
                .unwrap();
            let direction = view_matrix_inverse
                .transform_vector(
                    &projection
                        .unproject_point(&Point3::new(
                            -self.mouse_position.x,
                            -self.mouse_position.y,
                            1.0,
                        ))
                        .coords,
                )
                .normalize();
            let origin = view_matrix_inverse
                .transform_point(&Point3::from(Vector3::zeros()))
                .coords;
            match World::intersect_planet(origin, -direction.xyz()) {
                Some(point) => Some(ProvinceKey(
                    self.world
                        .provinces
                        .0
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, province)| {
                            FloatOrd(
                                (point
                                    - self
                                        .planet
                                        .get_transform()
                                        .transform_point(&Point3::from(province.position))
                                        .coords)
                                    .magnitude(),
                            )
                        })
                        .expect("Failed to find closest provice to click")
                        .0,
                )),
                None => None,
            }
        }
    }
}
