use std::{fmt::Display, mem::MaybeUninit};

use float_ord::FloatOrd;
use nalgebra::{Vector3, Vector4, Isometry3, Point3, Orthographic3, Matrix4, Vector2};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use variant_count::VariantCount;

use crate::{support::Vertex, planet_gen::MeshOutput, province_gen::Islands};

#[repr(usize)]
#[derive(Clone, Copy, VariantCount, Debug, IntoPrimitive, TryFromPrimitive)]
pub enum Culture {
    CultureA,
    CultureB,
    CultureC,
}
#[repr(usize)]
#[derive(Clone, Copy, VariantCount, Debug, IntoPrimitive, TryFromPrimitive)]
pub enum Industry {
    Farm,
    Mill,
    Mine,
    Smelter,
}

#[repr(usize)]
#[derive(Clone, Copy, VariantCount, Debug, IntoPrimitive, TryFromPrimitive)]
pub enum Good {
    Grain,
    Food,
    Ore,
    Metal,
}

// #[repr(usize)]
// #[derive(Clone, Copy, VariantCount, Debug, IntoPrimitive, TryFromPrimitive)]
// pub enum Job{
//     Farmer,
//     Miller,
//     Miner,
//     Smelter,
// }

#[derive(Clone, Debug)]
pub struct Market {
    pub price: [f64; Good::VARIANT_COUNT],
    pub supply: [f64; Good::VARIANT_COUNT],
    pub demand: [f64; Good::VARIANT_COUNT],
    previous_supply_demand_error: [f64; Good::VARIANT_COUNT],
    supply_demand_error_integral: [f64; Good::VARIANT_COUNT],
}

#[derive(Clone, Debug)]
pub struct PopSlice {
    pub population: f64,
    pub money: f64,
    pub owned_goods: [f64; Good::VARIANT_COUNT],
    individual_demand: [f64; Good::VARIANT_COUNT],
    individual_supply: [f64; Good::VARIANT_COUNT],
}

const NUM_SLICES: usize = Culture::VARIANT_COUNT * Industry::VARIANT_COUNT;

const GOOD_STORAGE_RATIO: f64 = f64::INFINITY;

fn production_curve(
    population: f64,
    curviness: f64,
    base_slope: f64,
    slope_falloff: f64,
    current_supply: f64,
) -> f64 {
    let b = 1.0 / slope_falloff;
    let output = base_slope
        * (population - (curviness * (b * b * population * population + 1.0).sqrt()) / b)
        + curviness * slope_falloff * base_slope;
    let target_amount = population * GOOD_STORAGE_RATIO;
    if current_supply > target_amount {
        output * (target_amount / current_supply)
    } else {
        output
    }
}
// fn production_curve(population:f64, curviness: f64,base_slope: f64, slope_falloff: f64) -> f64{
//     population*base_slope
// }

fn get_inputs(industry: Industry, population: f64, delta_year: f64) -> [f64; Good::VARIANT_COUNT] {
    let mut output = [0.0; Good::VARIANT_COUNT];

    match industry {
        Industry::Farm => {}
        Industry::Mill => {
            output[Good::Grain as usize] = population * 1.0 * delta_year;
        }
        Industry::Mine => {}
        Industry::Smelter => {
            output[Good::Ore as usize] = population * 1.0 * delta_year;
        }
    }

    return output;
}

fn get_needs(population: f64, delta_year: f64) -> [f64; Good::VARIANT_COUNT] {
    let mut output = [0.0; Good::VARIANT_COUNT];

    output[Good::Food as usize] = population * 1.0 * delta_year;
    output[Good::Metal as usize] = population * 0.1 * delta_year;

    return output;
}

fn get_outputs(
    industry: Industry,
    population: f64,
    delta_year: f64,
    current_supply: f64,
) -> [f64; Good::VARIANT_COUNT] {
    let mut output = [0.0; Good::VARIANT_COUNT];

    match industry {
        Industry::Farm => {
            output[Good::Grain as usize] =
                production_curve(population, 0.5, 1.0, 5000.0, current_supply) * delta_year;
        }
        Industry::Mill => {
            output[Good::Food as usize] =
                production_curve(population, 1.0, 60.0, 100.0, current_supply) * delta_year;
        }
        Industry::Mine => {
            output[Good::Ore as usize] =
                production_curve(population, 1.0, 6.5, 1000.0, current_supply) * delta_year;
        }
        Industry::Smelter => {
            output[Good::Metal as usize] =
                production_curve(population, 1.0, 1000.0, 100.0, current_supply) * delta_year;
        }
    }

    return output;
}
#[derive(Debug)]
pub struct Pops {
    pub pop_slices: [PopSlice; NUM_SLICES],
}

impl Pops {
    pub fn new(total_population: f64) -> Self {
        const SLICE_COUNT: usize = NUM_SLICES;

        unsafe {
            let mut pop_slices: MaybeUninit<[PopSlice; SLICE_COUNT]> = MaybeUninit::uninit();
            for i in 0..SLICE_COUNT {
                (pop_slices.as_mut_ptr() as *mut PopSlice)
                    .offset(i as isize)
                    .write(PopSlice {
                        population: total_population / SLICE_COUNT as f64,
                        money: 10_000_000.0 / SLICE_COUNT as f64,
                        // owned_goods: [1_000_000_000_000.0 / SLICE_COUNT as f64; Good::VARIANT_COUNT],
                        owned_goods: [1000.0 / SLICE_COUNT as f64; Good::VARIANT_COUNT],
                        // met_inputs: [0.0; Good::VARIANT_COUNT],
                        individual_demand: [0.0; Good::VARIANT_COUNT],
                        individual_supply: [0.0; Good::VARIANT_COUNT],
                    })
            }
            let pop_slices = pop_slices.assume_init();
            Self { pop_slices }
        }
    }
    pub fn population(&self) -> f64 {
        self.pop_slices.iter().map(|slice| slice.population).sum()
    }
}

#[derive(Debug)]
pub struct Province {
    pub point_indices: Vec<usize>,
    pub pops: Pops,
    pub market: Market,
    pub position: Vector3<f64>,
}
impl Display for Province {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out_string = String::new();

        out_string.push_str("Market:\n");
        for i in 0..Good::VARIANT_COUNT {
            out_string.push_str(
                format!(
                    "Good: {:?}, Supply: {:.2}, Demand: {:.2}, Price: {:.2}\n",
                    Good::try_from(i).unwrap(),
                    self.market.supply[i],
                    self.market.demand[i],
                    self.market.price[i]
                )
                .as_str(),
            );
        }

        out_string.push_str("Pops:\n");
        for (i, slice) in self.pops.pop_slices.iter().enumerate() {
            let culture = Culture::try_from(i / Industry::VARIANT_COUNT).unwrap();
            let industry = Industry::try_from(i % Industry::VARIANT_COUNT).unwrap();
            out_string.push_str(
                format!(
                    "Culture: {:?}, Building: {:?}, Population: {:.2}, Money: {:.2}\n",
                    culture, industry, slice.population, slice.money
                )
                .as_str(),
            );

            out_string.push_str("Owned goods:\n");
            for i in 0..Good::VARIANT_COUNT {
                out_string.push_str(
                    format!(
                        "   Good: {:?}, Amount: {:.2}\n",
                        Good::try_from(i).unwrap(),
                        slice.owned_goods[i]
                    )
                    .as_str(),
                );
            }
        }

        write!(f, "{:}", out_string)
    }
}


// fn project_point_on_plane(point: Vector3<f32>, origin: Vector3<f32>, normal: Vector3<f32>) -> Point3<f32>{
//     let v = point - origin;
//     let dist = v.dot(&normal);

//     let isometry = Isometry3::look_at_rh(&Point3::from(origin), &Point3::from(origin + 100000.0*normal), &Vector3::new(0.0,-1.0, 0.0));
//     // dbg!(point);
//     isometry.transform_point(&Point3::from(point - dist * normal))
    
// }

// fn project_point_on_plane(point: Vector3<f32>, origin: Vector3<f32>, normal: Vector3<f32>) -> Vector2<f32>{
//     let v = point - origin;
//     let dist = v.dot(&normal);

//     let isometry = Isometry3::look_at_rh(&Point3::from(origin), &Point3::from(origin + 100000.0*normal), &Vector3::new(0.0,-1.0, 0.0));
//     // dbg!(point);
//     isometry.transform_point(&Point3::from(point - dist * normal)).coords.xy()
    
// }

fn project_point_on_plane(point: Vector3<f32>, origin: Vector3<f32>, normal: Vector3<f32>) -> Vector2<f32>{
    let v = point - origin;
    let dist = v.dot(&normal);

    let isometry = Isometry3::look_at_rh(&Point3::from(origin), &Point3::from(origin + 100000.0*normal), &Vector3::new(0.0,-1.0, 0.0));
    // dbg!(point);
    isometry.transform_point(&Point3::from(point - dist * normal)).coords.xy()
    
}

#[derive(Debug)]
pub struct World {
    pub points: Vec<Vector3<f32>>,
    pub provinces: Box<[Province]>,
}

impl World {
    pub const RADIUS: f64 = 6_378_137.0; //in meters
    
    pub fn intersect_planet(ray_origin: Vector3<f64>, ray_direction: Vector3<f64>) -> Option<Vector3<f64>>{
        let a = ray_direction.dot(&ray_direction);
        let b = 2.0 * ray_origin.dot(&ray_direction);
        let c = ray_origin.dot(&ray_origin) - (Self::RADIUS * Self::RADIUS);
        let discriminant = b*b - 4.0*a*c;
        if discriminant < 0.0{
            return None;
        }else{
            let ray_ratio = (-b + discriminant.sqrt()) / (2.0 * a);
            return Some(ray_origin + ray_direction * ray_ratio);
        }
    }
    
    pub fn new(vertices: &[Vector3<f32>],province_indices_slice: &[Vec<usize>]) -> Self {
        
        let mut provinces = vec![];
        for province_indices in province_indices_slice{
            let province_origin  = province_indices.iter().map(|&index| vertices[index]).sum::<Vector3<f32>>() / province_indices.len() as f32;
            let normal = province_origin.normalize();
            let province_origin = province_origin.normalize() * Self::RADIUS as f32;
            let mut sorted_indices = province_indices.clone();


            // after many attempts this seemed to work the best https://stackoverflow.com/a/65422749
            let origin_latlong = (
                normal.z.asin(),
                normal.y.atan2(normal.x)
            );
            sorted_indices.sort_by(|&a,&b|{
                let a_normal = vertices[a].normalize();
                let a_latlong = (
                    a_normal.z.asin(),
                    a_normal.y.atan2(a_normal.x)
                );
                let b_normal = vertices[b].normalize();
                let b_latlong = (
                    b_normal.z.asin(),
                    b_normal.y.atan2(b_normal.x)
                );

                let a1 = (a_latlong.0 - origin_latlong.0).atan2(a_latlong.1 - origin_latlong.1);
                let a2 = (b_latlong.0 - origin_latlong.0).atan2(b_latlong.1 - origin_latlong.1);

                FloatOrd(a1).cmp(&FloatOrd(a2))
            });

            // let right_vector = Vector2::new(1.0, 0.0);
            // sorted_indices.sort_by_key(|&index|{
            //     let projected_point = project_point_on_plane(vertices[index], province_origin, normal);
            //     FloatOrd((-projected_point).angle(&right_vector))
            // });

            // let reference_vector = normal.cross(&Vector3::new(0.0,-1.0,0.0));
            // sorted_indices.sort_by_key(|&index| {
            //     let vector = province_origin - vertices[index];
            //     FloatOrd(vector.angle(&reference_vector))
            // });

            // let lowermost = Vector3::new(-100000.0, 0.0, 0.0) - province_indices.iter().map(|&index|{
            //     project_point_on_plane(vertices[index], province_origin, normal)
            // }).max_by_key(|a|FloatOrd(a.y)).unwrap().coords;
            // sorted_indices.sort_by_key(|&index|{
            //     let point = vertices[index];
            //     let projected_point = project_point_on_plane(point, province_origin, normal);
            //     let vector = Vector3::new(-100000.0, 0.0, 0.0) - projected_point.coords;

            //     FloatOrd(vector.angle(&lowermost))
            //     // FloatOrd(normal.dot(&(lowermost.cross(&vector))))
            // });

            let mut out_indices = vec![];
            for index in sorted_indices.windows(2){
                out_indices.push(index[0]);
                out_indices.push(index[1]);
            }
            out_indices.push(*sorted_indices.last().unwrap());
            out_indices.push(*sorted_indices.first().unwrap());

            provinces.push(Province {
                point_indices: out_indices,
                position: province_origin.cast(),
                pops: Pops::new(10_000.0),
                market: Market {
                    price: [1.0; Good::VARIANT_COUNT],
                    supply: [0.0; Good::VARIANT_COUNT],
                    demand: [0.0; Good::VARIANT_COUNT],
                    previous_supply_demand_error: [0.0; Good::VARIANT_COUNT],
                    supply_demand_error_integral: [0.0; Good::VARIANT_COUNT],
                },
            })

        }

        Self {
            points:vertices.iter().map(|vertex| *vertex).collect(),
            provinces: provinces.into_boxed_slice()
        }
    }

    pub fn process(&mut self, delta_year: f64) {
        for (_province_index, province) in self.provinces.iter_mut().enumerate() {
            province.market.demand = [0.0; Good::VARIANT_COUNT];
            province.market.supply = [0.0; Good::VARIANT_COUNT];

            for (index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                // slice.met_inputs = [0.0; Good::VARIANT_COUNT];
                slice.individual_demand = [0.0; Good::VARIANT_COUNT];
                slice.individual_supply = [0.0; Good::VARIANT_COUNT];

                let _culture = Culture::try_from(index / Industry::VARIANT_COUNT).unwrap();
                let industry = Industry::try_from(index % Industry::VARIANT_COUNT).unwrap();

                let goods_needed: Vec<_> = get_inputs(industry, slice.population, delta_year)
                    .iter()
                    .zip(get_needs(slice.population, delta_year).iter())
                    .map(|(a, b)| a + b)
                    .collect();
                let num_distinct_goods_wanted = goods_needed.iter().filter(|a| **a > 0.0).count();

                let mut leftover_money = slice.money;
                // println!("goods_needed: {:?}", goods_needed);
                for i in 0..Good::VARIANT_COUNT {
                    let good = Good::try_from(i).unwrap();

                    let good_needed = goods_needed[good as usize];
                    let good_needed_to_be_bought =
                        (good_needed - slice.owned_goods[good as usize]).max(0.0);

                    let affordable_good = (slice.money / num_distinct_goods_wanted as f64)
                        / province.market.price[good as usize];
                    let good_demand = (good_needed_to_be_bought * 1.5).min(affordable_good);
                    slice.individual_demand[good as usize] = good_demand;
                    province.market.demand[good as usize] += good_demand;

                    let good_supply = (slice.owned_goods[good as usize] - good_needed).max(0.0);
                    slice.individual_supply[good as usize] = good_supply;
                    province.market.supply[good as usize] += good_supply;

                    leftover_money -= good_demand * province.market.price[good as usize];
                }
                if leftover_money > 0.0 {
                    for good_index in 0..Good::VARIANT_COUNT {
                        let amount_to_spend = (leftover_money / Good::VARIANT_COUNT as f64) * 0.5;
                        let amount_to_buy = amount_to_spend / province.market.price[good_index];
                        slice.individual_demand[good_index] += amount_to_buy;
                        province.market.demand[good_index] += amount_to_buy;
                    }
                }
            }
            for good_index in 0..Good::VARIANT_COUNT {
                let good = Good::try_from(good_index).unwrap();

                for (index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                    let _culture = Culture::try_from(index / Industry::VARIANT_COUNT).unwrap();
                    let _industry = Industry::try_from(index % Industry::VARIANT_COUNT).unwrap();

                    //Selling
                    let sell_ratio = if province.market.supply[good as usize] > 0.0 {
                        (province.market.demand[good as usize]
                            / province.market.supply[good as usize])
                            .min(1.0)
                    } else {
                        0.0
                    };
                    let amount_sold = slice.individual_supply[good as usize] * sell_ratio;
                    slice.owned_goods[good as usize] -= amount_sold;
                    slice.money += amount_sold * province.market.price[good as usize];

                    //Buying
                    let buy_ratio = if province.market.demand[good as usize] > 0.0 {
                        (province.market.supply[good as usize]
                            / province.market.demand[good as usize])
                            .min(1.0)
                    } else {
                        0.0
                    };
                    let amount_bought = slice.individual_demand[good as usize] * buy_ratio;
                    slice.owned_goods[good as usize] += amount_bought;
                    // slice.money -= (amount_bought * province.market.price[good as usize]).min(slice.money);
                    slice.money -= amount_bought * province.market.price[good as usize];
                    // slice.money = slice.money.max(0.0);

                    // println!("good: {:?}, amount_bought: {:}, amount_sold: {:}",good, amount_bought, amount_sold);
                }

                // if province.market.demand[good as usize] > province.market.supply[good as usize]{
                //     province.market.price[good as usize] += (province.market.price[good as usize] * 0.01 * delta_year).min(0.01);
                // }else{
                //     province.market.price[good as usize] -= (province.market.price[good as usize] * 0.01 * delta_year).min(0.01);
                // }

                //PID for price

                const PRICE_PID_P: f64 = 0.0000025;
                const PRICE_PID_I: f64 = 0.0000000;
                const PRICE_PID_D: f64 = 0.0000000;

                let supply_demand_delta =
                    province.market.supply[good as usize] - province.market.demand[good as usize];
                let error = 0.0 - supply_demand_delta;
                let proportional = error;
                province.market.supply_demand_error_integral[good as usize] += error * delta_year;
                let derivative = (error
                    - province.market.previous_supply_demand_error[good as usize])
                    / delta_year;
                let output = PRICE_PID_P * proportional
                    + PRICE_PID_I * province.market.supply_demand_error_integral[good as usize]
                    + PRICE_PID_D * derivative;
                province.market.previous_supply_demand_error[good as usize] = error;

                province.market.price[good as usize] += output.clamp(
                    -0.1 * province.market.price[good as usize],
                    0.1 * province.market.price[good as usize],
                );
                province.market.price[good as usize] =
                    province.market.price[good as usize].max(0.01);
            }

            let mut migration_pool = 0.0;
            //consume daily living needs
            for slice in province.pops.pop_slices.iter_mut() {
                let needs = get_needs(slice.population, delta_year);
                let minimum_met_need = needs
                    .iter()
                    .enumerate()
                    .map(|(good_index, need_amount)| {
                        // println!("good: {:?}, amount: {:}",Good::try_from(good_index).unwrap() , need_amount);
                        if need_amount <= &0.0 {
                            1.0
                        } else {
                            (slice.owned_goods[good_index] / need_amount).min(1.0)
                        }
                    })
                    .min_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap();

                for (good_index, &need) in needs.iter().enumerate() {
                    slice.owned_goods[good_index] -= need * minimum_met_need;
                }

                if minimum_met_need > 0.5 {
                    slice.population += slice.population * 0.01 * delta_year;
                } else {
                    slice.population -=
                        (slice.population * 0.01 * delta_year).min(slice.population);
                }
                if minimum_met_need < 0.9 {
                    let migration_amount =
                        (slice.population * 0.1 * delta_year).min(slice.population);
                    migration_pool += migration_amount;
                    slice.population -= migration_amount;
                }
            }
            for slice in province.pops.pop_slices.iter_mut() {
                let migration_amount = migration_pool / NUM_SLICES as f64;
                slice.population += migration_amount;
            }

            //use remaining resources for industry
            let mut _money_sum = 0.0;
            for (index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                _money_sum += slice.money;
                let industry = Industry::try_from(index % Industry::VARIANT_COUNT).unwrap();
                let inputs = get_inputs(industry, slice.population, delta_year);
                let minimum_met_input = inputs
                    .iter()
                    .enumerate()
                    .map(|(good_index, need_amount)| {
                        // println!("good: {:?}, amount: {:}",Good::try_from(good_index).unwrap() , need_amount);
                        if need_amount <= &0.0 {
                            1.0
                        } else {
                            (slice.owned_goods[good_index] / need_amount).min(1.0)
                        }
                    })
                    .min_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap();

                // let outputs = get_outputs(industry, slice.population,delta_year);

                // if outputs.iter().enumerate().find(|(good_index, output)| slice.owned_goods[*good_index] + **output * minimum_met_input > slice.population * GOOD_STORAGE_RATIO).is_none(){
                //     for (good_index,output) in outputs.iter().enumerate(){
                //         let good = Good::try_from(good_index).unwrap();
                //         slice.owned_goods[good as usize] -= inputs[good as usize] * minimum_met_input;
                //         slice.owned_goods[good as usize] += output * minimum_met_input;
                //     }
                // }
                let outputs = get_outputs(
                    industry,
                    slice.population,
                    delta_year,
                    slice.owned_goods.into_iter().reduce(f64::max).unwrap(),
                );
                for (good_index, output) in outputs.iter().enumerate() {
                    let good = Good::try_from(good_index).unwrap();
                    slice.owned_goods[good as usize] -= inputs[good as usize] * minimum_met_input;
                    slice.owned_goods[good as usize] += output * minimum_met_input;
                }
            }
            let mut tax_bank = 0.0;
            let mut _population_sum = 0.0;
            for (_index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                let tax_amount = (slice.money * 0.9 * delta_year).min(slice.money);
                tax_bank += tax_amount;
                slice.money -= tax_amount;
                _population_sum += slice.population;
            }
            for (_index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                let tax_payment = tax_bank * (1.0 / NUM_SLICES as f64);
                slice.money += tax_payment;
            }

            // dbg!(money_sum);
        }
    }
}
