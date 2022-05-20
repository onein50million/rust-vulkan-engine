use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Display,
    mem::MaybeUninit,
};

use float_ord::FloatOrd;
use nalgebra::Vector3;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use variant_count::VariantCount;

use crate::{
    renderer::ElevationVertex,
    support::{map_range_linear, map_range_linear_f64},
};

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
    Labor, //Building things
}

#[repr(usize)]
#[derive(Clone, Copy, VariantCount, Debug, IntoPrimitive, TryFromPrimitive)]
pub enum Good {
    Grain,
    Food,
    Ore,
    Metal,
}

#[repr(usize)]
#[derive(Clone, Copy, VariantCount, Debug, IntoPrimitive, TryFromPrimitive)]
pub enum IntangibleGoods {
    //Aren't owned by anyone, reside in the province level, can't be moved or traded, only consumed
    Work, // in man-years
}

#[derive(Clone, Copy, Debug)]
pub struct IndustryData {
    pub productivity: f64,
    pub size: f64,
}

#[derive(Clone, Debug)]
pub struct Market {
    pub price: [f64; Good::VARIANT_COUNT],
    pub supply: [f64; Good::VARIANT_COUNT],
    pub demand: [f64; Good::VARIANT_COUNT],
    previous_supply_demand_error: [f64; Good::VARIANT_COUNT],
    supply_demand_error_integral: [f64; Good::VARIANT_COUNT],
}

impl Market {
    fn process(
        &mut self,
        delta_year: f64,
        pop_slices: &mut [PopSlice],
        spend_leftovers_ratio: f64,
    ) {
        for (index, slice) in pop_slices.iter_mut().enumerate() {
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

                let affordable_good =
                    (slice.money / num_distinct_goods_wanted as f64) / self.price[good as usize];
                let good_demand = (good_needed_to_be_bought * 1.5).min(affordable_good);
                slice.individual_demand[good as usize] = good_demand;
                self.demand[good as usize] += good_demand;

                let good_supply = (slice.owned_goods[good as usize] - good_needed).max(0.0);
                slice.individual_supply[good as usize] = good_supply;
                self.supply[good as usize] += good_supply;

                leftover_money -= good_demand * self.price[good as usize];
            }
            leftover_money *= spend_leftovers_ratio;
            if leftover_money > 0.0 {
                for good_index in 0..Good::VARIANT_COUNT {
                    let amount_to_spend = (leftover_money / Good::VARIANT_COUNT as f64) * 0.5;
                    let amount_to_buy = amount_to_spend / self.price[good_index];
                    slice.individual_demand[good_index] += amount_to_buy;
                    self.demand[good_index] += amount_to_buy;
                }
            }
        }
    }
    fn buy_and_sell(&mut self, good: Good, delta_year: f64, pop_slices: &mut [PopSlice]) {
        for (index, slice) in pop_slices.iter_mut().enumerate() {
            let _culture = Culture::try_from(index / Industry::VARIANT_COUNT).unwrap();
            let _industry = Industry::try_from(index % Industry::VARIANT_COUNT).unwrap();

            //Selling
            let sell_ratio = if self.supply[good as usize] > 0.0 {
                (self.demand[good as usize] / self.supply[good as usize]).min(1.0)
            } else {
                0.0
            };
            let amount_sold = slice.individual_supply[good as usize] * sell_ratio;
            slice.owned_goods[good as usize] -= amount_sold;
            slice.money += amount_sold * self.price[good as usize];

            //Buying
            let buy_ratio = if self.demand[good as usize] > 0.0 {
                (self.supply[good as usize] / self.demand[good as usize]).min(1.0)
            } else {
                0.0
            };
            let amount_bought = slice.individual_demand[good as usize] * buy_ratio;
            slice.owned_goods[good as usize] += amount_bought;
            slice.money -= amount_bought * self.price[good as usize];
        }
    }
    fn update_price(&mut self, good: Good, delta_year: f64) {
        //PID for price

        const PRICE_PID_P: f64 = 0.1;
        const PRICE_PID_I: f64 = 0.00;
        const PRICE_PID_D: f64 = 0.00;

        let supply_demand_delta = self.supply[good as usize] - self.demand[good as usize];
        let error_divisor = 1000000.0; //make the error a lot smaller so it's easier to deal with
        let error = (0.0 - supply_demand_delta) / error_divisor;
        let proportional = error;
        self.supply_demand_error_integral[good as usize] += error * delta_year;
        let derivative = (error - self.previous_supply_demand_error[good as usize]) / delta_year;
        let output = PRICE_PID_P * proportional
            + PRICE_PID_I * self.supply_demand_error_integral[good as usize]
            + PRICE_PID_D * derivative;
        self.previous_supply_demand_error[good as usize] = error;

        self.price[good as usize] += output;
        // self.price[good as usize] += output.clamp(
        //     -0.1 * self.price[good as usize],
        //     0.1 * self.price[good as usize],
        // );

        // if self.supply[good as usize] > self.demand[good as usize]{
        //     self.price[good as usize] *= 0.9;
        // }else{
        //     self.price[good as usize] *= 1.1;
        // }

        self.price[good as usize] = self.price[good as usize].clamp(0.01, 1000.0);
    }
}

#[derive(Clone, Debug)]
pub struct PopSlice {
    pub population: f64,
    pub money: f64,
    pub owned_goods: [f64; Good::VARIANT_COUNT],
    individual_demand: [f64; Good::VARIANT_COUNT],
    individual_supply: [f64; Good::VARIANT_COUNT],
    previous_met_needs: VecDeque<f64>,
}
impl PopSlice {
    const NUM_MET_NEEDS_STORED: usize = 100;
}

const NUM_SLICES: usize = Culture::VARIANT_COUNT * Industry::VARIANT_COUNT;

const GOOD_STORAGE_RATIO: f64 = 1.0;

const FARM_MODIFIER: f64 = 2.1;
const MILL_MODIFIER: f64 = 2.1;
const MINE_MODIFIER: f64 = 6.5;
const SMELT_MODIFIER: f64 = 2.1;

// fn production_curve(
//     population: f64,
//     curviness: f64,
//     base_slope: f64,
//     slope_falloff: f64,
//     current_supply: f64,
// ) -> f64 {
//     let b = 1.0 / slope_falloff;
//     let output = base_slope
//         * (population - (curviness * (b * b * population * population + 1.0).sqrt()) / b)
//         + curviness * slope_falloff * base_slope;
//     let target_amount = population * GOOD_STORAGE_RATIO;
//     if current_supply > target_amount {
//         output * (target_amount / current_supply)
//     } else {
//         output
//     }
// }

fn production_curve(population: f64, curviness: f64, base_slope: f64, slope_falloff: f64) -> f64 {
    //https://www.desmos.com/calculator/majp9k0bcy
    let linear = population * base_slope;
    let output = if population < slope_falloff {
        linear
    } else {
        (1.0 - curviness) * linear
            + curviness
                * (((-1.0 / (base_slope * (population - slope_falloff)).exp()) + 1.0)
                    + slope_falloff * base_slope)
    };
    output
}

// fn production_curve(population:f64, curviness: f64,base_slope: f64, slope_falloff: f64) -> f64{
//     population*base_slope
// }

fn get_inputs(industry: Industry, population: f64, delta_year: f64) -> [f64; Good::VARIANT_COUNT] {
    let mut output = [0.0; Good::VARIANT_COUNT];

    match industry {
        Industry::Farm => {}
        Industry::Mill => {
            output[Good::Grain as usize] = population * 1.0 * delta_year * MILL_MODIFIER;
        }
        Industry::Mine => {}
        Industry::Smelter => {
            output[Good::Ore as usize] = population * 4.0 * delta_year * SMELT_MODIFIER;
        }
        Industry::Labor => {}
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
    industry_data: IndustryData,
    population: f64,
    delta_year: f64,
) -> [f64; Good::VARIANT_COUNT] {
    let mut output = [0.0; Good::VARIANT_COUNT];
    match industry {
        Industry::Farm => {
            output[Good::Grain as usize] = production_curve(
                population,
                1.0,
                (Industry::VARIANT_COUNT as f64) * FARM_MODIFIER * industry_data.productivity,
                100.0 * industry_data.size,
            ) * delta_year;
        }
        Industry::Mill => {
            output[Good::Food as usize] = production_curve(
                population,
                1.0,
                (Industry::VARIANT_COUNT as f64) * MILL_MODIFIER * industry_data.productivity,
                100.0 * industry_data.size,
            ) * delta_year;
        }
        Industry::Mine => {
            output[Good::Ore as usize] = production_curve(
                population,
                1.0,
                (Industry::VARIANT_COUNT as f64) * MINE_MODIFIER * industry_data.productivity,
                100.0 * industry_data.size,
            ) * delta_year;
        }
        Industry::Smelter => {
            output[Good::Metal as usize] = production_curve(
                population,
                1.0,
                (Industry::VARIANT_COUNT as f64) * SMELT_MODIFIER * industry_data.productivity,
                100.0 * industry_data.size,
            ) * delta_year;
        }
        Industry::Labor => {} //Produces work in a separate stage
    }

    return output;
}

fn get_intangible_outputs(
    industry: Industry,
    population: f64,
    delta_year: f64,
) -> [f64; IntangibleGoods::VARIANT_COUNT] {
    let mut output = [0.0; IntangibleGoods::VARIANT_COUNT];
    match industry {
        Industry::Farm => {}
        Industry::Mill => {}
        Industry::Mine => {}
        Industry::Smelter => {}
        Industry::Labor => output[IntangibleGoods::Work as usize] = population * delta_year,
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
                let slice_population = total_population / SLICE_COUNT as f64;
                (pop_slices.as_mut_ptr() as *mut PopSlice)
                    .offset(i as isize)
                    .write(PopSlice {
                        population: slice_population,
                        money: 1000.0,
                        // owned_goods: [1_000_000_000_000.0 / SLICE_COUNT as f64; Good::VARIANT_COUNT],
                        owned_goods: [0.0 * slice_population as f64; Good::VARIANT_COUNT],
                        // met_inputs: [0.0; Good::VARIANT_COUNT],
                        individual_demand: [0.0; Good::VARIANT_COUNT],
                        individual_supply: [0.0; Good::VARIANT_COUNT],
                        previous_met_needs: VecDeque::with_capacity(PopSlice::NUM_MET_NEEDS_STORED),
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
    pub province_id: usize,
    pub province_area: f64,
    pub pops: Pops,
    pub market: Market,
    pub position: Vector3<f64>,
    pub industry_data: [IndustryData; Industry::VARIANT_COUNT],
    pub intangible_goods: [f64; IntangibleGoods::VARIANT_COUNT],
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

#[derive(Debug)]
pub struct World {
    pub points: Vec<Vector3<f32>>,
    pub provinces: Box<[Province]>,
    pub global_market: Market,
}

// //http://repositorium.sdum.uminho.pt/bitstream/1822/6429/1/ConcaveHull_ACM_MYS.pdf
// fn concave_hull(points_list: &[usize], k: i32, points: &[ElevationVertex]) -> Vec<usize>{
//     let kk = k.max(3);
//     let mut dataset = points_list.to_vec();
//     if dataset.len() < 3{
//         panic!("A minimum of 3 dissimilar points is required")

//     }
//     if dataset.len() == 3{
//         return dataset;
//     }
//     let kk = kk.min(dataset.len() - 1);
//     let &first_point = dataset.iter().min_by_key(|&&index| FloatOrd(points[index].position.y)).unwrap();
//     let hull = first_point;
//     let mut current_point = first_point;
//     dataset.retain(|&a| a != current_point);
//     let mut previous_angle = 0;
//     let step = 5;
//     while (current_point != first_point || step == 2) && dataset.len() > 0{
//         if step == 5{
//             dataset.push(first_point);
//         }
//         let mut k_nearest_points = dataset.to_owned();
//         k_nearest_points.sort_unstable_by_key(|&i| FloatOrd((points[i].position - points[current_point].position).magnitude()));
//         k_nearest_points.truncate(kk);

//         k_nearest_points.sort_by_key(|&i| {
//             points[i].position.angle()
//         });
//         sorted_indices.sort_by(|&a, &b| {
//             let a_normal = vertices[a].normalize();
//             let a_latlong = (a_normal.z.asin(), a_normal.y.atan2(a_normal.x));
//             let b_normal = vertices[b].normalize();
//             let b_latlong = (b_normal.z.asin(), b_normal.y.atan2(b_normal.x));

//             let a1 = (a_latlong.0 - origin_latlong.0).atan2(a_latlong.1 - origin_latlong.1);
//             let a2 = (b_latlong.0 - origin_latlong.0).atan2(b_latlong.1 - origin_latlong.1);

//             FloatOrd(a1).cmp(&FloatOrd(a2))
//         });
//     }
// }

pub struct ProvinceData {
    num_samples: usize,
    elevation: f32,
    aridity: f32,
    ore: f32,
}
impl ProvinceData {
    pub fn new() -> Self {
        Self {
            num_samples: 0,
            elevation: 0.0,
            aridity: 0.0,
            ore: 0.0,
        }
    }
    pub fn add_sample(&mut self, elevation: f32, aridity: f32, ore: f32) {
        self.num_samples += 1;
        self.elevation += elevation;
        self.aridity += aridity;
        self.ore += ore;
    }
    pub fn elevation(&self) -> f32 {
        self.elevation / self.num_samples as f32
    }
    pub fn aridity(&self) -> f32 {
        self.aridity / self.num_samples as f32
    }
    pub fn ore(&self) -> f32 {
        self.ore / self.num_samples as f32
    }
}

impl World {
    pub const RADIUS: f64 = 6_378_137.0; //in meters

    pub fn intersect_planet(
        ray_origin: Vector3<f64>,
        ray_direction: Vector3<f64>,
    ) -> Option<Vector3<f64>> {
        let a = ray_direction.dot(&ray_direction);
        let b = 2.0 * ray_origin.dot(&ray_direction);
        let c = ray_origin.dot(&ray_origin) - (Self::RADIUS * Self::RADIUS);
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return None;
        } else {
            let ray_ratio = (-b + discriminant.sqrt()) / (2.0 * a);
            return Some(ray_origin + ray_direction * ray_ratio);
        }
    }

    pub fn new(
        vertices: &[Vector3<f32>],
        province_index_map: &HashMap<usize, Vec<usize>>,
        province_data_map: &HashMap<usize, ProvinceData>,
    ) -> Self {
        let mut provinces = vec![];
        for (&province_id, province_indices) in province_index_map.iter() {
            if province_indices.len() < 3 {
                println!("degenerate province given");
                continue;
            }
            let mut left_indices = province_indices.to_vec();
            let mut out_indices = Vec::with_capacity(province_indices.len());
            let mut current_index = left_indices.pop().unwrap();
            while left_indices.len() > 0 {
                let nearest_neighbour = left_indices.remove(
                    left_indices
                        .iter()
                        .enumerate()
                        .min_by_key(|(index, point_index)| {
                            FloatOrd(
                                (vertices[**point_index] - vertices[current_index]).magnitude(),
                            )
                        })
                        .expect("Odd number of province indices?")
                        .0,
                );

                out_indices.push(current_index);
                out_indices.push(nearest_neighbour);
                current_index = nearest_neighbour;
            }
            out_indices.push(*out_indices.last().unwrap());
            out_indices.push(*out_indices.first().unwrap());

            let province_origin = province_indices
                .iter()
                .map(|&index| vertices[index])
                .sum::<Vector3<f32>>()
                / province_indices.len() as f32;
            let province_origin = province_origin.normalize() * Self::RADIUS as f32;
            let mut industry_data = [IndustryData {
                productivity: f64::NAN,
                size: f64::NAN,
            }; Industry::VARIANT_COUNT];
            let province_area = province_data_map.get(&province_id).unwrap().num_samples as f64;
            for i in 0..Industry::VARIANT_COUNT {
                let industry: Industry = i.try_into().unwrap();
                industry_data[i].productivity = match industry {
                    Industry::Farm => {
                        1.0 - province_data_map.get(&province_id).unwrap().aridity() as f64
                    }
                    Industry::Mill => 1.0,
                    Industry::Mine => province_data_map.get(&province_id).unwrap().ore() as f64,
                    Industry::Smelter => 1.0,
                    Industry::Labor => 1.0,
                };
                industry_data[i].size = province_area;
            }
            provinces.push(Province {
                point_indices: out_indices,
                position: province_origin.cast(),
                pops: Pops::new(province_area * 1_000.0 * (fastrand::f64() + 1.0)),
                market: Market {
                    price: [1.0; Good::VARIANT_COUNT],
                    supply: [0.0; Good::VARIANT_COUNT],
                    demand: [0.0; Good::VARIANT_COUNT],
                    previous_supply_demand_error: [0.0; Good::VARIANT_COUNT],
                    supply_demand_error_integral: [0.0; Good::VARIANT_COUNT],
                },
                industry_data,
                province_id,
                province_area,
                intangible_goods: [0.0; IntangibleGoods::VARIANT_COUNT],
            });
        }

        Self {
            points: vertices.iter().map(|vertex| *vertex).collect(),
            provinces: provinces.into_boxed_slice(),
            global_market: Market {
                price: [1.0; Good::VARIANT_COUNT],
                supply: [0.0; Good::VARIANT_COUNT],
                demand: [0.0; Good::VARIANT_COUNT],
                previous_supply_demand_error: [0.0; Good::VARIANT_COUNT],
                supply_demand_error_integral: [0.0; Good::VARIANT_COUNT],
            },
        }
    }

    pub fn process(&mut self, delta_year: f64) {
        //Process market first, selling yesterday's produced goods and buying needs for today
        for province in self.provinces.iter_mut() {
            province.intangible_goods[IntangibleGoods::Work as usize] = 0.0; //Work doesn't carry over between ticks
            province.market.demand = [0.0; Good::VARIANT_COUNT];
            province.market.supply = [0.0; Good::VARIANT_COUNT];
            province
                .market
                .process(delta_year, &mut province.pops.pop_slices, 0.5);
            for good_index in 0..Good::VARIANT_COUNT {
                let good = Good::try_from(good_index).unwrap();
                province
                    .market
                    .buy_and_sell(good, delta_year, &mut province.pops.pop_slices);
                province.market.update_price(good, delta_year);
            }
        }
        self.global_market.demand = [0.0; Good::VARIANT_COUNT];
        self.global_market.supply = [0.0; Good::VARIANT_COUNT];
        for province in self.provinces.iter_mut() {
            self.global_market
                .process(delta_year, &mut province.pops.pop_slices, 1.0);
        }
        for good_index in 0..Good::VARIANT_COUNT {
            let good = Good::try_from(good_index).unwrap();
            for province in self.provinces.iter_mut() {
                self.global_market
                    .buy_and_sell(good, delta_year, &mut province.pops.pop_slices);
            }
            self.global_market.update_price(good, delta_year);
        }

        let mut global_tax_bank = 0.0;
        let mut global_migration_pool = 0.0;
        for (_province_index, province) in self.provinces.iter_mut().enumerate() {
            let mut migration_pool = [0.0; Industry::VARIANT_COUNT];
            //consume daily living needs
            for (slice_index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                let needs = get_needs(slice.population, delta_year);
                let industry_index = slice_index % Industry::VARIANT_COUNT;
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

                slice
                    .previous_met_needs
                    .push_back(minimum_met_need.min(1.0));
                if slice.previous_met_needs.len() > PopSlice::NUM_MET_NEEDS_STORED {
                    slice.previous_met_needs.remove(0);
                }
                for (good_index, &need) in needs.iter().enumerate() {
                    slice.owned_goods[good_index] -= need * minimum_met_need;
                }

                let met_needs_ratio = slice.previous_met_needs.iter().sum::<f64>()
                    / (PopSlice::NUM_MET_NEEDS_STORED.min(slice.previous_met_needs.len())) as f64;

                let growth_ratio = if met_needs_ratio > 0.999 {
                    0.01
                } else if met_needs_ratio > 0.01 {
                    0.0
                } else {
                    -0.1
                };

                slice.population +=
                    (slice.population * growth_ratio * delta_year).max(-slice.population);
                if met_needs_ratio < 0.95 {
                    let migration_amount =
                        (slice.population * 0.1 * delta_year).min(slice.population);
                    migration_pool[industry_index] += migration_amount * (10.0 / 10.0);
                    global_migration_pool += migration_amount * (0.0 / 10.0);
                    slice.population -= migration_amount;
                }
                // if minimum_met_need < 0.50 {
                //     let migration_amount =
                //         (slice.population * 0.1 * delta_year).min(slice.population);
                //     migration_pool += migration_amount * (5.0 / 10.0);
                //     global_migration_pool += migration_amount * (5.0 / 10.0);
                //     slice.population -= migration_amount;
                // }
            }

            // let mut migrated_amount = 0.0;
            //process provincial migrations
            for (slice_index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                let industry_index = slice_index % Industry::VARIANT_COUNT;
                let migration_amount = migration_pool
                    .iter()
                    .enumerate()
                    .filter(|&(pool_industry_index, _migration)| {
                        pool_industry_index != industry_index
                    })
                    .map(|a| a.1)
                    .sum::<f64>()
                    / (Culture::VARIANT_COUNT * (Industry::VARIANT_COUNT - 1)) as f64;
                slice.population += migration_amount;
                // migrated_amount += migration_amount;
            }
            // if (migrated_amount - migration_pool.iter().sum::<f64>()).abs() > 1.0{
            //     dbg!(migrated_amount);
            //     dbg!(migration_pool);
            // }

            let industry_populations = {
                let mut industry_populations = [0.0; Industry::VARIANT_COUNT];

                for (slice_index, slice) in province.pops.pop_slices.iter().enumerate() {
                    let industry =
                        Industry::try_from(slice_index % Industry::VARIANT_COUNT).unwrap();
                    industry_populations[industry as usize] += slice.population;
                }
                industry_populations
            };
            //use remaining resources for industry
            let mut _money_sum = 0.0;
            for (slice_index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                _money_sum += slice.money;
                let industry = Industry::try_from(slice_index % Industry::VARIANT_COUNT).unwrap();
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

                let mut outputs = get_outputs(
                    industry,
                    province.industry_data[industry as usize],
                    industry_populations[industry as usize],
                    delta_year,
                );

                //correct for entire industry population
                for output in &mut outputs {
                    *output *= slice.population / industry_populations[industry as usize];
                }
                let intangible_outputs =
                    get_intangible_outputs(industry, slice.population, delta_year);

                for (intangible_good_index, output) in intangible_outputs.iter().enumerate() {
                    province.intangible_goods[intangible_good_index] += output;
                }
                for (good_index, output) in outputs.iter().enumerate() {
                    let good = Good::try_from(good_index).unwrap();
                    slice.owned_goods[good as usize] -= inputs[good as usize] * minimum_met_input;
                    slice.owned_goods[good as usize] += output * minimum_met_input;
                    slice.owned_goods[good as usize] =
                        slice.owned_goods[good as usize].min(slice.population * GOOD_STORAGE_RATIO)
                }
            }
            let mut tax_bank = 0.0;
            let mut _population_sum = 0.0;
            for (_index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                let tax_amount = (slice.money * 0.1 * delta_year).min(slice.money);
                let local_tax_proportion = 0.9;
                tax_bank += tax_amount * local_tax_proportion;
                global_tax_bank += tax_amount * (1.0 - local_tax_proportion);
                slice.money -= tax_amount;
                _population_sum += slice.population;
            }
            for (_index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                let tax_payment = tax_bank * (1.0 / NUM_SLICES as f64);
                slice.money += tax_payment;
            }

            // dbg!(money_sum);
        }
        let num_provinces = self.provinces.len();
        for province in self.provinces.iter_mut() {
            let migration_pool = global_migration_pool / num_provinces as f64;
            let tax_bank = global_tax_bank / num_provinces as f64;
            for slice in province.pops.pop_slices.iter_mut() {
                let migration_amount = migration_pool / NUM_SLICES as f64;
                slice.population += migration_amount;

                let tax_payment = tax_bank * (1.0 / NUM_SLICES as f64);
                slice.money += tax_payment;
            }
        }
    }
}
