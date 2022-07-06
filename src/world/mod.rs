use std::{
    cell::UnsafeCell,
    collections::{hash_map::Entry, HashMap, HashSet, VecDeque},
    fmt::Display,
    hash::Hash,
    mem::MaybeUninit,
    ops::{Add, Index, IndexMut, Mul},
    time::Instant,
};

pub mod organization;

use bincode::{Decode, Encode};
use egui::plot::Value;
use float_ord::FloatOrd;
use nalgebra::{ComplexField, Vector3};
use nohash_hasher::{BuildNoHashHasher, IsEnabled, NoHashHasher};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use variant_count::VariantCount;

use crate::{
    support::{hash_usize_fast, map_range_linear_f64},
    world::organization::Ideology,
};

use self::organization::{Military, MilitaryType, Organization, OrganizationKey, Relation};

const BASE_INDUSTRY_SIZE: f64 = 1000.0;
const TRICKLEBACK_RATIO: f64 = 0.25;

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
    Forage,
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct IndustryData {
    pub productivity: f64,
    pub size: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
        trade_modifier: f64, //modifies supply and demand so that some provinces are less efficient to import/export to/from
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

                let affordable_good = if num_distinct_goods_wanted > 0 {
                    (slice.money / num_distinct_goods_wanted as f64) / self.price[good as usize]
                } else {
                    0.0
                };
                if affordable_good.is_nan() {
                    dbg!(slice.money);
                    dbg!(num_distinct_goods_wanted);
                    dbg!(self.price[good as usize]);
                }
                assert!(!slice.money.is_nan());
                assert!(!self.price[good as usize].is_nan());
                assert!(!affordable_good.is_nan());
                // let good_demand = (good_needed_to_be_bought * 1.5).min(affordable_good);
                let good_demand = (good_needed_to_be_bought).min(affordable_good) * trade_modifier;
                assert!(!good_needed_to_be_bought.is_nan());
                assert!(!good_demand.is_nan());
                slice.individual_demand[good as usize] = good_demand;
                self.demand[good as usize] += good_demand;

                let good_supply =
                    (slice.owned_goods[good as usize] - good_needed).max(0.0) * trade_modifier;
                slice.individual_supply[good as usize] = good_supply;
                self.supply[good as usize] += good_supply;

                leftover_money -= good_demand * self.price[good as usize];
            }
            leftover_money *= spend_leftovers_ratio;
            if leftover_money > 0.0 {
                for good_index in 0..Good::VARIANT_COUNT {
                    let amount_to_spend = (leftover_money / Good::VARIANT_COUNT as f64) * 0.5;
                    let amount_to_buy = amount_to_spend / self.price[good_index];
                    assert!(!amount_to_buy.is_nan());

                    slice.individual_demand[good_index] += amount_to_buy * trade_modifier;
                    assert!(!trade_modifier.is_nan());
                    self.demand[good_index] += amount_to_buy * trade_modifier;
                }
            }
        }
    }
    fn buy_and_sell(&mut self, good: Good, delta_year: f64, pop_slices: &mut [PopSlice]) {
        let mut trader_total_profit = 0.0;
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
            assert!(!slice.money.is_nan());
            let price_of_goods_sold = amount_sold * self.price[good as usize];
            slice.money += price_of_goods_sold;
            assert!(!slice.money.is_nan());

            //Buying
            let buy_ratio = if self.demand[good as usize] > 0.0 {
                (self.supply[good as usize] / self.demand[good as usize]).min(1.0)
            } else {
                0.0
            };
            let amount_bought = slice.individual_demand[good as usize] * buy_ratio;
            assert!(!buy_ratio.is_nan());
            assert!(!slice.individual_demand[good as usize].is_nan());
            assert!(!amount_bought.is_nan());

            slice.owned_goods[good as usize] += amount_bought;
            assert!(!slice.owned_goods[good as usize].is_nan());

            let price_of_goods_bought = amount_bought * self.price[good as usize];
            assert!(!self.price[good as usize].is_nan());
            assert!(!price_of_goods_bought.is_nan());

            slice.money -= price_of_goods_bought;

            assert!(!slice.money.is_nan());
        }
    }
    fn update_price(&mut self, good: Good, delta_year: f64, error_divisor: f64) {
        //PID for price
        const PRICE_PID_P: f64 = 1.0;
        const PRICE_PID_I: f64 = 0.00;
        const PRICE_PID_D: f64 = 0.00;

        let supply_demand_delta = self.supply[good as usize] - self.demand[good as usize];
        // let error_divisor = 1000000.0; //make the error a lot smaller so it's easier to deal with
        let error = (0.0 - supply_demand_delta) / error_divisor;
        let proportional = error * delta_year;
        self.supply_demand_error_integral[good as usize] += error * delta_year;
        let derivative = (error - self.previous_supply_demand_error[good as usize]) / delta_year;
        let output = PRICE_PID_P * proportional
            + PRICE_PID_I * self.supply_demand_error_integral[good as usize]
            + PRICE_PID_D * derivative;
        self.previous_supply_demand_error[good as usize] = error;

        self.price[good as usize] += output;
        assert!(!self.price[good as usize].is_nan());

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
        assert!(!self.price[good as usize].is_nan());

        // self.price[good as usize] = 0.01;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PopSlice {
    pub population: f64,
    pub money: f64,
    pub owned_goods: [f64; Good::VARIANT_COUNT],
    individual_demand: [f64; Good::VARIANT_COUNT],
    individual_supply: [f64; Good::VARIANT_COUNT],
    previous_met_needs: VecDeque<f64>,
    pub minimum_met_needs: f64,
    pub trickleback: f64,
}
impl PopSlice {
    const NUM_MET_NEEDS_STORED: usize = 30;
}

const NUM_SLICES: usize = Culture::VARIANT_COUNT * Industry::VARIANT_COUNT;

const GOOD_STORAGE_RATIO: f64 = 1000.0;

const SIM_START_YEAR: f64 = 2022.0;
const NEEDS_GRACE_PERIOD: f64 = 100.0;

// const FARM_MODIFIER: f64 = 2.1;
// const MILL_MODIFIER: f64 = 2.1;
// const MINE_MODIFIER: f64 = 6.5;
// const SMELT_MODIFIER: f64 = 2.1;
const FORAGE_MODIFIER: f64 = 1.0;
const FARM_MODIFIER: f64 = 1.1;
const MILL_MODIFIER: f64 = 1.1;
const MINE_MODIFIER: f64 = 10.1;
const SMELT_MODIFIER: f64 = 1.1;

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

fn kernel(value: f64, horizontal_offset: f64, vertical_offset: f64, width: f64) -> f64 {
    1.0 - ((value - horizontal_offset) * (1.0 / width)).powf(2.0) + vertical_offset
}

// fn production_curve(population:f64, curviness: f64,base_slope: f64, slope_falloff: f64) -> f64{
//     population*base_slope
// }

fn get_inputs(industry: Industry, population: f64, delta_year: f64) -> [f64; Good::VARIANT_COUNT] {
    let mut output = [0.0; Good::VARIANT_COUNT];

    match industry {
        Industry::Forage => {}
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
        Industry::Forage => {
            // output[Good::Food as usize] = production_curve(
            //     population,
            //     1.0,
            //     (Industry::VARIANT_COUNT as f64) * FORAGE_MODIFIER * industry_data.productivity,
            //     0.0001 * industry_data.size,
            // ) * delta_year;
            // output[Good::Metal as usize] = production_curve(
            //     population,
            //     1.0,
            //     (Industry::VARIANT_COUNT as f64) * FORAGE_MODIFIER * industry_data.productivity,
            //     0.001 * industry_data.size,
            // ) * delta_year;
        }
        Industry::Farm => {
            output[Good::Grain as usize] = production_curve(
                population,
                1.0,
                (Industry::VARIANT_COUNT as f64) * FARM_MODIFIER * industry_data.productivity,
                industry_data.size,
            ) * delta_year;
            output[Good::Food as usize] = production_curve(
                population,
                1.0,
                (Industry::VARIANT_COUNT as f64)
                    * FARM_MODIFIER
                    * 0.01
                    * industry_data.productivity,
                industry_data.size,
            ) * delta_year;
        }
        Industry::Mill => {
            output[Good::Food as usize] = production_curve(
                population,
                1.0,
                (Industry::VARIANT_COUNT as f64) * MILL_MODIFIER * industry_data.productivity,
                industry_data.size,
            ) * delta_year;
        }
        Industry::Mine => {
            output[Good::Ore as usize] = production_curve(
                population,
                1.0,
                (Industry::VARIANT_COUNT as f64) * MINE_MODIFIER * industry_data.productivity,
                industry_data.size,
            ) * delta_year;
            output[Good::Metal as usize] = production_curve(
                population,
                1.0,
                (Industry::VARIANT_COUNT as f64)
                    * MINE_MODIFIER
                    * 0.01
                    * industry_data.productivity,
                industry_data.size,
            ) * delta_year;
        }
        Industry::Smelter => {
            output[Good::Metal as usize] = production_curve(
                population,
                1.0,
                (Industry::VARIANT_COUNT as f64) * SMELT_MODIFIER * industry_data.productivity,
                industry_data.size,
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
        Industry::Forage => {}
        Industry::Farm => {}
        Industry::Mill => {}
        Industry::Mine => {}
        Industry::Smelter => {}
        Industry::Labor => output[IntangibleGoods::Work as usize] = population * delta_year,
    }
    return output;
}

#[derive(Serialize, Deserialize)]
pub struct Histories {
    pub population: ProvinceMap<VecDeque<(f64, f64)>>,
    pub prices: ProvinceMap<Box<[VecDeque<(f64, f64)>]>>,
    pub supply: ProvinceMap<Box<[VecDeque<(f64, f64)>]>>,
    pub demand: ProvinceMap<Box<[VecDeque<(f64, f64)>]>>,
    pub global_prices: Box<[VecDeque<(f64, f64)>]>,
    pub global_supply: Box<[VecDeque<(f64, f64)>]>,
    pub global_demand: Box<[VecDeque<(f64, f64)>]>,
}
impl Histories {
    fn new(pre_sim_steps_capacity: usize, num_provinces: usize) -> Self {
        // let mut population = HashMap::with_capacity_and_hasher(num_provinces, NoHashHasher::default());
        // for key in province_keys{
        //     population.entry(key)
        // }
        let population =
            vec![VecDeque::with_capacity(pre_sim_steps_capacity); num_provinces].into_boxed_slice();
        let prices = vec![
            vec![VecDeque::with_capacity(pre_sim_steps_capacity); Good::VARIANT_COUNT]
                .into_boxed_slice();
            num_provinces
        ]
        .into_boxed_slice();
        let supply = vec![
            vec![VecDeque::with_capacity(pre_sim_steps_capacity); Good::VARIANT_COUNT]
                .into_boxed_slice();
            num_provinces
        ]
        .into_boxed_slice();
        let demand = vec![
            vec![VecDeque::with_capacity(pre_sim_steps_capacity); Good::VARIANT_COUNT]
                .into_boxed_slice();
            num_provinces
        ]
        .into_boxed_slice();

        // Self {
        //     population: [VecDeque::with_capacity(pre_sim_steps_capacity); num_provinces]
        //         .into_boxed_slice(),
        //     prices: vec![
        //         vec![VecDeque::with_capacity(pre_sim_steps_capacity); Good::VARIANT_COUNT]
        //             .into_boxed_slice();
        //         num_provinces
        //     ]
        //     .into_boxed_slice(),
        //     global_prices: vec![
        //         VecDeque::with_capacity(pre_sim_steps_capacity);
        //         Good::VARIANT_COUNT
        //     ]
        //     .into_boxed_slice(),
        // }
        Self {
            population: ProvinceMap(population),
            prices: ProvinceMap(prices),
            supply: ProvinceMap(supply),
            demand: ProvinceMap(demand),
            global_prices: vec![
                VecDeque::with_capacity(pre_sim_steps_capacity);
                Good::VARIANT_COUNT
            ]
            .into_boxed_slice(),
            global_supply: vec![
                VecDeque::with_capacity(pre_sim_steps_capacity);
                Good::VARIANT_COUNT
            ]
            .into_boxed_slice(),
            global_demand: vec![
                VecDeque::with_capacity(pre_sim_steps_capacity);
                Good::VARIANT_COUNT
            ]
            .into_boxed_slice(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Pops {
    pub pop_slices: [PopSlice; NUM_SLICES],
}

impl Pops {
    pub fn new(total_population: f64) -> Self {
        const SLICE_COUNT: usize = NUM_SLICES;
        unsafe {
            let mut pop_slices: MaybeUninit<[PopSlice; SLICE_COUNT]> = MaybeUninit::uninit();
            let rng = fastrand::Rng::new();
            for i in 0..SLICE_COUNT {
                let slice_population = (total_population / SLICE_COUNT as f64) * (rng.f64() + 0.5);
                (pop_slices.as_mut_ptr() as *mut PopSlice)
                    .offset(i as isize)
                    .write(PopSlice {
                        population: slice_population,
                        money: 1.0 * slice_population,
                        // owned_goods: [1_000_000_000_000.0 / SLICE_COUNT as f64; Good::VARIANT_COUNT],
                        owned_goods: [0.0 * slice_population as f64; Good::VARIANT_COUNT],
                        // met_inputs: [0.0; Good::VARIANT_COUNT],
                        individual_demand: [0.0; Good::VARIANT_COUNT],
                        individual_supply: [0.0; Good::VARIANT_COUNT],
                        previous_met_needs: VecDeque::with_capacity(PopSlice::NUM_MET_NEEDS_STORED),
                        minimum_met_needs: f64::NAN,
                        trickleback: 0.0,
                    })
            }
            let pop_slices = pop_slices.assume_init();
            Self { pop_slices }
        }

        // let pop_slices: Vec<_> = std::iter::repeat_with(||{
        //     let slice_population = total_population / SLICE_COUNT as f64;
        //     PopSlice {
        //                         population: slice_population,
        //                         money: 1000.0,
        //                         // owned_goods: [1_000_000_000_000.0 / SLICE_COUNT as f64; Good::VARIANT_COUNT],
        //                         owned_goods: [0.0 * slice_population as f64; Good::VARIANT_COUNT],
        //                         // met_inputs: [0.0; Good::VARIANT_COUNT],
        //                         individual_demand: [0.0; Good::VARIANT_COUNT],
        //                         individual_supply: [0.0; Good::VARIANT_COUNT],
        //                         previous_met_needs: VecDeque::with_capacity(PopSlice::NUM_MET_NEEDS_STORED),
        //                         minimum_met_needs: f64::NAN,
        //                     }
        // }).take(SLICE_COUNT).collect();
        // Self { pop_slices: pop_slices.try_into().unwrap() }
    }
    pub fn population(&self) -> f64 {
        self.pop_slices.iter().map(|slice| slice.population).sum()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Province {
    pub name: String,
    pub point_indices: Vec<usize>,
    pub neighbouring_provinces: Box<[ProvinceKey]>,
    pub province_key: ProvinceKey,
    pub province_area: f64,
    pub pops: Pops,
    pub market: Market,
    pub position: Vector3<f64>,
    pub industry_data: [IndustryData; Industry::VARIANT_COUNT],
    pub intangible_goods: [f64; IntangibleGoods::VARIANT_COUNT],
    pub feb_temp: f64,
    pub july_temp: f64,
    pub trader_cost_multiplier: f64, //Excess goes to traders in global market. Abstraction for difficulties in exporting from remote or blockaded provinces, WIP and not really functional
    pub tax_bank: f64,
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

impl Province {
    pub fn get_current_temp(&self, current_year: f64) -> f64 {
        let mix_factor = (current_year.sin() + 1.0) / 2.0;
        self.feb_temp * (1.0 - mix_factor) + self.july_temp * mix_factor
    }
}

#[derive(Clone, Debug)]
pub struct ProvinceData {
    name: String,
    num_samples: usize,
    elevation: f64,
    population: f64,
    aridity: f64,
    feb_temps: f64,
    july_temps: f64,
    ore: f64,
    organization_owner: Vec<Option<usize>>,
}
impl ProvinceData {
    pub fn new() -> Self {
        Self {
            num_samples: 0,
            elevation: 0.0,
            aridity: 0.0,
            feb_temps: 0.0,
            july_temps: 0.0,
            ore: 0.0,
            population: 0.0,
            organization_owner: vec![],
            name: String::new(),
        }
    }
    pub fn add_sample(
        &mut self,
        name: &str,
        elevation: f64,
        aridity: f64,
        feb_temps: f64,
        july_temps: f64,
        ore: f64,
        population: f64,
        owner: Option<usize>,
    ) {
        self.num_samples += 1;
        self.elevation += elevation;
        self.aridity += aridity;
        self.feb_temps += feb_temps;
        self.july_temps += july_temps;
        self.ore += ore;
        self.population += population;
        self.organization_owner.push(owner);
        self.name = name.to_string();
    }
    pub fn elevation(&self) -> f64 {
        if self.num_samples > 0 {
            self.elevation / self.num_samples as f64
        } else {
            0.0
        }
    }
    pub fn aridity(&self) -> f64 {
        if self.num_samples > 0 {
            self.aridity / self.num_samples as f64
        } else {
            0.0
        }
    }
    pub fn feb_temps(&self) -> f64 {
        if self.num_samples > 0 {
            self.feb_temps / self.num_samples as f64
        } else {
            0.0
        }
    }
    pub fn july_temps(&self) -> f64 {
        if self.num_samples > 0 {
            self.july_temps / self.num_samples as f64
        } else {
            0.0
        }
    }
    pub fn ore(&self) -> f64 {
        if self.num_samples > 0 {
            self.ore / self.num_samples as f64
        } else {
            0.0
        }
    }
    pub fn population(&self) -> f64 {
        self.population
    }
    pub fn owner(&self) -> Option<usize> {
        let mut occurences = HashMap::with_capacity(self.organization_owner.len());
        for owner in &self.organization_owner{
            if let Some(owner) = owner{
                *occurences.entry(owner).or_insert(0usize) += 1;
            } 
        }
        occurences.iter().max_by_key(|o|o.1).map(|(&&id, _)| id)
    }
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct ProvinceKey(pub usize);
impl IsEnabled for ProvinceKey {}

pub type OrganizationMap<T> = HashMap<OrganizationKey, T, BuildNoHashHasher<OrganizationKey>>;
pub type OrganizationMap2D<T> =
    HashMap<(OrganizationKey, OrganizationKey), T, BuildNoHashHasher<OrganizationKey>>;

#[derive(Serialize, Deserialize)]
pub struct ProvinceMap<T>(pub Box<[T]>);

impl<T> Index<ProvinceKey> for ProvinceMap<T> {
    type Output = T;

    fn index(&self, index: ProvinceKey) -> &Self::Output {
        &self.0[index.0]
    }
}

impl<T> IndexMut<ProvinceKey> for ProvinceMap<T> {
    fn index_mut(&mut self, index: ProvinceKey) -> &mut Self::Output {
        &mut self.0[index.0]
    }
}

// #[derive(Serialize, Deserialize, Debug)]
// pub struct TravelCost{
//     cost: f64
// }

// #[derive(Serialize, Deserialize)]
// pub struct TravelCostMap
// {
//     map: ProvinceMap2D<TravelCost>
// }
// impl TravelCostMap
// {
//     pub fn new(provinces: &HashMap<ProvinceKey, Province, BuildNoHashHasher<ProvinceKey>>) -> Self{
//         let mut costs = ProvinceMap2D::with_capacity_and_hasher(provinces.len() * provinces.len(), BuildNoHashHasher::default());
//         dbg!(provinces.len());
//         println!("new travel cost");
//         let mut i = 0;
//         for province_a_key in provinces.keys(){
//             dbg!(i);
//             for province_b_key in provinces.keys(){
//                 i += 1;
//                 let (&first_province_key, &second_province_key) = if province_a_key > province_b_key{
//                      (province_a_key,province_b_key)
//                 } else{(province_b_key, province_a_key)};

//                 let distance = provinces[&first_province_key].position.dot(&provinces[&second_province_key].position).acos();

//                 costs.entry((first_province_key,second_province_key)).or_insert(TravelCost{
//                     cost: distance,
//                 });
//             }
//         }
//         println!("travel cost done");
//         Self{map: costs}
//     }

//     pub fn get_travel_cost(&self, province_a_key: ProvinceKey, province_b_key: ProvinceKey) -> &TravelCost{
//         let (first_province_key, second_province_key) = if province_a_key > province_b_key{
//             (province_a_key,province_b_key)
//        } else{(province_b_key, province_a_key)};
//        self.map.get(&(first_province_key, second_province_key)).unwrap()
//     }
//     pub fn get_travel_cost_mut(&mut self, province_a_key: ProvinceKey, province_b_key: ProvinceKey) -> &mut TravelCost{
//         let (first_org_key, second_org_key) = if province_a_key > province_b_key{
//             (province_a_key,province_b_key)
//        } else{(province_b_key, province_a_key)};
//        self.map.get_mut(&(first_org_key, second_org_key)).unwrap()
//     }

//     pub fn get_average_cost(&self,province: ProvinceKey) -> f64{
//         let mut count = 0.0;
//         self.map.iter().filter(|(&(key_a,key_b),_)| key_a == province || key_b == province).map(|((_, _), cost)|{
//             count += 1.0;
//             cost.cost
//         }).sum::<f64>() / count
//     }
// }

#[derive(Serialize, Deserialize)]
pub struct RelationMap {
    map: OrganizationMap2D<Relation>,
}
impl RelationMap {
    pub fn new(
        organizations: &HashMap<OrganizationKey, Organization, BuildNoHashHasher<OrganizationKey>>,
    ) -> Self {
        let mut relations = OrganizationMap2D::with_capacity_and_hasher(
            organizations.len(),
            BuildNoHashHasher::default(),
        );
        for org_a_key in organizations.keys() {
            for org_b_key in organizations.keys() {
                let (&first_org_key, &second_org_key) = if org_a_key > org_b_key {
                    (org_a_key, org_b_key)
                } else {
                    (org_b_key, org_a_key)
                };

                relations
                    .entry((first_org_key, second_org_key))
                    .or_insert(Relation { at_war: false });
            }
        }
        Self { map: relations }
    }

    pub fn get_relations(
        &self,
        org_a_key: OrganizationKey,
        org_b_key: OrganizationKey,
    ) -> &Relation {
        let (first_org_key, second_org_key) = if org_a_key > org_b_key {
            (org_a_key, org_b_key)
        } else {
            (org_b_key, org_a_key)
        };
        self.map.get(&(first_org_key, second_org_key)).unwrap()
    }
    pub fn get_relations_mut(
        &mut self,
        org_a_key: OrganizationKey,
        org_b_key: OrganizationKey,
    ) -> &mut Relation {
        let (first_org_key, second_org_key) = if org_a_key > org_b_key {
            (org_a_key, org_b_key)
        } else {
            (org_b_key, org_a_key)
        };
        self.map.get_mut(&(first_org_key, second_org_key)).unwrap()
    }
}

fn get_battle_seed(current_year: f64, offset: usize, max: usize) -> usize {
    let seed = (current_year * 365.0 * 4.0) as usize + offset;
    hash_usize_fast(seed) % max
}

#[derive(Serialize, Deserialize)]
pub struct World {
    pub points: Vec<Vector3<f32>>,
    pub provinces: ProvinceMap<Province>,
    pub global_market: Market,
    pub histories: Histories,
    pub current_year: f64,
    pub organizations: OrganizationMap<Organization>,
    pub relations: RelationMap,
    pub selected_province: Option<ProvinceKey>,
    pub targeted_province: Option<ProvinceKey>,
    pub selected_organization: Option<OrganizationKey>,
    pub player_organization: OrganizationKey,
}
impl World {
    pub const RADIUS: f64 = 6_378_137.0; //in meters
    pub const STEP_LENGTH: f64 = 1.0 / 365.0 / 24.0;
    pub const PRE_SIM_STEP_LENGTH: f64 = 0.5;

    pub fn transfer_troops(
        &mut self,
        organization: OrganizationKey,
        source_province: ProvinceKey,
        destination_province: ProvinceKey,
        ratio: f64,
    ) {
        for military in &mut self
            .organizations
            .get_mut(&organization)
            .unwrap()
            .militaries
        {
            let transfer_amount = military.deployed_forces[source_province] * ratio;
            military.deployed_forces[destination_province] += transfer_amount;
            military.deployed_forces[source_province] -= transfer_amount;
        }
    }

    pub fn process_command(&mut self, command: &str) -> String {
        let mut words = command.split_whitespace();
        match words.next() {
            Some(word) => match word {
                "get_total_pops" => {
                    return format!("{:.0}\n", self.get_total_population());
                }
                "get_total_money" => {
                    return format!("{:.0}\n", self.get_total_money());
                }
                "add_pops" => match self.selected_province {
                    Some(province) => match words.next() {
                        Some(argument) => match argument.parse::<f64>() {
                            Ok(pops_to_add) => {
                                for pop in &mut self.provinces[province].pops.pop_slices {
                                    pop.population += pops_to_add / (NUM_SLICES as f64)
                                }
                                return "Success\n".to_string();
                            }
                            Err(_) => return "Invalid argument type\n".to_string(),
                        },
                        None => return "Must enter an argument\n".to_string(),
                    },
                    None => return "Must select a province\n".to_string(),
                },
                "add_money" => match self.selected_province {
                    Some(province) => match words.next() {
                        Some(argument) => match argument.parse::<f64>() {
                            Ok(money_to_add) => {
                                for pop in &mut self.provinces[province].pops.pop_slices {
                                    pop.money += money_to_add / (NUM_SLICES as f64)
                                }
                                return "Success\n".to_string();
                            }
                            Err(_) => return "Invalid argument type\n".to_string(),
                        },
                        None => return "Must enter an argument\n".to_string(),
                    },
                    None => return "Must select a province\n".to_string(),
                },
                "set_travel" => match self.selected_province {
                    Some(province) => match words.next() {
                        Some(argument) => match argument.parse() {
                            Ok(new_trader_cost) => {
                                self.provinces[province].trader_cost_multiplier = new_trader_cost;
                                return "Success\n".to_string();
                            }
                            Err(_) => return "Invalid argument type\n".to_string(),
                        },
                        None => return "Must enter an argument\n".to_string(),
                    },
                    None => return "Must select a province\n".to_string(),
                },
                "set_travel_all" => match words.next() {
                    Some(argument) => match argument.parse() {
                        Ok(new_trader_cost) => {
                            for province in self.provinces.0.iter_mut() {
                                province.trader_cost_multiplier = new_trader_cost;
                            }
                            return "Success\n".to_string();
                        }
                        Err(_) => return "Invalid argument type\n".to_string(),
                    },
                    None => return "Must enter an argument\n".to_string(),
                },
                "list_orgs" => {
                    let mut out_string = String::new();
                    for (key, org) in &self.organizations {
                        out_string.push_str(&format!("Key: {:}, org: {:}\n", key.0, org.name));
                    }
                    return out_string;
                }
                "add_troops" => {
                    if let (Some(selected_province_key), Some(selected_org_key)) =
                        (self.selected_province, self.selected_organization)
                    {
                        match self
                            .organizations
                            .get_mut(&selected_org_key)
                            .unwrap()
                            .militaries
                            .iter_mut()
                            .find(|m| matches!(m.military_type, MilitaryType::Army))
                        {
                            Some(military) => {
                                if let Ok(amount) = words
                                    .next()
                                    .unwrap_or_else(|| return "No amount provided\n")
                                    .parse::<f64>()
                                {
                                    military.deployed_forces[selected_province_key] += amount;
                                    return format!("Added {amount} troops in province {:} for organization {:}\n", selected_province_key.0, selected_org_key.0);
                                } else {
                                    return "Invalid amount provided\n".to_string();
                                }
                            }
                            None => return "Organization has no army\n".to_string(),
                        }
                    } else {
                        return "You must have a province and organization selected\n".to_string();
                    }
                }
                "add_mil" => match self.selected_organization {
                    Some(org_key) => match words.next() {
                        Some(military_type) => {
                            let military_type = match military_type.to_lowercase().as_str() {
                                "army" => MilitaryType::Army,
                                "enforcer" => MilitaryType::Enforcer,
                                _ => return "Invalid military type\n".to_string(),
                            };
                            self.organizations
                                .get_mut(&org_key)
                                .unwrap()
                                .militaries
                                .push(Military::new(military_type, self.provinces.0.len()));
                            return "Succesfully created a military\n".to_string();
                        }
                        None => return "You must enter a type\n".to_string(),
                    },
                    None => return "Must select an organization\n".to_string(),
                },
                "declare_war_all" => match self.selected_organization {
                    Some(org_key) => {
                        for &enemy_key in self.organizations.keys() {
                            if enemy_key != org_key {
                                self.relations.get_relations_mut(org_key, enemy_key).at_war = true;
                            }
                        }
                        return "Successfully declared war on all organizations\n".to_string();
                    }
                    None => return "Must select an organization\n".to_string(),
                },
                _ => return "Invalid command\n".to_string(),
            },
            None => return "No command entered\n".to_string(),
        }
    }

    // pub fn update_travel_times(&mut self){
    //     for (&key, province) in &mut self.provinces{
    //         let cost = self.travel_costs.get_average_cost(key);
    //         province.trader_cost_multiplier = 1.0 + cost;
    //     }
    // }

    pub fn get_total_population(&self) -> f64 {
        self.provinces
            .0
            .iter()
            .map(|a| a.pops.population())
            .sum::<f64>()
            .max(1.0)
    }
    pub fn get_total_money(&self) -> f64 {
        self.provinces
            .0
            .iter()
            .map(|a| a.pops.pop_slices.iter().map(|a| a.money).sum::<f64>())
            .sum::<f64>()
            .max(1.0)
    }

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

    pub fn load(file: &mut std::fs::File) -> Self {
        let mut reader = snap::read::FrameDecoder::new(file);
        serde_json::from_reader(&mut reader).expect("Failed to decode World")
    }

    pub fn save(&self, filename: &str) {
        return;
        let out_writer = std::fs::File::create(filename).expect("Failed to open file for writing");
        let mut out_writer = snap::write::FrameEncoder::new(out_writer);
        serde_json::to_writer(&mut out_writer, self).expect("Failed to encode World")
    }

    pub fn new(
        vertices: &[Vector3<f64>],
        province_indices: &ProvinceMap<Vec<usize>>,
        province_data: &ProvinceMap<ProvinceData>,
        nation_names: &[String],
    ) -> Self {
        println!("Creating world");
        let mut organizations = HashMap::with_hasher(BuildNoHashHasher::default());
        let num_provinces = province_indices.0.len();
        let mut provinces = vec![];

        let total_area = province_data
            .0
            .iter()
            .map(|d| d.num_samples as f64)
            .sum::<f64>();
        let mut country_id_to_org_key: HashMap<usize, OrganizationKey, BuildNoHashHasher<usize>> =
            HashMap::with_hasher(BuildNoHashHasher::default());
        let rng = fastrand::Rng::new();
        for (province_key, province_indices) in province_indices.0.iter().enumerate() {
            let province_key = ProvinceKey(province_key);
            if province_indices.len() < 3 {
                println!("degenerate province given");
                continue;
            }
            let out_indices = province_indices.to_vec();

            let province_origin = province_indices
                .iter()
                .map(|&index| vertices[index].cast::<f32>())
                .sum::<Vector3<f32>>()
                / province_indices.len() as f32;
            let province_origin = province_origin.normalize() * Self::RADIUS as f32;
            let mut industry_data = [IndustryData {
                productivity: f64::NAN,
                size: f64::NAN,
            }; Industry::VARIANT_COUNT];
            let province_area = province_data[province_key].num_samples as f64;
            let population = province_data[province_key].population().max(10_000.0);
            for i in 0..Industry::VARIANT_COUNT {
                let industry: Industry = i.try_into().unwrap();
                    let pop_size = population / (Industry::VARIANT_COUNT as f64);
                industry_data[i].size = pop_size
                        * match industry {
                            Industry::Forage => 1.0,
                            Industry::Farm => {
                                (province_data[province_key].aridity() as f64 - 0.3).clamp(0.0, 1.1)
                            }
                            Industry::Mill => 1.0,
                            Industry::Mine => province_data[province_key].ore() as f64,
                            Industry::Smelter => 1.0,
                            Industry::Labor => 1.0,
                        };
            }

            provinces.push(Province {
                name: province_data[province_key].name.clone(),
                point_indices: out_indices,
                position: province_origin.cast(),
                pops: Pops::new(population),
                market: Market {
                    price: [1.0; Good::VARIANT_COUNT],
                    supply: [0.0; Good::VARIANT_COUNT],
                    demand: [0.0; Good::VARIANT_COUNT],
                    previous_supply_demand_error: [0.0; Good::VARIANT_COUNT],
                    supply_demand_error_integral: [0.0; Good::VARIANT_COUNT],
                },
                industry_data,
                province_key,
                province_area,
                intangible_goods: [0.0; IntangibleGoods::VARIANT_COUNT],
                trader_cost_multiplier: 1.0,
                feb_temp: province_data[province_key].feb_temps(),
                july_temp: province_data[province_key].july_temps(),
                neighbouring_provinces: vec![].into_boxed_slice(),
                tax_bank: 0.0,
            });

            if let Some(owner) = province_data[province_key].owner() {
                let key = country_id_to_org_key.entry(owner).or_insert_with(|| {
                    let new_org = Organization::new(&nation_names[owner], num_provinces);
                    let new_org_key = new_org.key;
                    organizations.entry(new_org_key).or_insert(new_org);
                    new_org_key
                });
                organizations.get_mut(key).unwrap().province_control[province_key] = 1.0;
            }
        }

        let provinces = ProvinceMap(provinces.into_boxed_slice());

        for i in 0..provinces.0.len() {
            let mut index_set: HashSet<_> =
                provinces[ProvinceKey(i)].point_indices.iter().collect();
            let mut neighbours = vec![];
            'neighbour_loop: for j in 0..provinces.0.len() {
                for neighbour_index in &provinces[ProvinceKey(j)].point_indices {
                    if index_set.insert(neighbour_index) {
                        neighbours.push(ProvinceKey(j));
                        continue 'neighbour_loop;
                    }
                }
            }
        }

        for (_, organization) in organizations.iter_mut() {
            let mut new_military = Military::new(MilitaryType::Army, num_provinces);
            for (province_key, &control) in organization.province_control.0.iter().enumerate() {
                let province_key = ProvinceKey(province_key);
                if control > 0.01 {
                    new_military.deployed_forces[province_key] +=
                        rng.f64() * provinces[province_key].pops.population() * 0.01 + 1000.0;
                }
            }
            organization.militaries.push(new_military);
        }
        const PRE_SIM_STEPS: usize = 1000;
        // const PRE_SIM_STEPS: usize = 100;

        let histories = Histories::new(PRE_SIM_STEPS, num_provinces);

        let relations = RelationMap::new(&organizations);
        let num_organizations = organizations.len();

        // for org in organizations.values(){
        //     println!("{}", org.name)
        // }
        let &player_organization = organizations
            .keys()
            .nth(rng.usize(0..num_organizations))
            .unwrap();
        // let player_organization = OrganizationKey(0);
        let mut world = Self {
            points: vertices.iter().map(|vertex| vertex.cast()).collect(),
            provinces,
            global_market: Market {
                price: [1.0; Good::VARIANT_COUNT],
                supply: [0.0; Good::VARIANT_COUNT],
                demand: [0.0; Good::VARIANT_COUNT],
                previous_supply_demand_error: [0.0; Good::VARIANT_COUNT],
                supply_demand_error_integral: [0.0; Good::VARIANT_COUNT],
            },
            histories,
            current_year: SIM_START_YEAR,
            organizations,
            relations,
            // travel_costs,
            selected_province: None,
            targeted_province: None,
            selected_organization: None,
            player_organization,
        };
        // println!("Update travel times");
        // world.update_travel_times();
        // println!("Done Updating travel times");

        //Do some simulation to help the state stabilize
        let mut last_time_check = Instant::now();
        for _ in 0..(PRE_SIM_STEPS / 2) {
            world.process(Self::PRE_SIM_STEP_LENGTH);
            if last_time_check.elapsed().as_secs_f64() > 0.5 {
                last_time_check = Instant::now();
                println!("Pre sim at year: {:}", world.current_year);
            }
        }
        //split pre simulation into two stages. First to do big stabilizations at a higher speed and then a second finer pass at the proper speed
        for _ in 0..(PRE_SIM_STEPS / 2) {
            world.process(Self::STEP_LENGTH);
            if last_time_check.elapsed().as_secs_f64() > 0.5 {
                last_time_check = Instant::now();
                println!("Pre sim at year: {:}", world.current_year);
            }
        }
        println!("Finished creating world");
        world
    }

    pub fn process(&mut self, delta_year: f64) {
        self.process_battles(delta_year);
        //Process market, selling yesterday's produced goods and buying needs for today
        for province in self.provinces.0.iter_mut() {
            //update productivities
            province.industry_data[Industry::Farm as usize].productivity = kernel(
                province.get_current_temp(self.current_year),
                24.0,
                0.1,
                25.0,
            )
            .clamp(0.0, 1.0)
                * 1.0;
            // province.industry_data[Industry::Farm as usize].productivity = ((self.current_year.sin() + 1.0) / 2.0) * 3.0;
            province.intangible_goods[IntangibleGoods::Work as usize] = 0.0; //Work doesn't carry over between ticks
            province.market.demand = [0.0; Good::VARIANT_COUNT];
            province.market.supply = [0.0; Good::VARIANT_COUNT];
            province
                .market
                .process(delta_year, &mut province.pops.pop_slices, 0.5, 1.0);
            for good_index in 0..Good::VARIANT_COUNT {
                let good = Good::try_from(good_index).unwrap();
                province
                    .market
                    .buy_and_sell(good, delta_year, &mut province.pops.pop_slices);
                province
                    .market
                    .update_price(good, delta_year, province.pops.population().max(1.0));
            }
        }
        self.global_market.demand = [0.0; Good::VARIANT_COUNT];
        self.global_market.supply = [0.0; Good::VARIANT_COUNT];
        for province in self.provinces.0.iter_mut() {
            self.global_market.process(
                delta_year,
                &mut province.pops.pop_slices,
                1.0,
                province.trader_cost_multiplier,
            );
        }
        for good_index in 0..Good::VARIANT_COUNT {
            let good = Good::try_from(good_index).unwrap();
            for province in self.provinces.0.iter_mut() {
                self.global_market
                    .buy_and_sell(good, delta_year, &mut province.pops.pop_slices);
            }
            self.global_market
                .update_price(good, delta_year, self.get_total_population());
        }

        let mut global_tax_bank = 0.0;
        let mut global_migration_pool_pops = 0.0;
        let mut global_migration_pool_money = 0.0;
        for province in self.provinces.0.iter_mut() {
            // let mut migration_pool = [0.0; Industry::VARIANT_COUNT];
            let mut migration_pool_pops = 0.0;
            let mut migration_pool_money = 0.0;
            let mut num_unhealthy_slices = 0;
            for (slice_index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                let needs = get_needs(slice.population, delta_year);
                let industry_index = slice_index % Industry::VARIANT_COUNT;
                let minimum_met_needs = needs
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
                slice.minimum_met_needs = minimum_met_needs;
                slice
                    .previous_met_needs
                    .push_back(minimum_met_needs.min(1.0));
                if slice.previous_met_needs.len() > PopSlice::NUM_MET_NEEDS_STORED {
                    slice.previous_met_needs.remove(0);
                }
                for (good_index, &need) in needs.iter().enumerate() {
                    slice.owned_goods[good_index] -= need * minimum_met_needs;
                }
                if minimum_met_needs < 0.9 {
                    num_unhealthy_slices += 1;
                }
            }
            let num_healthy_slices = NUM_SLICES - num_unhealthy_slices;

            //consume daily living needs
            for (slice_index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                let population_change = if slice.minimum_met_needs > 0.9 {
                    slice.population
                        * 0.02
                        * delta_year
                        * map_range_linear_f64(slice.minimum_met_needs, 0.98, 1.0, 0.1, 1.0)
                } else if slice.minimum_met_needs > 0.25 {
                    ((((hash_usize_fast((self.current_year * 365.0 * 24.0) as usize) as f64)
                        / (usize::MAX as f64))
                        * 2.0)
                        - 1.0)
                        * 0.5
                } else {
                    (slice.population * -0.3 * delta_year).min(-100.0 * delta_year)
                }
                .max(-slice.population);
                let cannibalism_amount =
                    (-population_change).max(0.0) / slice.population * TRICKLEBACK_RATIO;
                slice.trickleback += cannibalism_amount;
                let trickleback_amount =
                    (slice.trickleback * 0.5 * delta_year).min(slice.trickleback);
                slice.population += trickleback_amount;
                slice.trickleback -= trickleback_amount;
                assert!(!slice.population.is_nan());
                slice.population += population_change;
                if slice.minimum_met_needs < 0.9 {
                    let migration_pops_amount =
                        (slice.population * 0.1 * delta_year).min(slice.population);
                    let migration_money_amount = if slice.population > 0.0 {
                        (migration_pops_amount / slice.population) * slice.money
                    } else {
                        0.0
                    };
                    const PROVINCE_MIGRATION_RATIO: f64 = 0.5;
                    const GLOBAL_MIGRATION_RATIO: f64 = 1.0 - PROVINCE_MIGRATION_RATIO;

                    migration_pool_pops += migration_pops_amount * PROVINCE_MIGRATION_RATIO;
                    migration_pool_money += migration_money_amount * PROVINCE_MIGRATION_RATIO;

                    global_migration_pool_pops += migration_pops_amount * GLOBAL_MIGRATION_RATIO;
                    global_migration_pool_money += migration_money_amount * GLOBAL_MIGRATION_RATIO;
                    slice.population -= migration_pops_amount;
                    slice.money -= migration_money_amount
                }
            }

            //process provincial migrations
            let mut total_migrations = 0.0;
            if migration_pool_pops > 0.0 {
                if num_healthy_slices > 0 {
                    for slice in province
                        .pops
                        .pop_slices
                        .iter_mut()
                        .filter(|s| s.minimum_met_needs >= 0.9)
                    {
                        let migration_amount = migration_pool_pops / (num_healthy_slices as f64);
                        slice.population += migration_amount;
                        assert!(!migration_pool_money.is_nan());
                        slice.money += migration_pool_money / (num_healthy_slices as f64);
                        total_migrations += migration_amount;
                    }
                } else {
                    for slice in province.pops.pop_slices.iter_mut() {
                        let migration_amount = migration_pool_pops / (NUM_SLICES as f64);
                        slice.population += migration_amount;
                        slice.money += migration_pool_money / (NUM_SLICES as f64);
                        total_migrations += migration_amount;
                    }
                }
            }
            // dbg!(num_unhealthy_slices);
            // dbg!(migration_pool);
            // dbg!(total_migrations);
            assert!((migration_pool_pops - total_migrations).abs() < 1.0);

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
                    slice.owned_goods[good as usize] = slice.owned_goods[good as usize]
                        .min(slice.population * GOOD_STORAGE_RATIO * delta_year)
                }
            }
            // let mut tax_bank = 0.0;
            let mut _population_sum = 0.0;
            for (_index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                let tax_amount = (slice.money * 0.1 * delta_year).min(slice.money);
                let local_tax_proportion = 1.0;
                province.tax_bank += tax_amount * local_tax_proportion;
                global_tax_bank += tax_amount * (1.0 - local_tax_proportion);
                slice.money -= tax_amount;
                if slice.money.is_nan() {
                    dbg!(tax_amount);
                }
                assert!(!slice.money.is_nan());

                _population_sum += slice.population;
            }
            // for (_index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
            //     let tax_payment = tax_bank * (1.0 / NUM_SLICES as f64);
            //     slice.money += tax_payment;
            //     assert!(!slice.money.is_nan());
            // }

            // dbg!(money_sum);
        }
        let num_provinces = self.provinces.0.len();
        for province in self.provinces.0.iter_mut() {
            let migration_pool_pops = global_migration_pool_pops / num_provinces as f64;
            let migration_pool_money = global_migration_pool_money / num_provinces as f64;
            let tax_bank = global_tax_bank / num_provinces as f64;
            for slice in province.pops.pop_slices.iter_mut() {
                let migration_amount = migration_pool_pops / NUM_SLICES as f64;
                slice.population += migration_amount;
                slice.money += migration_pool_money / NUM_SLICES as f64;

                let tax_payment = tax_bank * (1.0 / NUM_SLICES as f64);
                slice.money += tax_payment;
                assert!(!slice.money.is_nan());
            }
        }
        for (province_key, province) in self.provinces.0.iter_mut().enumerate() {
            // let control_sum: f64 = self.organizations.values().flat_map(|o|o.province_control.iter().filter(|(&k, _)| k == province_key)).map(|p|p.1).sum();
            for organization in self.organizations.values_mut() {
                let transfer_amount =
                    province.tax_bank * organization.province_control[ProvinceKey(province_key)];
                organization.money += transfer_amount;
                province.tax_bank -= transfer_amount
            }
        }
        for organization in self.organizations.values_mut() {
            let redistribution_amount =
                (organization.money * 0.5 * delta_year).min(organization.money);
            organization.money -= redistribution_amount;
            let province_control_sum = organization.province_control.0.iter().sum::<f64>();
            for (province_key, province_control) in
                organization.province_control.0.iter().enumerate()
            {
                for slice in &mut self.provinces[ProvinceKey(province_key)].pops.pop_slices {
                    let org_payment = redistribution_amount
                        * (province_control / province_control_sum)
                        / (NUM_SLICES as f64);
                    slice.money += org_payment;
                }
            }
        }
        self.add_new_tick(self.current_year);
        self.current_year += delta_year;
    }

    fn process_battles(&mut self, delta_year: f64) {
        for province_key in 0..self.provinces.0.len() {
            let province_key = ProvinceKey(province_key);
            // self.friendly_combatants.clear();
            // self.enemy_combatants.clear();

            let combat_orgs: Box<[&Organization]> = self
                .organizations
                .values()
                .filter(|o| {
                    o.militaries
                        .iter()
                        .any(|m| m.deployed_forces[province_key] > 0.5)
                })
                .collect();
            let mut combat_orgs_filtered = OrganizationMap::with_capacity_and_hasher(
                combat_orgs.len(),
                BuildNoHashHasher::default(),
            );
            for i in 0..combat_orgs.len() {
                for j in 0..combat_orgs.len() {
                    if self
                        .relations
                        .get_relations(combat_orgs[i].key, combat_orgs[j].key)
                        .at_war
                    {
                        combat_orgs_filtered
                            .entry(combat_orgs[i].key)
                            .or_insert(Vec::with_capacity(combat_orgs.len()))
                            .push(combat_orgs[j].key);
                    }
                }
            }
            let combat_orgs: HashMap<_, _, BuildNoHashHasher<OrganizationKey>> =
                combat_orgs_filtered
                    .iter()
                    .filter(|&(_k, v)| v.len() > 0)
                    .collect();
            if combat_orgs.len() <= 0 {
                continue;
            }

            let first_index = get_battle_seed(
                self.current_year,
                province_key.0 as usize,
                combat_orgs.len(),
            );
            let &first_combatant_key = combat_orgs.keys().nth(first_index).unwrap();

            let second_index = get_battle_seed(
                self.current_year,
                province_key.0 as usize,
                combat_orgs[first_combatant_key].len(),
            );
            let second_combatant_key = combat_orgs[first_combatant_key][second_index];

            // for (org_key, org) in self.organizations{
            //     if org.militaries.iter().any(|m|match m.deployed_forces.get(province_key) {
            //                             Some(&troop_count) => troop_count < 1.0,
            //                             None => true,
            //     }){
            //         continue;
            //     }

            // }

            // for (&friendly_key, friendly_org) in &self.organizations{
            //     if friendly_org.militaries
            //         .iter()
            //         .any(|m| match m.deployed_forces.get(province_key) {
            //             Some(&troop_count) => troop_count < 1.0,
            //             None => true,
            //         }){
            //             continue;
            //         }
            //         self.friendly_combatants.push(friendly_key);
            // }

            // for friendly_key in self
            //     .organizations
            //     .iter()
            //     .filter(|(k, o)| {
            //         o.militaries
            //             .iter()
            //             .any(|m| match m.deployed_forces.get(province_key) {
            //                 Some(&troop_count) => troop_count > 0.5,
            //                 None => false,
            //             })
            //     })
            //     .filter(|(&org_key, _)| {
            //         self.organizations
            //             .iter()
            //             .filter(|(&k, _)| k != org_key)
            //             .any(|(&enemy_key, enemy)| {
            //                 self.relations.get_relations(org_key, enemy_key).at_war
            //                     && enemy.militaries.iter().any(|m| {
            //                         match m.deployed_forces.get(province_key) {
            //                             Some(&troop_count) => troop_count > 0.5,
            //                             None => false,
            //                         }
            //                     })
            //             })
            //     })
            //     .map(|(&k, _)| k)
            // {
            //     self.friendly_combatants.push(friendly_key);
            // }

            // if self.friendly_combatants.len() == 0 {
            //     continue;
            // }
            // let first_index = get_battle_seed(
            //     self.current_year,
            //     province_key.0 as usize,
            //     self.friendly_combatants.len(),
            // );
            // let first_combatant_key = self.friendly_combatants[first_index];

            // for (&enemy_key, enemy_org) in &self.organizations {
            //     if self
            //         .relations
            //         .get_relations(first_combatant_key, enemy_key)
            //         .at_war
            //         && enemy_org.militaries.iter().any(|m| {
            //             match m.deployed_forces.get(province_key) {
            //                 Some(&troop_count) => troop_count > 0.5,
            //                 None => false,
            //             }
            //         })
            //     {
            //         continue;
            //     }
            //     self.enemy_combatants.push(enemy_key)
            // }

            // for enemy_key in self.enemy_combatants = self.organizations
            //         .iter()
            //         .filter(|(&enemy_key, enemy_org)| {
            //             self.relations
            //                 .get_relations(first_combatant_key, enemy_key)
            //                 .at_war
            //                 && enemy_org.militaries.iter().any(|m| {
            //                     match m.deployed_forces.get(province_key) {
            //                         Some(&troop_count) => troop_count > 0.5,
            //                         None => false,
            //                     }
            //                 })
            //         }).map(|(&k,_)|k){
            //             self.enemy_combatants.push(enemy_key);
            //         }

            // if self.enemy_combatants.len() == 0 {
            //     continue;
            // }
            // let second_index = get_battle_seed(
            //     self.current_year,
            //     province_key.0 as usize,
            //     self.enemy_combatants.len(),
            // );
            // let second_combatant_key = self.enemy_combatants[second_index];

            // let (army_ratio, first_combatant_key, second_combatant_key, avg_size) = {
            //     let num_first_orgs = self
            //         .organizations
            //         .iter()
            //         .filter(|(k, o)| {
            //             o.militaries
            //                 .iter()
            //                 .any(|m| match m.deployed_forces.get(province_key) {
            //                     Some(&troop_count) => troop_count > 0.5,
            //                     None => false,
            //                 })
            //         })
            //         .filter(|(&org_key, _)| {
            //             self.organizations
            //                 .iter()
            //                 .filter(|(&k, _)| k != org_key)
            //                 .any(|(&enemy_key, enemy)| {
            //                     self.relations.get_relations(org_key, enemy_key).at_war
            //                         && enemy.militaries.iter().any(|m| {
            //                             match m.deployed_forces.get(province_key) {
            //                                 Some(&troop_count) => troop_count > 0.5,
            //                                 None => false,
            //                             }
            //                         })
            //                 })
            //         })
            //         .count();

            //     if num_first_orgs == 0{
            //         continue;
            //     }
            //     let first_index =
            //         get_battle_seed(self.current_year, province_key.0 as usize, num_first_orgs);
            //     let (first_combatant_key, first_combatant) = self
            //         .organizations
            //         .iter()
            //         .filter(|(k, o)| {
            //             o.militaries
            //                 .iter()
            //                 .any(|m| match m.deployed_forces.get(province_key) {
            //                     Some(&troop_count) => troop_count > 0.5,
            //                     None => false,
            //                 })
            //         })
            //         .filter(|(&org_key, organization)| {
            //             self.organizations.iter().any(|(&enemy_key, enemy)| {
            //                 self.relations.get_relations(org_key, enemy_key).at_war
            //                     && enemy.militaries.iter().any(|m| {
            //                         match m.deployed_forces.get(province_key) {
            //                             Some(&troop_count) => troop_count > 0.5,
            //                             None => false,
            //                         }
            //                     })
            //             })
            //         })
            //         .nth(first_index)
            //         .unwrap();

            //         let num_second_orgs = self.organizations
            //         .iter()
            //         .filter(|(&enemy_key, enemy_org)| {
            //             self.relations
            //                 .get_relations(*first_combatant_key, enemy_key)
            //                 .at_war
            //                 && enemy_org.militaries.iter().any(|m| {
            //                     match m.deployed_forces.get(province_key) {
            //                         Some(&troop_count) => troop_count > 0.5,
            //                         None => false,
            //                     }
            //                 })
            //         })
            //         .count();

            //     if num_second_orgs == 0{
            //         continue;
            //     }
            //     let second_index = get_battle_seed(
            //         self.current_year,
            //         province_key.0 as usize,
            //         num_second_orgs,
            //     );
            //     let (second_combatant_key, second_combatant) = self
            //         .organizations
            //         .iter()
            //         .filter(|(&enemy_key, enemy_org)| {
            //             self.relations
            //                 .get_relations(*first_combatant_key, enemy_key)
            //                 .at_war
            //                 && enemy_org.militaries.iter().any(|m| {
            //                     match m.deployed_forces.get(province_key) {
            //                         Some(&troop_count) => troop_count > 0.5,
            //                         None => false,
            //                     }
            //                 })
            //         })
            //         .nth(second_index)
            //         .unwrap();

            //     // println!("Fight between {:} and {:}", first_index, second_index);
            // };

            let first_army_size: f64 = self.organizations[&first_combatant_key]
                .militaries
                .iter()
                .map(|m| m.deployed_forces[province_key])
                .sum();
            let second_army_size: f64 = self.organizations[&second_combatant_key]
                .militaries
                .iter()
                .map(|m| m.deployed_forces[province_key])
                .sum();
            let avg_size = (first_army_size + second_army_size) / 2.0;

            let army_ratio = if first_army_size > second_army_size {
                first_army_size / second_army_size
            } else {
                second_army_size / first_army_size
            };

            for first_army in &mut self
                .organizations
                .get_mut(&first_combatant_key)
                .unwrap()
                .militaries
            {
                let losses = avg_size * (1.0 / army_ratio) * delta_year;
                first_army.deployed_forces[province_key] -=
                    losses.min(first_army.deployed_forces[province_key]);
            }

            for second_army in &mut self
                .organizations
                .get_mut(&second_combatant_key)
                .unwrap()
                .militaries
            {
                let losses = avg_size * army_ratio * delta_year;
                second_army.deployed_forces[province_key] -=
                    losses.min(second_army.deployed_forces[province_key]);
            }
        }
    }

    fn add_new_tick(&mut self, time: f64) {
        //TODO refactor so that adding new graphs is easier
        const HISTORY_TICKS: usize = 1000;
        for (key, province) in self.provinces.0.iter().enumerate() {
            let key = ProvinceKey(key);
            if self.histories.population[key].len() > HISTORY_TICKS {
                self.histories.population[key].pop_back();
            }
            self.histories.population[key].push_front((time, province.pops.population()));

            for good_index in 0..Good::VARIANT_COUNT {
                if self.histories.prices[key][good_index].len() > HISTORY_TICKS {
                    self.histories.prices[key][good_index].pop_back();
                }
                self.histories.prices[key][good_index]
                    .push_front((time, province.market.price[good_index]));
                //supply
                if self.histories.supply[key][good_index].len() > HISTORY_TICKS {
                    self.histories.supply[key][good_index].pop_back();
                }
                self.histories.supply[key][good_index]
                    .push_front((time, province.market.supply[good_index]));
                //demand
                if self.histories.demand[key][good_index].len() > HISTORY_TICKS {
                    self.histories.demand[key][good_index].pop_back();
                }
                self.histories.demand[key][good_index]
                    .push_front((time, province.market.demand[good_index]));
            }
        }
        for good_index in 0..Good::VARIANT_COUNT {
            if self.histories.global_prices[good_index].len() > HISTORY_TICKS {
                self.histories.global_prices[good_index].pop_back();
            }
            self.histories.global_prices[good_index]
                .push_front((time as f64, self.global_market.price[good_index]));
            //supply
            if self.histories.global_supply[good_index].len() > HISTORY_TICKS {
                self.histories.global_supply[good_index].pop_back();
            }
            self.histories.global_supply[good_index]
                .push_front((time as f64, self.global_market.supply[good_index]));
            //demand
            if self.histories.global_demand[good_index].len() > HISTORY_TICKS {
                self.histories.global_demand[good_index].pop_back();
            }
            self.histories.global_demand[good_index]
                .push_front((time as f64, self.global_market.demand[good_index]));
        }
    }
}
