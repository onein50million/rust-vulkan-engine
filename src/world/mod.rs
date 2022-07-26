use std::{
    cell::UnsafeCell,
    collections::{HashMap, HashSet, VecDeque},
    f64::consts::PI,
    fmt::Display,
    hash::Hash,
    ops::{Index, IndexMut},
    path::PathBuf,
    time::Instant,
};

pub mod ideology;
pub mod language;
pub mod organization;
mod party_names;
pub mod questions;

use float_ord::FloatOrd;
use nalgebra::Vector3;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use variant_count::VariantCount;

use crate::{
    support::{hash_usize_fast, map_range_linear_f64},
    world::{
        ideology::{
            positions::{
                Assimilation, ExportDifficulty, Immigration, ImportDifficulty, Irredentism,
                LanguageSupport, Research, SeparationOfPowers, TaxMethod, TaxRate, WarSupport,
                WelfareSupport,
            },
            Ideology,
        },
        language::{Language, LanguageManager, LanguageMap},
        party_names::PARTY_NAMES,
    },
};

use self::{
    ideology::{Beliefs, PoliticalParty, PoliticalPartyMap},
    organization::{
        Branch, BranchControlDecision, BranchControlFlag, BranchMap, Decision, DecisionCategory,
        DecisionControlFlag, Organization, OrganizationKey, Relation,
    },
};

const BASE_INDUSTRY_SIZE: f64 = 1000.0;
const TRICKLEBACK_RATIO: f64 = 0.25;

#[repr(usize)]
#[derive(Clone, Copy, VariantCount, Debug, IntoPrimitive, TryFromPrimitive)]
pub enum Culture {
    CultureA,
    // CultureB,
    // CultureC,
    // D,
    // E,
    // F,
    // G,
    // H,
    // J,
    // K,
    // L,
    // M,
    // N,
    // O,
    // P,
    // Q,
    // R,
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
    pub beliefs: Beliefs,
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
    pub pop_slices: Box<[PopSlice]>,
    // pub pop_slices: [PopSlice; NUM_SLICES],
}

impl Pops {
    pub fn new(total_population: f64) -> Self {
        // const SLICE_COUNT: usize = NUM_SLICES;
        // unsafe {
        //     let mut pop_slices: MaybeUninit<[PopSlice; SLICE_COUNT]> = MaybeUninit::uninit();
        //     let rng = fastrand::Rng::new();
        //     for i in 0..SLICE_COUNT {
        //         let slice_population = (total_population / SLICE_COUNT as f64) * (rng.f64() + 0.5);
        //         (pop_slices.as_mut_ptr() as *mut PopSlice)
        //             .offset(i as isize)
        //             .write(PopSlice {
        //                 population: slice_population,
        //                 money: 10.0 * slice_population,
        //                 // owned_goods: [1_000_000_000_000.0 / SLICE_COUNT as f64; Good::VARIANT_COUNT],
        //                 owned_goods: [0.0 * slice_population as f64; Good::VARIANT_COUNT],
        //                 // met_inputs: [0.0; Good::VARIANT_COUNT],
        //                 individual_demand: [0.0; Good::VARIANT_COUNT],
        //                 individual_supply: [0.0; Good::VARIANT_COUNT],
        //                 previous_met_needs: VecDeque::with_capacity(PopSlice::NUM_MET_NEEDS_STORED),
        //                 minimum_met_needs: f64::NAN,
        //                 trickleback: 0.0,
        //             })
        //     }
        //     let pop_slices = pop_slices.assume_init();
        //     Self { pop_slices }
        // }

        let mut out = Vec::with_capacity(NUM_SLICES);
        let rng = fastrand::Rng::new();
        for _ in 0..NUM_SLICES {
            let slice_population = (total_population / NUM_SLICES as f64) * (rng.f64() + 0.5);
            out.push(PopSlice {
                population: slice_population,
                money: 10.0 * slice_population,
                // owned_goods: [1_000_000_000_000.0 / SLICE_COUNT as f64; Good::VARIANT_COUNT],
                owned_goods: [0.0 * slice_population as f64; Good::VARIANT_COUNT],
                // met_inputs: [0.0; Good::VARIANT_COUNT],
                individual_demand: [0.0; Good::VARIANT_COUNT],
                individual_supply: [0.0; Good::VARIANT_COUNT],
                previous_met_needs: VecDeque::with_capacity(PopSlice::NUM_MET_NEEDS_STORED),
                minimum_met_needs: f64::NAN,
                trickleback: 0.0,
                beliefs: Beliefs::new_random(),
            })
        }

        Self {
            pop_slices: out.into_boxed_slice(),
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
    pub province_area: f64,
    pub pops: Pops,
    pub tax_rate: f64,
    pub market: Market,
    pub position: Vector3<f64>,
    pub industry_data: [IndustryData; Industry::VARIANT_COUNT],
    pub intangible_goods: [f64; IntangibleGoods::VARIANT_COUNT],
    pub feb_temp: f64,
    pub july_temp: f64,
    pub trader_cost_multiplier: f64, //Excess goes to traders in global market. Abstraction for difficulties in exporting from remote or blockaded provinces, WIP and not really functional
    pub tax_bank: f64,
    pub troop_bank: f64,
    // pub recruit_limiter: f64, //ratio of troops that are currently recruited from the province
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
        let mix_factor = ((current_year * PI * 2.0).sin() + 1.0) / 2.0;
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
    languages: Vec<u32>,
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
            languages: vec![],
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
        language: u32,
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
        self.languages.push(language);
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
        for owner in &self.organization_owner {
            if let Some(owner) = owner {
                *occurences.entry(owner).or_insert(0usize) += 1;
            }
        }
        occurences.iter().max_by_key(|o| o.1).map(|(&&id, _)| id)
    }
    pub fn language(&self) -> HashMap<u32, usize> {
        let mut out = HashMap::new();
        for &language in &self.languages {
            *out.entry(language).or_insert(0) += 1;
        }
        out
    }
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct ProvinceKey(pub usize);

// pub type OrganizationMap<T> = HashMap<OrganizationKey, T, BuildNoHashHasher<OrganizationKey>>;
pub type OrganizationMap2D<T> = HashMap<(OrganizationKey, OrganizationKey), T>;

#[derive(Serialize, Deserialize)]
pub struct OrganizationMap<T>(pub Vec<T>);

impl<T> Index<OrganizationKey> for OrganizationMap<T> {
    type Output = T;

    fn index(&self, index: OrganizationKey) -> &Self::Output {
        &self.0[index.0]
    }
}

impl<T> IndexMut<OrganizationKey> for OrganizationMap<T> {
    fn index_mut(&mut self, index: OrganizationKey) -> &mut Self::Output {
        &mut self.0[index.0]
    }
}

impl<T> OrganizationMap<T> {
    pub fn into_iter(self) -> impl Iterator<Item = (OrganizationKey, T)> {
        self.0
            .into_iter()
            .enumerate()
            .map(|(key, org)| (OrganizationKey(key), org))
    }
    pub fn iter(&self) -> impl Iterator<Item = (OrganizationKey, &T)> {
        self.0
            .iter()
            .enumerate()
            .map(|(key, org)| (OrganizationKey(key), org))
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (OrganizationKey, &mut T)> {
        self.0
            .iter_mut()
            .enumerate()
            .map(|(key, org)| (OrganizationKey(key), org))
    }
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }
}

#[derive(Serialize, Deserialize, Debug)]
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

impl<T> ProvinceMap<T> {
    // fn into_iter(self) -> impl Iterator<Item = (ProvinceKey, T)>{
    //     self.0.into_iter().enumerate().map(|(key, org)| (ProvinceKey(key), org))
    // }
    pub fn iter(&self) -> impl Iterator<Item = (ProvinceKey, &T)> {
        self.0
            .iter()
            .enumerate()
            .map(|(key, org)| (ProvinceKey(key), org))
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (ProvinceKey, &mut T)> {
        self.0
            .iter_mut()
            .enumerate()
            .map(|(key, org)| (ProvinceKey(key), org))
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
    pub fn new(organizations: &[Organization]) -> Self {
        let mut relations = OrganizationMap2D::with_capacity(organizations.len());
        for org_a_key in 0..organizations.len() {
            let org_a_key = OrganizationKey(org_a_key);
            for org_b_key in 0..organizations.len() {
                let org_b_key = OrganizationKey(org_b_key);
                let (first_org_key, second_org_key) = if org_a_key.0 > org_b_key.0 {
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
    pub language_manager: LanguageManager,
    pub branches: BranchMap<Branch>,
    pub political_parties: PoliticalPartyMap<PoliticalParty>,
    available_party_names: Vec<String>,
    branch_control_decisions: BranchMap<BranchControlDecision>,
}
impl World {
    pub const RADIUS: f64 = 6_378_137.0; //in meters
    pub const STEP_LENGTH: f64 = 1.0 / 365.0 / 24.0;
    pub const PRE_SIM_STEP_LENGTH: f64 = 0.5;

    // fn get_other_orgs(&self, org_key: OrganizationKey) -> impl Iterator<Item = &Organization> {
    //     self.organizations.values().filter(move |o| o.key != org_key)
    // }
    // fn get_other_orgs_mut(&mut self, org_key: OrganizationKey) -> impl Iterator<Item = &mut Organization> {
    //     self.organizations.values_mut().filter(move |o| o.key != org_key)
    // }

    pub fn process_command(&mut self, command: &str) -> String {
        let mut words = command.split_whitespace();
        match words.next() {
            Some(word) => match word {
                "get_branch_ideology" => match self.selected_organization {
                    Some(organization) => match words.next() {
                        Some(branch_idx) => match branch_idx.parse::<usize>() {
                            Ok(branch_idx) => {
                                match self.organizations[organization].branches.get(branch_idx) {
                                    Some(branch) => {
                                        return format!(
                                            "{:?}\n",
                                            self.branches[*branch].controlling_party
                                        )
                                    }
                                    None => return "Branch out of bounds\n".to_string(),
                                }
                            }
                            Err(_) => return "could not parse branch_idx\n".to_string(),
                        },
                        None => return "Missing argument\n".to_string(),
                    },
                    None => return "Must select an organization\n".to_string(),
                },
                "get_neighbours" => match self.selected_province {
                    Some(province) => {
                        return format!("{:?}", self.provinces[province].neighbouring_provinces)
                    }
                    None => return "Must select a province\n".to_string(),
                },
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
                                for pop in self.provinces[province].pops.pop_slices.iter_mut() {
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
                                for pop in self.provinces[province].pops.pop_slices.iter_mut() {
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
                    for (key, org) in self.organizations.0.iter().enumerate() {
                        let key = OrganizationKey(key);
                        out_string.push_str(&format!("Key: {:}, org: {:}\n", key.0, org.name));
                    }
                    return out_string;
                }
                "add_troops" => {
                    if let (Some(selected_province_key), Some(selected_org_key)) =
                        (self.selected_province, self.selected_organization)
                    {
                        if let Ok(amount) = words
                            .next()
                            .unwrap_or_else(|| return "No amount provided\n")
                            .parse::<f64>()
                        {
                            self.organizations[selected_org_key]
                                .military
                                .deployed_forces[selected_province_key] += amount;
                            return format!(
                                "Added {amount} troops in province {:} for organization {:}\n",
                                selected_province_key.0, selected_org_key.0
                            );
                        } else {
                            return "Invalid amount provided\n".to_string();
                        }
                    } else {
                        return "You must have a province and organization selected\n".to_string();
                    }
                }
                // "add_mil" => match self.selected_organization {
                //     Some(org_key) => match words.next() {
                //         Some(military_type) => {
                //             let military_type = match military_type.to_lowercase().as_str() {
                //                 "army" => MilitaryType::Army,
                //                 "enforcer" => MilitaryType::Enforcer,
                //                 _ => return "Invalid military type\n".to_string(),
                //             };
                //             self.organizations
                //                 .get_mut(&org_key)
                //                 .unwrap()
                //                 .military
                //                 .push(Military::new(military_type, self.provinces.0.len()));
                //             return "Succesfully created a military\n".to_string();
                //         }
                //         None => return "You must enter a type\n".to_string(),
                //     },
                //     None => return "Must select an organization\n".to_string(),
                // },
                "declare_war_all" => match self.selected_organization {
                    Some(org_key) => {
                        for enemy_key in self.organizations.0.iter().enumerate().map(|a| a.0) {
                            let enemy_key = OrganizationKey(enemy_key);
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
        nation_names_and_definitions: Box<[(String, Option<String>)]>,
        color_to_language_name: HashMap<u32, String>,
    ) -> Self {
        println!("Creating world");

        let num_provinces = province_indices.0.len();
        let mut languages: HashMap<u32, Language> = color_to_language_name
            .into_iter()
            .map(|(color, name)| {
                (
                    color,
                    Language {
                        name,
                        province_presence: ProvinceMap(vec![0.0; num_provinces].into_boxed_slice()),
                    },
                )
            })
            .collect();
        let mut branches = BranchMap::new();
        let mut organizations = OrganizationMap(
            nation_names_and_definitions
                .into_vec()
                .into_iter()
                .map(|(name, definition)| {
                    Organization::new(name, num_provinces, &mut branches, definition)
                })
                .collect::<Vec<_>>(),
        );
        let mut provinces = vec![];

        // let total_area = province_data
        //     .0
        //     .iter()
        //     .map(|d| d.num_samples as f64)
        //     .sum::<f64>();
        // let mut country_id_to_org_key: HashMap<usize, OrganizationKey, BuildNoHashHasher<usize>> =
        //     HashMap::with_hasher(BuildNoHashHasher::default());
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
                productivity: 1.0,
                size: f64::NAN,
            }; Industry::VARIANT_COUNT];
            let province_area = province_data[province_key].num_samples as f64;
            for (color, language_count) in province_data[province_key]
                .language()
                .into_iter()
                .filter(|&(color, _)| color != 0)
            {
                let ratio = language_count as f64 / province_area;
                match languages.get_mut(&color) {
                    Some(language) => language.province_presence[province_key] = ratio,
                    None => eprintln!("Invalid color in language map: {:}", color),
                };
            }
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
                province_area,
                intangible_goods: [0.0; IntangibleGoods::VARIANT_COUNT],
                trader_cost_multiplier: 1.0,
                feb_temp: province_data[province_key].feb_temps(),
                july_temp: province_data[province_key].july_temps(),
                neighbouring_provinces: vec![].into_boxed_slice(),
                tax_bank: 0.0,
                tax_rate: 0.1,
                troop_bank: 0.0,
                // recruit_limiter: 0.0,
            });

            if let Some(owner) = province_data[province_key].owner() {
                // let key = country_id_to_org_key.entry(owner).or_insert_with(|| {
                //     let new_org = Organization::new(&nation_names[owner], num_provinces);
                //     let new_org_key = new_org.key;
                //     organizations.entry(new_org_key).or_insert(new_org);
                //     new_org_key
                // });
                organizations[OrganizationKey(owner)].province_control[province_key] = 1.0;
            }
        }

        let mut provinces = ProvinceMap(provinces.into_boxed_slice());

        // for i in 0..provinces.0.len() {
        //     let mut index_set: HashSet<_> =
        //         provinces[ProvinceKey(i)].point_indices.iter().collect();
        //     let mut neighbours = vec![];
        //     'neighbour_loop: for j in 0..provinces.0.len() {
        //         for neighbour_index in &provinces[ProvinceKey(j)].point_indices {
        //             if index_set.insert(neighbour_index) {
        //                 neighbours.push(ProvinceKey(j));
        //                 continue 'neighbour_loop;
        //             }
        //         }
        //     }
        // }
        const NUM_NEIGHBOURS: usize = 5;
        for i in 0..provinces.0.len() {
            let province_key = ProvinceKey(i);
            let mut closest_provinces: Box<[_]> = provinces
                .0
                .iter()
                .enumerate()
                .map(|(k, p)| {
                    (
                        ProvinceKey(k),
                        p.position
                            .normalize()
                            .dot(&provinces[province_key].position.normalize())
                            .acos(),
                    )
                })
                .collect();
            closest_provinces.sort_by_key(|p| FloatOrd(p.1));
            provinces[province_key].neighbouring_provinces = closest_provinces[1..NUM_NEIGHBOURS]
                .iter()
                .map(|p| p.0)
                .collect();
        }
        // for (_, organization) in organizations.iter_mut() {
        //     let mut new_military = Military::new(MilitaryType::Army, num_provinces);
        //     for (province_key, &control) in organization.province_control.0.iter().enumerate() {
        //         let province_key = ProvinceKey(province_key);
        //         if control > 0.01 {
        //             new_military.deployed_forces[province_key] +=
        //                 rng.f64() * provinces[province_key].pops.population() * 0.01 + 1000.0;
        //         }
        //     }
        //     organization.military.push(new_military);
        // }
        // const PRE_SIM_STEPS: usize = 1000;
        const PRE_SIM_STEPS: usize = 100;

        let histories = Histories::new(PRE_SIM_STEPS, num_provinces);

        let relations = RelationMap::new(&organizations.0);
        let num_organizations = organizations.0.len();

        // for org in organizations.values(){
        //     println!("{}", org.name)
        // }
        let player_organization = OrganizationKey(rng.usize(0..num_organizations));
        // let player_organization = OrganizationKey(0);

        let mut num_uncontrolled_provinces = 0;
        for (province_key, province) in provinces.iter_mut() {
            if (1.0
                - organizations
                    .0
                    .iter()
                    .map(|o| o.province_control[province_key])
                    .sum::<f64>())
            .abs()
                > 0.1
            {
                // dbg!(&province.name);
                // let mut nearest_owner = None;
                // let mut current_province = province_key
                // let mut visited = HashSet::new();
                // loop{
                //     visited.insert(current_province);
                //     for &neighbour in provinces[current_province].neighbouring_provinces.iter(){
                //         if provinces[neighbour].
                //     }
                // }
                let num_orgs = organizations.0.len();
                num_uncontrolled_provinces += 1;

                for org in organizations.0.iter_mut() {
                    org.province_control[province_key] = 1.0 / num_orgs as f64;
                }
            }
        }
        dbg!(num_uncontrolled_provinces);

        let languages = LanguageMap(languages.into_values().collect());
        let mut political_parties = PoliticalPartyMap::new();
        for (org_key, organization) in organizations.iter() {
            // let majority_language = languages.iter().max_by_key(|(_, l)|FloatOrd(l.province_presence[province_key])).unwrap().0;
            let mut language_count = LanguageMap(vec![0usize; languages.0.len()]);
            for (province_key, &control) in organization.province_control.iter() {
                if control > 0.1 {
                    for (language_key, language) in languages.iter() {
                        if language.province_presence[province_key] > 0.01 {
                            language_count[language_key] += 1;
                        }
                    }
                }
            }
            let majority_language = language_count.iter().max_by_key(|l| *l.1).unwrap().0;
            political_parties.0.push(PoliticalParty {
                name: format!("{:} Egalitarian Party", organization.name),
                ideology: Ideology {
                    tax_rate: TaxRate::High,
                    tax_method: TaxMethod::Progressive,
                    import_difficulty: ImportDifficulty::Trivial,
                    export_difficulty: ExportDifficulty::Trivial,
                    immigration: Immigration::Unlimited,
                    assimilation: Assimilation::None,
                    war_support: WarSupport::Weak,
                    irredentism: Irredentism::None,
                    language_support: LanguageSupport::Equality,
                    welfare_support: WelfareSupport::FullNeeds,
                    separation_of_powers: SeparationOfPowers::ManyBranches,
                    research: Research::Priority,
                    primary_language: majority_language,
                },
                home_org: org_key,
            });
            political_parties.0.push(PoliticalParty {
                name: format!("{:} Fascist Party", organization.name),
                ideology: Ideology {
                    tax_rate: TaxRate::High,
                    tax_method: TaxMethod::Flat,
                    import_difficulty: ImportDifficulty::Inconvenient,
                    export_difficulty: ExportDifficulty::Diffiicult,
                    immigration: Immigration::None,
                    assimilation: Assimilation::Genocide,
                    war_support: WarSupport::Jingoistic,
                    irredentism: Irredentism::Lebensraum,
                    language_support: LanguageSupport::SingleSuperior,
                    welfare_support: WelfareSupport::FullNeeds,
                    separation_of_powers: SeparationOfPowers::OneBranch,
                    research: Research::Technocratic,
                    primary_language: majority_language,
                },
                home_org: org_key,
            });
            political_parties.0.push(PoliticalParty {
                name: format!("{:} Conservative Party", organization.name),
                ideology: Ideology {
                    tax_rate: TaxRate::Low,
                    tax_method: TaxMethod::Flat,
                    import_difficulty: ImportDifficulty::Inconvenient,
                    export_difficulty: ExportDifficulty::Inconvenient,
                    immigration: Immigration::SkilledLabor,
                    assimilation: Assimilation::Encouraged,
                    war_support: WarSupport::Medium,
                    irredentism: Irredentism::None,
                    language_support: LanguageSupport::SingleSuperior,
                    welfare_support: WelfareSupport::None,
                    separation_of_powers: SeparationOfPowers::OneBranch,
                    research: Research::Priority,
                    primary_language: majority_language,
                },
                home_org: org_key,
            });
        }
        // political_parties.0.push(PoliticalParty{
        //     name: "Default party".to_string(),
        //     ideology: Ideology::new(),
        // });

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
            language_manager: LanguageManager { languages },
            branch_control_decisions: BranchMap(vec![
                BranchControlDecision::None;
                branches.0.len()
            ]),
            branches,
            political_parties,
            available_party_names: PARTY_NAMES.into_iter().map(|s| s.to_string()).collect(),
        };
        // println!("Update travel times");
        // world.update_travel_times();
        // println!("Done Updating travel times");

        //Do some simulation to help the state stabilize
        println!("Starting pre sim");
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
        let rng = fastrand::Rng::new();
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
                        // .min(slice.population * GOOD_STORAGE_RATIO * delta_year)
                        .min(slice.population * GOOD_STORAGE_RATIO)
                }
            }
            //collect taxes and troops
            // let max_recruit_ratio = 0.01;
            let recruit_speed = 0.01;
            let recruit_limiter_gain_speed = 1.0;
            let recruit_limiter_decay_speed = recruit_limiter_gain_speed * 0.1;

            for (_index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                let tax_amount = (slice.money * province.tax_rate * delta_year).min(slice.money);
                let local_tax_proportion = 1.0;
                province.tax_bank += tax_amount * local_tax_proportion;
                global_tax_bank += tax_amount * (1.0 - local_tax_proportion);
                slice.money -= tax_amount;

                // let recruit_ratio = (1.0 - province.recruit_limiter).max(0.0) * recruit_speed * delta_year;
                // let recruit_amount = (slice.population * recruit_ratio).min(slice.population);
                // slice.population -= recruit_amount;
                // province.troop_bank += recruit_amount;
                let recruit_ratio = recruit_speed * delta_year;
                let recruit_amount = (slice.population * recruit_ratio).min(slice.population);
                slice.population -= recruit_amount;
                province.troop_bank += recruit_amount;

                if slice.money.is_nan() {
                    dbg!(tax_amount);
                }
                assert!(!slice.money.is_nan());
            }
            // province.recruit_limiter -= recruit_limiter_decay_speed * delta_year;
            // province.recruit_limiter += (recruit_limiter_gain_speed * delta_year);
            // province.recruit_limiter = province.recruit_limiter.clamp(0.0, 1.0);
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
        //distribute taxes and troops to controlling organizations
        for (province_key, province) in self.provinces.0.iter_mut().enumerate() {
            let province_key = ProvinceKey(province_key);
            for organization in self.organizations.0.iter_mut() {
                let transfer_amount =
                    province.tax_bank * organization.province_control[province_key];
                organization.money += transfer_amount;
                // province.tax_bank -= transfer_amount;

                let troop_transfer_amount =
                    province.troop_bank * organization.province_control[province_key];
                organization.military.deployed_forces[province_key] += troop_transfer_amount;
            }
            province.troop_bank = 0.0;
            province.tax_bank = 0.0;
        }

        // let num_provinces = self.provinces.0.len();
        // let province_key = ProvinceKey(rng.usize(..num_provinces));
        // let province = &self.provinces[province_key];
        // for slice in province.pops.pop_slices.iter(){
        //     let majority_language = self.language_manager.languages.iter().max_by_key(|(_, l)|FloatOrd(l.province_presence[province_key])).unwrap().0;
        //     if self.political_parties.iter().map(|a|FloatOrd(slice.beliefs.to_ideology(majority_language).distance(&a.1.ideology))).min().unwrap().0 > 2.0{
        //         self.political_parties.0.push(
        //             PoliticalParty{
        //                 name: self.available_party_names.pop().unwrap_or(String::from("OUT OF NAMES")),
        //                 ideology: slice.beliefs.to_ideology(majority_language),
        //             }
        //         )
        //     }
        // }

        self.process_organizations(delta_year);
        self.add_new_tick(self.current_year);
        self.current_year += delta_year;
    }

    fn process_organizations(&mut self, delta_year: f64) {
        let rng = fastrand::Rng::new();
        // let org_keys: Box<_> = self.organizations.0.iter().enumerate().map(|a|a.0).collect();
        let num_orgs = self.organizations.0.len();
        for organization_key in 0..self.organizations.0.len() {
            let organization_key = OrganizationKey(organization_key);

            //decision phase

            let organization = &mut self.organizations[organization_key];
            for &branch_key in &organization.branches {
                {
                    let branch = &mut self.branches[branch_key];

                    //elections
                    if branch.elected
                        && (self.current_year - branch.last_election) > branch.term_length
                    {
                        branch.last_election = self.current_year;

                        let mut party_votes =
                            PoliticalPartyMap(vec![0.0; self.political_parties.len()]);

                        for (controlled_province, _) in
                            organization.province_control.iter().filter(|a| *a.1 > 0.1)
                        {
                            let province = &self.provinces[controlled_province];
                            let majority_language = self
                                .language_manager
                                .languages
                                .iter()
                                .max_by_key(|(_, l)| {
                                    FloatOrd(l.province_presence[controlled_province])
                                })
                                .unwrap()
                                .0;
                            for (party_key, party) in self.political_parties.iter() {
                                for slice in province.pops.pop_slices.iter() {
                                    let slice_ideology =
                                        slice.beliefs.to_ideology(majority_language);

                                    let ideology_difference =
                                        slice_ideology.distance(&party.ideology) + if party.home_org != organization_key{
                                            1.0
                                        }else{
                                            0.0
                                        };

                                    let variance =
                                        map_range_linear_f64(rng.f64(), 0.0, 1.0, 0.9, 1.1);
                                    party_votes[party_key] +=
                                        (slice.population / (ideology_difference + 1.0)) * variance;
                                    // party_votes[party_key] += rng.f64();
                                }
                            }
                        }
                        // branch.controlling_party = self.political_parties[party_votes.iter().max_by_key(|v|FloatOrd(*v.1)).unwrap().0].ideology.clone();
                        // if organization_key == self.player_organization{
                        //     println!("Election in your organization for branch {:}", branch.name);
                        //     dbg!(branch.controlling_party);
                        // }
                        branch.controlling_party =
                            party_votes.iter().max_by_key(|v| FloatOrd(*v.1)).unwrap().0;
                        // if organization_key == self.player_organization{
                        //     dbg!(branch.controlling_party);
                        // }
                    }
                }
                //branch appointmnets
                let branches_ptr = { self.branches.0.as_mut_ptr().clone() };
                for (&controlled_branch, branch_control_flag) in
                    &self.branches[branch_key].branch_control
                {
                    if branch_control_flag.contains(BranchControlFlag::APPOINT)
                        && (self.current_year - self.branches[controlled_branch].last_election)
                            > self.branches[controlled_branch].term_length
                    {
                        //Borrowing from a vec twice is safe as long as the indices don't overlap
                        unsafe {
                            assert_ne!(controlled_branch, branch_key);
                            assert!(controlled_branch.0 < self.branches.0.len());
                            (*branches_ptr.offset(controlled_branch.0 as isize)).last_election =
                                self.current_year;
                        }
                        if matches!(
                            self.branch_control_decisions[controlled_branch],
                            BranchControlDecision::None
                        ) {
                            self.branch_control_decisions[controlled_branch] =
                                BranchControlDecision::Appoint(
                                    self.branches[branch_key].controlling_party,
                                );
                        }
                    }
                }
                //branch approvals
                for (&controlled_branch, branch_control_flag) in
                    &self.branches[branch_key].branch_control
                {
                    if branch_control_flag.contains(BranchControlFlag::APPROVE) {
                        match self.branch_control_decisions[controlled_branch] {
                            BranchControlDecision::None | BranchControlDecision::Vetoed => {}
                            BranchControlDecision::Appoint(appointed_party) => {
                                if self.political_parties[appointed_party].ideology.distance(
                                    &self.political_parties
                                        [self.branches[branch_key].controlling_party]
                                        .ideology,
                                ) < 2.0
                                {
                                    self.branch_control_decisions[controlled_branch] =
                                        BranchControlDecision::Vetoed;
                                }
                            }
                        }
                    }
                }

                for (decision, decision_control) in self.branches[branch_key]
                    .decision_control
                    .iter()
                    .enumerate()
                {
                    let decision = DecisionCategory::try_from(decision).unwrap();
                    if decision_control.contains(DecisionControlFlag::ENACT) {
                        if matches!(organization.decisions[decision as usize], Decision::None) {
                            let controlling_ideology = &self.political_parties
                                [self.branches[branch_key].controlling_party]
                                .ideology;

                            organization.decisions[decision as usize] = match decision {
                                DecisionCategory::SetTaxes => {
                                    let num_controlled_provinces = organization
                                        .province_control
                                        .iter()
                                        .filter(|&(_, &p)| p > 0.1)
                                        .count();
                                    let new_tax_rate = match controlling_ideology.tax_rate {
                                        TaxRate::Low => 0.05,
                                        TaxRate::Medium => 0.2,
                                        TaxRate::High => 0.5,
                                    };
                                    if num_controlled_provinces > 0 {
                                        Decision::SetTaxes(
                                            organization
                                                .province_control
                                                .iter()
                                                .filter(|&(_, &p)| p > 0.1)
                                                .nth(rng.usize(..num_controlled_provinces))
                                                .unwrap()
                                                .0,
                                            new_tax_rate,
                                        )
                                    } else {
                                        Decision::None
                                    }
                                }
                                DecisionCategory::DeclareWar => {
                                    let war_chance = match controlling_ideology.war_support {
                                        WarSupport::Pacifistic => 0.0,
                                        WarSupport::Weak => 0.01,
                                        WarSupport::Medium => 0.1,
                                        WarSupport::Strong => 1.0,
                                        WarSupport::Jingoistic => 10.0,
                                    } * delta_year;
                                    if rng.f64() < war_chance {
                                        Decision::DeclareWar(OrganizationKey(rng.usize(..num_orgs)))
                                    } else {
                                        Decision::None
                                    }
                                }
                                DecisionCategory::MoveTroops => {
                                    let source_count = organization
                                        .province_control
                                        .iter()
                                        .filter(|&(i, &c)| c > 0.9)
                                        .count();

                                    let dest_count = organization
                                        .province_control
                                        .iter()
                                        .filter(|&(i, &c)| c > 0.1)
                                        .flat_map(|(key, _)| {
                                            self.provinces[key].neighbouring_provinces.iter()
                                        })
                                        .filter(|&&k| organization.province_control[k] < 0.9)
                                        .count();
                                    if source_count > 0
                                        && dest_count > 0
                                        && rng.f64() < 1.0 * delta_year
                                    {
                                        let source = organization
                                            .province_control
                                            .iter()
                                            .filter(|&(i, &c)| c > 0.9)
                                            .nth(rng.usize(..source_count))
                                            .unwrap()
                                            .0;
                                        let &dest = organization
                                            .province_control
                                            .iter()
                                            .filter(|&(i, &c)| c > 0.1)
                                            .flat_map(|(key, _)| {
                                                self.provinces[key].neighbouring_provinces.iter()
                                            })
                                            .filter(|&&k| organization.province_control[k] < 0.9)
                                            .nth(rng.usize(..dest_count))
                                            .unwrap();
                                        Decision::MoveTroops(source, dest, 1.0)
                                    } else {
                                        Decision::None
                                    }
                                }
                            }
                        }
                    }
                }
            }
            //veto/approval phase
            for decision in &mut organization.decisions {
                if !matches!(decision, Decision::None) && rng.f64() < 0.1 * delta_year {
                    *decision = Decision::Vetoed;
                }
            }
            for decision in 0..organization.decisions.len() {
                match organization.decisions[decision] {
                    Decision::None => {}
                    Decision::Vetoed => {}
                    Decision::SetTaxes(target_province, tax_rate) => {
                        self.provinces[target_province].tax_rate = tax_rate;
                    }
                    Decision::DeclareWar(target_org) => {
                        self.relations
                            .get_relations_mut(organization_key, target_org)
                            .at_war = true;
                    }
                    Decision::MoveTroops(source, dest, ratio) => {
                        organization.transfer_troops(source, dest, ratio)
                    }
                };
                organization.decisions[decision] = Decision::None;
            }
            for (branch_key, branch_decision) in self.branch_control_decisions.iter_mut() {
                match branch_decision {
                    BranchControlDecision::None => {}
                    BranchControlDecision::Vetoed => {}
                    BranchControlDecision::Appoint(party) => {
                        self.branches[branch_key].controlling_party = *party;
                    }
                }
                *branch_decision = BranchControlDecision::None
            }

            //choose action
            // let org_keys = org_keys.iter().filter(|&&k| k != organization_key);
            // let action_decider = rng.f64();
            // let mut action = None;
            // if action_decider > 0.5 {
            //     let mut source_iter = self.organizations[organization_key]
            //         .province_control
            //         .0
            //         .iter()
            //         .enumerate()
            //         .filter(|&(i, &c)| c > 0.9);
            //     let source_count = source_iter.clone().count();
            //     if source_count > 0 {
            //         if let Some(source) = source_iter.nth(rng.usize(..(source_count))) {
            //             let dest_iter = self.organizations[organization_key]
            //                 .province_control
            //                 .0
            //                 .iter()
            //                 .enumerate()
            //                 .filter(|&(i, &c)| c > 0.1)
            //                 .flat_map(|(key, _)| {
            //                     self.provinces[ProvinceKey(key)]
            //                         .neighbouring_provinces
            //                         .iter()
            //                 });
            //             let dest_count = dest_iter.clone().count();
            //             if dest_count > 0 {
            //                 if let Some(&dest) = dest_iter.clone().nth(rng.usize(..dest_count)) {
            //                     action = Some(Action::MoveTroops {
            //                         source: ProvinceKey(source.0),
            //                         dest,
            //                         ratio: rng.f64(),
            //                     });
            //                 }
            //             }
            //         }
            //     }
            // } else {
            //     let action_decider = rng.f64();
            //     let law = if action_decider > 0.5 {
            //         Legislation::DeclareWar(OrganizationKey(
            //             rng.usize(..self.organizations.0.len()),
            //         ))
            //     } else {
            //         Legislation::SetTaxRate(rng.f64() * 0.2)
            //     };
            //     action = Some(Action::EnactLaw(law));
            // }

            // //Handle action
            // if let Some(action) = action {
            //     match action {
            //         Action::EnactLaw(law) => {
            //             match law {
            //                 Legislation::SetTaxRate(new_rate) => {
            //                     for province_key in self.organizations[organization_key]
            //                         .province_control
            //                         .0
            //                         .iter()
            //                         .enumerate()
            //                         .filter(|&(p_key, &c)| c > 0.1)
            //                         .map(|a| ProvinceKey(a.0))
            //                     {
            //                         self.provinces[province_key].tax_rate = new_rate;
            //                     }
            //                 }
            //                 Legislation::DeclareWar(target) => {
            //                     self.relations
            //                         .get_relations_mut(organization_key, target)
            //                         .at_war = true
            //                 } // Law::DeclareWar(target) => {}
            //             }
            //         }
            //         Action::MoveTroops {
            //             source,
            //             dest,
            //             ratio,
            //         } => self.organizations[organization_key].transfer_troops(source, dest, ratio),
            //     }
            // }

            let redistribution_amount =
                (self.organizations[organization_key].money * 0.5 * delta_year)
                    .min(self.organizations[organization_key].money);
            self.organizations[organization_key].money -= redistribution_amount;
            let province_control_sum = self.organizations[organization_key]
                .province_control
                .0
                .iter()
                .sum::<f64>();
            if province_control_sum > 0.8 {
                for (province_key, province_control) in self.organizations[organization_key]
                    .province_control
                    .0
                    .iter()
                    .enumerate()
                {
                    for slice in self.provinces[ProvinceKey(province_key)]
                        .pops
                        .pop_slices
                        .iter_mut()
                    {
                        let org_payment = redistribution_amount
                            * (province_control / province_control_sum)
                            / (NUM_SLICES as f64);
                        slice.money += org_payment;
                    }
                }
            }
        }
        for (province_key, province) in self.provinces.iter() {
            let province_key = province_key;
            if let Some((biggest_org_key, biggest_organization)) = self
                .organizations
                .0
                .iter_mut()
                .enumerate()
                .map(|(k, o)| (OrganizationKey(k), o))
                // .filter(|(k, o)| o.military.deployed_forces[province_key] > 1.0)
                .max_by_key(|(_org_key, org)| FloatOrd(org.military.deployed_forces[province_key]))
            {
                let control_increase = (1.0 * delta_year)
                    .min(1.0 - biggest_organization.province_control[province_key]);
                // biggest_organization.province_control[province_key] += control_increase;

                // let num_losing_orgs = self.organizations.iter().filter(|(k,o)|o.military.deployed_forces[province_key] > 1.0 && o.key != biggest_org_key).count();
                // let losing_control_sum = self
                //     .organizations.0
                //     .iter()
                //     .filter(|(_, o)| {
                //         o.military.deployed_forces[province_key] > 1.0 && o.key != biggest_org_key
                //     })
                //     .map(|o)| o.province_control[province_key])
                //     .sum::<f64>();
                let losing_control_sum = self
                    .organizations
                    .iter()
                    .filter(|&(k, _)| k != biggest_org_key)
                    .map(|(_, o)| o.province_control[province_key])
                    .sum::<f64>();
                let control_increase = control_increase.min(losing_control_sum);

                for (org_key, org) in self.organizations.iter_mut() {
                    if org_key != biggest_org_key && losing_control_sum > 0.0 {
                        let control_ratio = org.province_control[province_key] / losing_control_sum;
                        org.province_control[province_key] -= control_ratio * control_increase;
                    } else if org_key == biggest_org_key {
                        org.province_control[province_key] += control_increase;
                    }
                }
            }
        }
    }
    fn process_battles(&mut self, delta_year: f64) {
        for province_key in 0..self.provinces.0.len() {
            let province_key = ProvinceKey(province_key);
            // self.friendly_combatants.clear();
            // self.enemy_combatants.clear();

            let combat_orgs: Box<[OrganizationKey]> = self
                .organizations
                .iter()
                .filter(|o| o.1.military.deployed_forces[province_key] > 0.5)
                .map(|a| a.0)
                .collect();

            let mut combat_orgs_filtered = HashMap::with_capacity(combat_orgs.len());
            // let mut combat_orgs_filtered = Vec::with_capacity(combat_orgs.len());
            for i in 0..combat_orgs.len() {
                for j in 0..combat_orgs.len() {
                    if self
                        .relations
                        .get_relations(combat_orgs[i], combat_orgs[j])
                        .at_war
                    {
                        combat_orgs_filtered
                            .entry(combat_orgs[i])
                            .or_insert(Vec::with_capacity(combat_orgs.len()))
                            .push(combat_orgs[j]);
                    }
                }
            }
            // let combat_orgs: HashMap<_, _, BuildNoHashHasher<OrganizationKey>> =
            //     combat_orgs_filtered
            //         .iter()
            //         .filter(|&(_k, v)| v.len() > 0)
            //         .collect();
            if combat_orgs_filtered.len() <= 0 {
                continue;
            }
            let first_index = get_battle_seed(
                self.current_year,
                province_key.0,
                combat_orgs_filtered.len(),
            );
            // dbg!(first_index);
            let &first_combatant_key = combat_orgs_filtered.keys().nth(first_index).unwrap();

            let second_index = get_battle_seed(
                self.current_year,
                province_key.0,
                combat_orgs_filtered[&first_combatant_key].len(),
            );
            let second_combatant_key = combat_orgs_filtered[&first_combatant_key][second_index];

            let first_army_size: f64 = self.organizations[first_combatant_key]
                .military
                .deployed_forces[province_key];
            let second_army_size: f64 = self.organizations[second_combatant_key]
                .military
                .deployed_forces[province_key];
            let avg_size = (first_army_size + second_army_size) / 2.0;

            // let army_ratio = if first_army_size > second_army_size {
            //     first_army_size / second_army_size
            // } else {
            //     second_army_size / first_army_size
            // };

            {
                let first_army = &mut self.organizations[first_combatant_key].military;

                // let losses = avg_size * (1.0 / army_ratio) * delta_year;
                let losses = avg_size * 10.0 * delta_year;
                first_army.deployed_forces[province_key] -=
                    losses.min(first_army.deployed_forces[province_key]);
            }

            {
                let second_army = &mut self.organizations[second_combatant_key].military;

                // let losses = avg_size * army_ratio * delta_year;
                let losses = avg_size * 10.0 * delta_year;
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
