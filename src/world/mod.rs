use std::{
    collections::{HashMap, HashSet, VecDeque},
    convert::TryFrom,
    f64::consts::{E, PI},
    fmt::Display,
    hash::Hash,
    ops::{Index, IndexMut},
    time::Instant,
};

pub mod agent;
pub mod equipment;
pub mod ideology;
pub mod language;
pub mod organization;
mod party_names;
pub mod questions;
pub mod recipes;

use float_ord::FloatOrd;
use nalgebra::Vector3;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use rand::{prelude::Distribution, rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use statrs::distribution::Gamma;
use variant_count::VariantCount;

use crate::{
    support::{exponential_decay, hash_usize_fast, map_range_linear_f64},
    world::{
        ideology::{
            positions::{
                Assimilation, ExportDifficulty, Immigration, ImportDifficulty, IndustrySupport,
                Irredentism, LanguageSupport, Research, SeparationOfPowers, TaxMethod, TaxRate,
                WarSupport, WelfareSupport,
            },
            Ideology,
        },
        language::{Language, LanguageManager, LanguageMap},
        party_names::PARTY_NAMES,
        recipes::get_recipe_count,
    },
};

use self::{
    ideology::{Beliefs, PoliticalParty, PoliticalPartyMap},
    organization::{
        Branch, BranchControlDecision, BranchControlFlag, BranchMap, DecisionCategory,
        DecisionControlFlag, DiplomaticOffer, DiplomaticOfferType, Military, Organization,
        OrganizationKey, Relation,
    },
    recipes::{get_enabled_buildings, Recipe, RECIPES},
};

const TRICKLEBACK_RATIO: f64 = 0.25;

#[repr(usize)]
#[derive(Clone, Copy, VariantCount, Debug, IntoPrimitive, TryFromPrimitive)]
pub enum Culture {
    CultureA,
}
#[repr(usize)]
#[derive(Clone, Copy, VariantCount, Debug, IntoPrimitive, TryFromPrimitive)]
pub enum Industry {
    Agriculture,
    Mining,
    Manufacturing,
    GeneralLabor,
    Unemployed,
}

#[repr(usize)]
#[derive(
    Clone, Copy, VariantCount, Debug, IntoPrimitive, TryFromPrimitive, Serialize, Deserialize,
)]
pub enum Good {
    Produce,
    AnimalProducts,
    ProcessedFood,
    Iron,
    NonFerrousMetal,
    Carbon,
    Steel,
    StableSuperheavyElement,
    Silica,
    Semiconductor,
    RawHydrocarbons,
    Plastics,
    Composites,
    Batteries,
    SolarCells,
    LowGradeElectronics,
    HighGradeElectronics,
    Explosives,
    SmallArms,
    HeavyWeaponry,
    Ammunition,
    Armor,
}

#[repr(usize)]
#[derive(Clone, Copy, VariantCount, Debug, IntoPrimitive, TryFromPrimitive)]
pub enum IntangibleGoods {
    //Aren't owned by anyone, reside in the province level, can't be moved or traded, only consumed
    Work, // in man-years
}
#[repr(usize)]
#[derive(
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Hash,
    VariantCount,
    Debug,
    IntoPrimitive,
    TryFromPrimitive,
)]
pub enum Building {
    ProduceFarm,
    LivestockFarm,
    IronMine,
    NonFerrousMetalMine,
    SilicaQuarry,
    FossilFuelExtraction,
    VerticalFarm,
    FoodProcessor,
    Metalworks,
    ParticleCollider,
    CompositesFacility,
    ChemicalPlant,
    SemiconductorFab,
    ArmsFactory,
}
impl Building {
    pub fn get_industry(&self) -> Industry {
        if AGRICULTURE_BUILDINGS.contains(self) {
            return Industry::Agriculture;
        }
        if MINING_BUILDINGS.contains(self) {
            return Industry::Mining;
        }
        if MANUFACTURING_BUILDINGS.contains(self) {
            return Industry::Manufacturing;
        }
        unreachable!()
    }
}

const AGRICULTURE_BUILDINGS: &[Building] = &[
    Building::ProduceFarm,
    Building::LivestockFarm,
    Building::VerticalFarm,
];

const MINING_BUILDINGS: &[Building] = &[
    Building::IronMine,
    Building::NonFerrousMetalMine,
    Building::SilicaQuarry,
    Building::FossilFuelExtraction,
];

const MANUFACTURING_BUILDINGS: &[Building] = &[
    Building::FoodProcessor,
    Building::Metalworks,
    Building::ParticleCollider,
    Building::CompositesFacility,
    Building::ChemicalPlant,
    Building::SemiconductorFab,
    Building::ArmsFactory,
];

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
    fn process_organizations(&mut self, organizations: &mut [Organization]) {
        for organization in organizations {
            organization.demand = [0.0; Good::VARIANT_COUNT];
            organization.supply = [0.0; Good::VARIANT_COUNT];

            let needs = organization.military.get_total_needs();
            let needs_sum = needs
                .iter()
                .enumerate()
                .map(|(i, need)| (need - organization.owned_goods[i]).max(0.0))
                .sum::<f64>();
            // if needs_sum.is_nan(){
            //     dbg!(needs);
            //     dbg!(organization.owned_goods);
            // }
            for i in 0..Good::VARIANT_COUNT {
                let good = Good::try_from(i).unwrap();

                let good_needed = needs[good as usize];
                let good_needed_to_be_bought =
                    (good_needed - organization.owned_goods[good as usize]).max(0.0);

                // let affordable_good = (organization.money / (good_needed/needs_sum)) / self.price[good as usize];
                let affordability_ratio = (good_needed_to_be_bought / needs_sum).max(0.0);
                let affordable_good =
                    (organization.money * affordability_ratio) / self.price[good as usize];
                assert!(affordability_ratio <= 1.0);
                let good_demand = (good_needed_to_be_bought).min(affordable_good).max(0.0);
                organization.demand[good as usize] = good_demand;
                self.demand[good as usize] += good_demand;

                let good_supply = 0.0; //TODO: Manual selling of goods
                organization.supply[good as usize] = good_supply;
                self.supply[good as usize] += good_supply;
            }
        }
    }
    fn process(
        &mut self,
        delta_year: f64,
        pop_slices: &mut [PopSlice],
        building_ratios: &[f64],
        recipe_ratios: &[f64],
        spend_leftovers_ratio: f64,
        trade_modifier: f64, //modifies supply and demand so that some provinces are less efficient to import/export to/from
    ) {
        for (index, slice) in pop_slices.iter_mut().enumerate() {
            // slice.met_inputs = [0.0; Good::VARIANT_COUNT];
            slice.individual_demand = [0.0; Good::VARIANT_COUNT];
            slice.individual_supply = [0.0; Good::VARIANT_COUNT];

            let _culture = Culture::try_from(index / Industry::VARIANT_COUNT).unwrap();
            let industry = Industry::try_from(index % Industry::VARIANT_COUNT).unwrap();
            let mut industry_needs = std::array::from_fn(|_| 0.0);

            let matching_buildings = get_enabled_buildings(industry);
            for (recipe_index, recipe) in RECIPES.iter().enumerate().filter(|(i, r)| {
                matching_buildings
                    .iter()
                    .any(|b| r.building as usize == *b as usize)
            }) {
                let building_ratio = building_ratios[recipe.building as usize];
                let recipe_ratio = recipe_ratios[recipe_index];
                let cheapest_good = Good::try_from(
                    (0..Good::VARIANT_COUNT)
                        .into_iter()
                        .min_by_key(|&g| FloatOrd(self.price[g]))
                        .unwrap(),
                )
                .unwrap();
                let building_population = slice.population
                    * building_ratio
                    * recipe_ratio
                    * (1.0 / Culture::VARIANT_COUNT as f64);
                add_inputs(
                    &mut industry_needs,
                    recipe,
                    cheapest_good,
                    building_population,
                    delta_year,
                )
            }
            // dbg!(&industry_needs);
            let goods_needed: Vec<_> = industry_needs
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
    fn buy_and_sell_organizations(&mut self, good: Good, organizations: &mut [Organization]) {
        for organization in organizations {
            // for g in &organization.owned_goods {
            //     assert!(!g.is_nan())
            // }
            //Selling
            let sell_ratio = if self.supply[good as usize] > 0.0 {
                (self.demand[good as usize] / self.supply[good as usize]).min(1.0)
            } else {
                0.0
            };
            let amount_sold = organization.supply[good as usize] * sell_ratio;
            organization.owned_goods[good as usize] -= amount_sold;
            let price_of_goods_sold = amount_sold * self.price[good as usize];
            organization.money += price_of_goods_sold;

            //Buying
            let buy_ratio = if self.demand[good as usize] > 0.0 {
                (self.supply[good as usize] / self.demand[good as usize]).min(1.0)
            } else {
                0.0
            };
            let amount_bought = organization.demand[good as usize] * buy_ratio;
            organization.owned_goods[good as usize] += amount_bought;
            // for g in &organization.owned_goods {
            //     assert!(!g.is_nan())
            // }
            let price_of_goods_bought = amount_bought * self.price[good as usize];
            organization.money -= price_of_goods_bought;
            // for g in &organization.owned_goods {
            //     assert!(!g.is_nan())
            // }
            // if organization.money < 0.0{
            //     dbg!(organization.money);
            //     dbg!(organization.demand);
            //     dbg!(organization.owned_goods);
            // }
            // assert!(organization.money >= 0.0);
        }
    }
    fn buy_and_sell(&mut self, good: Good, pop_slices: &mut [PopSlice]) {
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
            // if slice.money < 0.0{
            //     dbg!(slice.money);
            // }
            // assert!(slice.money >= 0.0);

            assert!(!slice.money.is_nan());
        }
    }
    fn update_price(&mut self, good: Good, delta_year: f64, error_divisor: f64) {
        const PRICE_CHANGE: f64 = 1.0;
        if self.supply[good as usize] > self.demand[good as usize] {
            self.price[good as usize] +=
                exponential_decay(self.price[good as usize], -PRICE_CHANGE, delta_year);
        } else {
            self.price[good as usize] +=
                exponential_decay(self.price[good as usize], PRICE_CHANGE, delta_year);
        }

        self.price[good as usize] = self.price[good as usize].clamp(0.01, 1_000_000.0);
        assert!(!self.price[good as usize].is_nan());
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
    pub need_bottleneck: Option<Good>,
    pub trickleback: f64,
    pub beliefs: Beliefs,
    pub produced_goods: [f64; Good::VARIANT_COUNT],
}
impl PopSlice {
    const NUM_MET_NEEDS_STORED: usize = 30;
}

const NUM_SLICES: usize = Culture::VARIANT_COUNT * Industry::VARIANT_COUNT;

// const GOOD_STORAGE_RATIO: f64 = 1000.0;

const SIM_START_YEAR: f64 = 2022.0;
const HEALTHY_SLICE_NEEDS_RATIO: f64 = 0.9;
const DAYS_IN_YEAR: f64 = 365.0;

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

fn add_inputs(
    inputs: &mut [f64; Good::VARIANT_COUNT],
    recipe: &Recipe,
    cheapest_good: Good,
    population: f64,
    delta_year: f64,
) {
    for recipe_inputs in recipe.inputs {
        match recipe_inputs {
            recipes::RecipeGood::Anything(amount) => {
                inputs[cheapest_good as usize] += *amount * delta_year * population;
            }
            recipes::RecipeGood::Good { good, amount } => {
                inputs[*good as usize] += *amount * delta_year * population;
            }
            recipes::RecipeGood::RandomRatioGood { good, weight } => {
                unreachable!();
            }
        }
    }
}

fn get_needs(population: f64, delta_year: f64) -> [f64; Good::VARIANT_COUNT] {
    let mut output = [0.0; Good::VARIANT_COUNT];

    output[Good::ProcessedFood as usize] = population * 1.0 * DAYS_IN_YEAR * delta_year;
    output[Good::LowGradeElectronics as usize] = population * 0.00001 * DAYS_IN_YEAR * delta_year;
    output[Good::HighGradeElectronics as usize] = population * 0.000001 * DAYS_IN_YEAR * delta_year;

    return output;
}

fn add_outputs(
    outputs: &mut [f64; Good::VARIANT_COUNT],
    recipe: &Recipe,
    industry_ratio: f64,
    industry_data: IndustryData,
    population: f64,
    delta_year: f64,
) {
    for recipe_outputs in recipe.outputs {
        match recipe_outputs {
            recipes::RecipeGood::Anything(amount) => {
                unreachable!();
            }
            recipes::RecipeGood::Good { good, amount } => {
                outputs[*good as usize] += production_curve(
                    population,
                    1.0,
                    *amount,
                    industry_data.size * industry_ratio,
                ) * delta_year;
            }
            recipes::RecipeGood::RandomRatioGood { good, weight } => {
                //TODO: Replace this placeholder with actual randomness
                outputs[*good as usize] += production_curve(
                    population,
                    1.0,
                    *weight,
                    industry_data.size * industry_ratio,
                ) * delta_year;
            }
        }
    }
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
        let mut rng = rand::thread_rng();
        for _ in 0..NUM_SLICES {
            let slice_population =
                (total_population / NUM_SLICES as f64) * (rng.gen::<f64>() + 0.5);
            out.push(PopSlice {
                population: slice_population,
                money: 10_000.0 * slice_population,
                // money: 0.01 * slice_population,
                // owned_goods: [1_000_000_000_000.0 / SLICE_COUNT as f64; Good::VARIANT_COUNT],
                owned_goods: [0.0 * slice_population as f64; Good::VARIANT_COUNT],
                // met_inputs: [0.0; Good::VARIANT_COUNT],
                individual_demand: [0.0; Good::VARIANT_COUNT],
                individual_supply: [0.0; Good::VARIANT_COUNT],
                previous_met_needs: VecDeque::with_capacity(PopSlice::NUM_MET_NEEDS_STORED),
                minimum_met_needs: f64::NAN,
                need_bottleneck: None,
                trickleback: 0.0,
                beliefs: Beliefs::new_random(),
                produced_goods: [0.0; Good::VARIANT_COUNT],
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
    pub recipe_ratios: [f64; RECIPES.len()], //what recipes are being used
    pub building_ratios: [f64; Building::VARIANT_COUNT], //what buildings have been built
    pub intangible_goods: [f64; IntangibleGoods::VARIANT_COUNT],
    pub feb_temp: f64,
    pub july_temp: f64,
    pub trader_cost_multiplier: f64, //Excess goes to traders in global market. Abstraction for difficulties in exporting from remote or blockaded provinces, WIP and not really functional
    pub tax_bank: f64,
    pub troop_bank: f64,
    pub combat_width: f64,
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
    organization_owner: Vec<Option<u16>>,
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
        owner: Option<u16>,
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
    pub fn owner(&self) -> Option<u16> {
        let mut occurences = HashMap::with_capacity(self.organization_owner.len());
        for owner in &self.organization_owner {
            if let Some(owner) = owner {
                *occurences.entry(owner).or_insert(0u16) += 1;
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

const BATTLE_DURATION: f64 = 365.0 * 4.0;
fn get_battle_seed(current_year: f64, offset: usize, max: usize) -> usize {
    let seed = (current_year * BATTLE_DURATION) as usize + offset;
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
    // pub player_organization: OrganizationKey,
    pub language_manager: LanguageManager,
    pub branches: BranchMap<Branch>,
    pub political_parties: PoliticalPartyMap<PoliticalParty>,
    available_party_names: Vec<String>,
    branch_control_decisions: BranchMap<BranchControlDecision>,
    last_delta_year: f64, //delta year for last tick, used for calculating gdp
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
                                .deployed_forces[selected_province_key]
                                .num_troops += amount;
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

    pub fn get_org_population(&self, organization_key: OrganizationKey) -> f64 {
        let mut population = 0.0;
        for (province_key, _) in self.organizations[organization_key]
            .province_control
            .iter()
            .filter(|(k, c)| **c > 0.9)
        {
            for slice in self.provinces[province_key].pops.pop_slices.iter() {
                population += slice.population
            }
        }
        population
    }
    pub fn get_org_pop_wealth(&self, organization_key: OrganizationKey) -> f64 {
        let mut wealth = 0.0;
        for (province_key, _) in self.organizations[organization_key]
            .province_control
            .iter()
            .filter(|(k, c)| **c > 0.9)
        {
            for slice in self.provinces[province_key].pops.pop_slices.iter() {
                wealth += slice.money;
                for (good_index, owned_amount) in slice.owned_goods.iter().enumerate() {
                    wealth += owned_amount * self.global_market.price[good_index];
                }
            }
        }
        wealth
    }

    pub fn get_org_wealth(&self, organization_key: OrganizationKey) -> f64 {
        let mut wealth = 0.0;
        wealth += self.organizations[organization_key].money;
        for (good_index, owned_amount) in self.organizations[organization_key]
            .owned_goods
            .iter()
            .enumerate()
        {
            wealth += owned_amount * self.global_market.price[good_index];
        }
        wealth
    }

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
            + self.organizations.values().map(|o| o.money).sum::<f64>()
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

    pub fn push_diplomatic_offer(
        &mut self,
        source_org: OrganizationKey,
        target_org: OrganizationKey,
        offer_type: DiplomaticOfferType,
    ) {
        self.organizations[target_org]
            .diplomatic_offers
            .push(DiplomaticOffer {
                from: source_org,
                time_sent: self.current_year,
                offer_type,
            })
    }

    pub fn load(file: &mut std::fs::File) -> Self {
        let mut reader = snap::read::FrameDecoder::new(file);
        serde_json::from_reader(&mut reader).expect("Failed to decode World")
    }

    pub fn save(&self, filename: &str) {
        return;
        // let out_writer = std::fs::File::create(filename).expect("Failed to open file for writing");
        // let mut out_writer = snap::write::FrameEncoder::new(out_writer);
        // serde_json::to_writer(&mut out_writer, self).expect("Failed to encode World")
    }

    pub fn new(
        vertices: &[Vector3<f32>],
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
        let mut rng = rand::thread_rng();
        for (province_key, province_indices) in province_indices.0.iter().enumerate() {
            let province_key = ProvinceKey(province_key);
            if province_indices.len() < 3 {
                println!("degenerate province given: {:}", province_key.0);
                dbg!(province_indices);
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
                    * rng.gen::<f64>()
                    * match industry {
                        Industry::Agriculture => {
                            (province_data[province_key].aridity() as f64 - 0.3).clamp(0.1, 1.1)
                        }
                        Industry::Mining => province_data[province_key].ore() as f64,
                        Industry::Manufacturing => 1.0,
                        Industry::GeneralLabor => 1.0,
                        Industry::Unemployed => 1.0,
                    };
            }

            let recipe_ratios = std::array::from_fn(|i| {
                let recipe = &RECIPES[i];
                let building = recipe.building;
                let num_recipes = get_recipe_count(building);
                1.0 / (num_recipes as f64)
            });

            let mut building_ratios = [0.0; Building::VARIANT_COUNT];
            for building in AGRICULTURE_BUILDINGS {
                building_ratios[*building as usize] = 1.0 / (AGRICULTURE_BUILDINGS.len() as f64);
            }
            for building in MINING_BUILDINGS {
                building_ratios[*building as usize] = 1.0 / (MINING_BUILDINGS.len() as f64);
            }
            for building in MANUFACTURING_BUILDINGS {
                building_ratios[*building as usize] = 1.0 / (MANUFACTURING_BUILDINGS.len() as f64);
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
                recipe_ratios,
                building_ratios,
                trader_cost_multiplier: 1.0,
                feb_temp: province_data[province_key].feb_temps(),
                july_temp: province_data[province_key].july_temps(),
                neighbouring_provinces: vec![].into_boxed_slice(),
                tax_bank: 0.0,
                tax_rate: 0.1,
                troop_bank: 0.0,
                combat_width: 1000.0,
                // recruit_limiter: 0.0,
            });

            if let Some(owner) = province_data[province_key].owner() {
                // let key = country_id_to_org_key.entry(owner).or_insert_with(|| {
                //     let new_org = Organization::new(&nation_names[owner], num_provinces);
                //     let new_org_key = new_org.key;
                //     organizations.entry(new_org_key).or_insert(new_org);
                //     new_org_key
                // });
                organizations[OrganizationKey(owner as usize)].province_control[province_key] = 1.0;
                organizations[OrganizationKey(owner as usize)]
                    .military
                    .deployed_forces[province_key]
                    .num_troops = 15_000.0;
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
        const PRE_SIM_STEPS: usize = 0;
        // const PRE_SIM_STEPS: usize = 0;

        let histories = Histories::new(PRE_SIM_STEPS, num_provinces);

        let relations = RelationMap::new(&organizations.0);
        let num_organizations = organizations.0.len();

        // for org in organizations.values(){
        //     println!("{}", org.name)
        // }
        let player_organization = OrganizationKey(rng.gen_range(0..num_organizations));
        // let player_organization = OrganizationKey(0);

        let mut num_uncontrolled_provinces = 0;
        for (province_key, province) in provinces.iter_mut() {
            if (organizations
                .0
                .iter()
                .map(|o| o.province_control[province_key])
                .sum::<f64>())
                < 0.1
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
                dbg!(province_key, &province.name);
                let num_orgs = organizations.0.len();
                num_uncontrolled_provinces += 1;

                // for org in organizations.0.iter_mut() {
                //     org.province_control[province_key] = 1.0 / num_orgs as f64;
                // }

                organizations[OrganizationKey(rng.gen_range(0..num_orgs))].province_control
                    [province_key] = 1.0;
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
                    industry: IndustrySupport::Indifferent,
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
                    industry: IndustrySupport::FactoryMustGrow,
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
                    industry: IndustrySupport::Indifferent,
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
            // player_organization,
            language_manager: LanguageManager { languages },
            branch_control_decisions: BranchMap(vec![
                BranchControlDecision::None;
                branches.0.len()
            ]),
            branches,
            political_parties,
            available_party_names: PARTY_NAMES.into_iter().map(|s| s.to_string()).collect(),
            last_delta_year: f64::NAN,
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
        self.last_delta_year = delta_year;
        // let rng = rand::thread_rng();
        self.process_battles(delta_year);
        //Process market, selling yesterday's produced goods and buying needs for today
        for province in self.provinces.0.iter_mut() {
            //update productivities
            province.industry_data[Industry::Agriculture as usize].productivity = kernel(
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
            province.market.process(
                delta_year,
                &mut province.pops.pop_slices,
                &province.building_ratios,
                &province.recipe_ratios,
                0.5,
                1.0,
            );
            for good_index in 0..Good::VARIANT_COUNT {
                let good = Good::try_from(good_index).unwrap();
                province
                    .market
                    .buy_and_sell(good, &mut province.pops.pop_slices);
                province
                    .market
                    .update_price(good, delta_year, province.pops.population().max(1.0));
            }
        }
        self.global_market.demand = [0.0; Good::VARIANT_COUNT];
        self.global_market.supply = [0.0; Good::VARIANT_COUNT];
        self.global_market
            .process_organizations(&mut self.organizations.0);
        for province in self.provinces.0.iter_mut() {
            self.global_market.process(
                delta_year,
                &mut province.pops.pop_slices,
                &province.building_ratios,
                &province.recipe_ratios,
                1.0,
                province.trader_cost_multiplier,
            );
        }
        for good_index in 0..Good::VARIANT_COUNT {
            let good = Good::try_from(good_index).unwrap();
            self.global_market
                .buy_and_sell_organizations(good, &mut self.organizations.0);
            for province in self.provinces.0.iter_mut() {
                self.global_market
                    .buy_and_sell(good, &mut province.pops.pop_slices);
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
                let _industry_index = slice_index % Industry::VARIANT_COUNT;
                let (need_bottleneck, minimum_met_needs) = needs
                    .iter()
                    .enumerate()
                    .map(|(good_index, need_amount)| {
                        // println!("good: {:?}, amount: {:}",Good::try_from(good_index).unwrap() , need_amount);
                        (
                            good_index,
                            if need_amount <= &0.0 {
                                1.0
                            } else {
                                (slice.owned_goods[good_index] / need_amount).min(1.0)
                            },
                        )
                    })
                    .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                    .unwrap();
                slice.minimum_met_needs = minimum_met_needs;
                slice.need_bottleneck = Some(Good::try_from(need_bottleneck).unwrap());
                slice
                    .previous_met_needs
                    .push_back(minimum_met_needs.min(1.0));
                if slice.previous_met_needs.len() > PopSlice::NUM_MET_NEEDS_STORED {
                    slice.previous_met_needs.remove(0);
                }
                for (good_index, &need) in needs.iter().enumerate() {
                    slice.owned_goods[good_index] -= need * minimum_met_needs;
                }
                if minimum_met_needs < HEALTHY_SLICE_NEEDS_RATIO {
                    num_unhealthy_slices += 1;
                }
            }
            let num_healthy_slices = NUM_SLICES - num_unhealthy_slices;

            //consume daily living needs
            for (slice_index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                // let population_change = if slice.minimum_met_needs > 0.9 {
                //     slice.population
                //         * 0.02
                //         * delta_year
                //         * map_range_linear_f64(slice.minimum_met_needs, 0.98, 1.0, 0.1, 1.0)
                // } else if slice.minimum_met_needs > 0.25 {
                //     ((((hash_usize_fast((self.current_year * 365.0 * 24.0) as usize) as f64)
                //         / (usize::MAX as f64))
                //         * 2.0)
                //         - 1.0)
                //         * 0.5
                // } else {
                //     (slice.population * -0.3 * delta_year).min(-100.0 * delta_year)
                // }
                // .max(-slice.population);
                let population_growth_rate = if slice.minimum_met_needs > 0.9 {
                    0.01 + map_range_linear_f64(slice.minimum_met_needs, 0.98, 1.0, 0.0, 0.02)
                } else if slice.minimum_met_needs > 0.25 {
                    // ((((hash_usize_fast((self.current_year * 365.0 * 24.0*20.0) as usize) as f64)
                    //     / (usize::MAX as f64))
                    //     * 2.0)
                    //     - 1.0)
                    //     * 0.01
                    0.0
                } else {
                    -0.1
                };
                // let population_change = slice.population * population_multiplication_factor - slice.population;
                // dbg!(population_multiplication_factor);
                // let population_change = slice.population * population_growth_rate - 1.0);
                let population_change =
                    exponential_decay(slice.population, population_growth_rate, delta_year);
                // dbg!(population_change);
                let cannibalism_amount =
                    (-population_change).max(0.0) / slice.population * TRICKLEBACK_RATIO;
                slice.trickleback += cannibalism_amount;
                // let trickleback_amount =
                //     (slice.trickleback * 0.5 * delta_year).min(slice.trickleback);
                let trickleback_amount = exponential_decay(slice.trickleback, 0.5, delta_year);

                slice.population += trickleback_amount;
                slice.trickleback -= trickleback_amount;
                assert!(!slice.population.is_nan());
                slice.population += population_change;
                // dbg!(slice.population);
                if slice.minimum_met_needs < 0.9 {
                    // let migration_pops_amount =
                    //     (slice.population * 0.1 * delta_year).min(slice.population);
                    let migration_pops_amount =
                        exponential_decay(slice.population, 0.1, delta_year);
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
                    slice.money -= migration_money_amount;
                    // assert!(slice.money >= 0.0)
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
                        .filter(|s| s.minimum_met_needs >= HEALTHY_SLICE_NEEDS_RATIO)
                    {
                        let migration_amount = migration_pool_pops / (num_healthy_slices as f64);
                        slice.population += migration_amount;
                        assert!(!migration_pool_money.is_nan());
                        slice.money += migration_pool_money / (num_healthy_slices as f64);
                        total_migrations += migration_amount;
                    }
                } else {
                    let slice = province
                        .pops
                        .pop_slices
                        .iter_mut()
                        .max_by_key(|s| FloatOrd(s.minimum_met_needs))
                        .unwrap();
                    let migration_amount = migration_pool_pops;
                    slice.population += migration_amount;
                    slice.money += migration_pool_money;
                    total_migrations += migration_amount;

                    // for slice in province.pops.pop_slices.iter_mut() {
                    //     let migration_amount = migration_pool_pops / (NUM_SLICES as f64);
                    //     slice.population += migration_amount;
                    //     slice.money += migration_pool_money / (NUM_SLICES as f64);
                    //     total_migrations += migration_amount;
                    // }
                }
            }
            // dbg!(num_unhealthy_slices);
            // dbg!(migration_pool_pops);
            // dbg!(total_migrations);
            assert!((migration_pool_pops - total_migrations).abs() < 1.0);

            // let industry_populations = {
            //     let mut industry_populations = [0.0; Industry::VARIANT_COUNT];

            //     for (slice_index, slice) in province.pops.pop_slices.iter().enumerate() {
            //         let industry =
            //             Industry::try_from(slice_index % Industry::VARIANT_COUNT).unwrap();
            //         industry_populations[industry as usize] += slice.population;
            //     }
            //     industry_populations
            // };

            //use remaining resources for industry
            let mut _money_sum = 0.0;
            let cheapest_good = Good::try_from(
                (0..Good::VARIANT_COUNT)
                    .into_iter()
                    .min_by_key(|&g| FloatOrd(province.market.price[g]))
                    .unwrap(),
            )
            .unwrap();

            for (slice_index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                slice.produced_goods = [0.0; Good::VARIANT_COUNT];
                _money_sum += slice.money;
                let industry = Industry::try_from(slice_index % Industry::VARIANT_COUNT).unwrap();

                let mut total_requested_inputs = [0.0; Good::VARIANT_COUNT];

                let matching_buildings = get_enabled_buildings(industry);
                for (recipe_index, recipe) in RECIPES.iter().enumerate().filter(|(i, r)| {
                    matching_buildings
                        .iter()
                        .any(|b| r.building as usize == *b as usize)
                }) {
                    let building_ratio = province.building_ratios[recipe.building as usize];
                    let recipe_ratio = province.recipe_ratios[recipe_index];
                    let building_population = slice.population
                        * building_ratio
                        * recipe_ratio
                        * (1.0 / Culture::VARIANT_COUNT as f64);
                    add_inputs(
                        &mut total_requested_inputs,
                        recipe,
                        cheapest_good,
                        building_population,
                        delta_year,
                    )
                }

                let actual_total_inputs: [_; Good::VARIANT_COUNT] =
                    std::array::from_fn(|i| total_requested_inputs[i].min(slice.owned_goods[i]));
                let allocated_amounts_and_ratio: [_; Good::VARIANT_COUNT] =
                    std::array::from_fn(|i| {
                        let allocation_ratio =
                            (actual_total_inputs[i] / total_requested_inputs[i]).min(1.0);
                        (allocation_ratio, allocation_ratio * slice.owned_goods[i])
                    });

                for (recipe_index, recipe) in RECIPES.iter().enumerate().filter(|(i, r)| {
                    matching_buildings
                        .iter()
                        .any(|b| r.building as usize == *b as usize)
                }) {
                    let building_ratio = province.building_ratios[recipe.building as usize];
                    let recipe_ratio = province.recipe_ratios[recipe_index];
                    let building_population = slice.population
                        * building_ratio
                        * recipe_ratio
                        * (1.0 / Culture::VARIANT_COUNT as f64);

                    let mut recipe_inputs = [0.0; Good::VARIANT_COUNT];
                    add_inputs(
                        &mut recipe_inputs,
                        recipe,
                        cheapest_good,
                        building_population,
                        delta_year,
                    );
                    // let min_met_input = (0..Good::VARIANT_COUNT).into_iter().map(|good_index|{
                    //     let (allocated_ratio, allocated_amount) = allocated_amounts_and_ratio[good_index];
                    //     FloatOrd((recipe_inputs[good_index] * allocated_ratio ) / allocated_amount)
                    // }).min().unwrap().0.min(1.0);
                    let min_met_input = (0..Good::VARIANT_COUNT)
                        .into_iter()
                        .filter(|&g| recipe_inputs[g] > 0.0)
                        .map(|good_index| {
                            let (allocated_ratio, _) = allocated_amounts_and_ratio[good_index];
                            FloatOrd(allocated_ratio)
                        })
                        .min()
                        .unwrap_or(FloatOrd(1.0))
                        .0
                        .min(1.0);
                    let mut outputs = [0.0; Good::VARIANT_COUNT];
                    add_outputs(
                        &mut outputs,
                        recipe,
                        recipe_ratio * building_ratio,
                        province.industry_data[industry as usize],
                        building_population,
                        delta_year,
                    );

                    for (good_index, output) in outputs.iter().enumerate() {
                        slice.produced_goods[good_index] += output;
                        // slice.produced_goods[good_index] -= recipe_inputs[good_index];
                        let good = Good::try_from(good_index).unwrap();
                        // let (allocated_ratio, _) = allocated_amounts_and_ratio[good_index];
                        slice.owned_goods[good as usize] -=
                            recipe_inputs[good as usize] * min_met_input;
                        slice.owned_goods[good as usize] += output * min_met_input;

                        slice.owned_goods[good as usize] -=
                            exponential_decay(slice.owned_goods[good as usize], 0.5, delta_year);
                        //Destroy goods over time
                        // slice.owned_goods[good as usize] = slice.owned_goods[good as usize]
                        //     .min(slice.population * GOOD_STORAGE_RATIO)
                    }
                }

                // //have to correct for entire industry population
                // for output in &mut outputs {
                //     *output *= slice.population / industry_populations[industry as usize];
                // }
            }
            //collect taxes and troops
            // let max_recruit_ratio = 0.01;
            let recruit_speed: f64 = 0.01;
            // let recruit_limiter_gain_speed = 1.0;
            // let recruit_limiter_decay_speed = recruit_limiter_gain_speed * 0.1;

            for (_index, slice) in province.pops.pop_slices.iter_mut().enumerate() {
                // let tax_amount = (slice.money * province.tax_rate * delta_year).min(slice.money);
                let tax_amount = exponential_decay(slice.money, province.tax_rate, delta_year);
                let local_tax_proportion = 1.0;
                province.tax_bank += tax_amount * local_tax_proportion;
                global_tax_bank += tax_amount * (1.0 - local_tax_proportion);
                slice.money -= tax_amount;
                // assert!(slice.money >= 0.0);

                if province.troop_bank < 10_000.0 {
                    let recruit_amount =
                        exponential_decay(slice.population, recruit_speed, delta_year);
                    slice.population -= recruit_amount;
                    province.troop_bank += recruit_amount;
                    if recruit_amount < 0.0 {
                        dbg!(recruit_amount);
                        dbg!(province.troop_bank);
                    }
                    assert!(recruit_amount > -1.0);
                    // dbg!(recruit_amount);
                    // dbg!(province.troop_bank);
                }

                if slice.money.is_nan() {
                    dbg!(tax_amount);
                }
                assert!(!slice.money.is_nan());
            }
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
            let mut new_troop_bank = province.troop_bank;
            assert!(!province.troop_bank.is_nan());
            for organization in self.organizations.0.iter_mut() {
                let transfer_amount =
                    province.tax_bank * organization.province_control[province_key];
                organization.money += transfer_amount;
                // province.tax_bank -= transfer_amount;
                assert!(!organization.province_control[province_key].is_nan());
                let num_provinces = organization
                    .province_control
                    .0
                    .iter()
                    .filter(|&&c| c > 0.1)
                    .count();
                let troop_transfer_amount =
                    province.troop_bank * organization.province_control[province_key];

                if num_provinces > 0 && organization.building_troops {
                    let added_troops = organization.military.forces_in_waiting.build_troops(
                        &mut organization.owned_goods,
                        troop_transfer_amount,
                        1.0 / num_provinces as f64,
                    );
                    new_troop_bank -= added_troops;
                    organization.military.forces_in_waiting.num_troops += added_troops;
                }
            }
            province.troop_bank = new_troop_bank;
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
        let mut rng = rand::thread_rng();
        // let org_keys: Box<_> = self.organizations.0.iter().enumerate().map(|a|a.0).collect();
        let num_orgs = self.organizations.0.len();
        for organization_key in 0..num_orgs {
            let organization_key = OrganizationKey(organization_key);
            let enemy_provinces: HashSet<_> = (0..num_orgs)
                .map(|o| OrganizationKey(o))
                .filter(|other_org| {
                    self.relations
                        .get_relations(organization_key, *other_org)
                        .at_war
                })
                .flat_map(|o| {
                    self.organizations[o]
                        .province_control
                        .iter()
                        .filter(|(p_key, c)| c > &&0.1)
                        .map(|(p, c)| p)
                })
                .collect();

            let total_troops = self.organizations[organization_key]
                .military
                .deployed_forces
                .0
                .iter()
                .map(|f| f.num_troops)
                .sum::<f64>()
                + self.organizations[organization_key]
                    .military
                    .forces_in_waiting
                    .num_troops;
            let total_population = self.get_org_population(organization_key);
            const TROOP_RATIO: f64 = 0.005;
            self.organizations[organization_key].building_troops =
                total_troops / total_population < TROOP_RATIO;

            let total_province_weight = self.organizations[organization_key]
                .military
                .province_weights
                .0
                .iter()
                .sum::<f64>();
            if total_province_weight > 0.0 {
                let mut troops_needed = 0.0;
                for province_key in (0..self.provinces.0.len()).map(|p| ProvinceKey(p)) {
                    let troops_in_province = self.organizations[organization_key]
                        .military
                        .deployed_forces[province_key]
                        .num_troops;
                    let province_weight = self.organizations[organization_key]
                        .military
                        .province_weights[province_key];
                    let weight_ratio = province_weight / total_province_weight;

                    let target_troop_amount = weight_ratio * total_troops;
                    let extra_troops = troops_in_province - target_troop_amount;
                    troops_needed += -(extra_troops.min(0.0));
                    let extra_troops = extra_troops.max(0.0);
                    self.organizations[organization_key]
                        .military
                        .deployed_forces[province_key]
                        .num_troops -= extra_troops;
                    self.organizations[organization_key]
                        .military
                        .forces_in_waiting
                        .num_troops += extra_troops;
                }
                if troops_needed > 0.0 {
                    let allocation_ratio = (self.organizations[organization_key]
                        .military
                        .forces_in_waiting
                        .num_troops
                        / troops_needed)
                        .min(1.0);
                    for province_key in (0..self.provinces.0.len()).map(|p| ProvinceKey(p)) {
                        let troops_in_province = self.organizations[organization_key]
                            .military
                            .deployed_forces[province_key]
                            .num_troops;
                        let province_weight = self.organizations[organization_key]
                            .military
                            .province_weights[province_key];
                        let weight_ratio = province_weight / total_province_weight;

                        let target_troop_amount = weight_ratio * total_troops;
                        let added_troops =
                            (target_troop_amount - troops_in_province) * allocation_ratio;
                        self.organizations[organization_key]
                            .military
                            .deployed_forces[province_key]
                            .num_troops += added_troops;
                        self.organizations[organization_key]
                            .military
                            .forces_in_waiting
                            .num_troops -= added_troops;
                    }
                }
            }

            let organization = &mut self.organizations[organization_key];
            // dbg!(organization.owned_goods);

            let mut goods_to_spend = organization.military.forces_in_waiting.supply_goods(
                &organization.owned_goods,
                organization.military.forces_in_waiting.num_troops / total_troops,
            );
            organization
                .military
                .forces_in_waiting
                .consume_supplies(false, delta_year);
            for (_, force) in organization.military.deployed_forces.iter_mut() {
                let troop_ratio = force.num_troops / total_troops;
                let province_access = 1.0;
                // for g in &organization.owned_goods {
                //     assert!(!g.is_nan())
                // }
                for (good_index, good) in force
                    .supply_goods(&organization.owned_goods, troop_ratio * province_access)
                    .into_iter()
                    .enumerate()
                {
                    goods_to_spend[good_index] += good;
                }
                force.consume_supplies(true, delta_year);
                // for g in &organization.owned_goods {
                //     assert!(!g.is_nan())
                // }
                if force.survival_needs_met < 0.5 {
                    force.num_troops -= exponential_decay(force.num_troops, 0.05, delta_year);
                }
            }
            for (good_index, amount) in goods_to_spend.into_iter().enumerate() {
                organization.owned_goods[good_index] -= amount;
                // if organization.owned_goods[good_index] < 0.0 {
                //     dbg!(organization.owned_goods[good_index]);
                // }
                assert!(organization.owned_goods[good_index] >= -0.1);
            }
            // dbg!(organization.owned_goods);

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

                                    let ideology_difference = slice_ideology
                                        .distance(&party.ideology)
                                        + if party.home_org != organization_key {
                                            // party.ideology.distance(&party.ideology) + if party.home_org != organization_key{
                                            1.0
                                        } else {
                                            0.0
                                        };

                                    let variance =
                                        map_range_linear_f64(rng.gen::<f64>(), 0.0, 1.0, 0.9, 1.1);
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
            }
            //veto/approval phase
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

            let province_control_sum = self.organizations[organization_key]
                .province_control
                .0
                .iter()
                .sum::<f64>();
            if province_control_sum > 0.1 {
                let redistribution_amount = exponential_decay(
                    self.organizations[organization_key].money,
                    self.organizations[organization_key].welfare_rate,
                    delta_year,
                );
                self.organizations[organization_key].money -= redistribution_amount;
                if self.organizations[organization_key].money < 0.0 {
                    dbg!(self.organizations[organization_key].money);
                }
                assert!(self.organizations[organization_key].money >= 0.0);

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
            const DIPLO_OFFER_LIFETIME: f64 = 1.0;
            self.organizations[organization_key]
                .diplomatic_offers
                .retain(|o| self.current_year - o.time_sent < DIPLO_OFFER_LIFETIME);
        }
        for (province_key, province) in self.provinces.iter() {
            let province_key = province_key;
            if let Some((biggest_org_key, biggest_organization)) = self
                .organizations
                .0
                .iter_mut()
                .enumerate()
                .map(|(k, o)| (OrganizationKey(k), o))
                .filter(|(k, o)| o.military.deployed_forces[province_key].num_troops > 1.0)
                .max_by_key(|(_org_key, org)| {
                    FloatOrd(org.military.deployed_forces[province_key].num_troops)
                })
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
                .filter(|o| o.1.military.deployed_forces[province_key].num_troops > 0.5)
                .map(|a| a.0)
                .collect();

            let mut combat_orgs_filtered = HashMap::with_capacity(combat_orgs.len());
            // let mut combat_orgs_filtered = Vec::with_capacity(combat_orgs.len());
            for i in 0..combat_orgs.len() {
                for j in 0..combat_orgs.len() {
                    if combat_orgs[i] != combat_orgs[j]
                        && self
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
                .deployed_forces[province_key]
                .num_troops;
            let second_army_size: f64 = self.organizations[second_combatant_key]
                .military
                .deployed_forces[province_key]
                .num_troops;

            let first_army_targets = first_army_size.min(self.provinces[province_key].combat_width);
            let second_army_targets =
                second_army_size.min(self.provinces[province_key].combat_width);

            let seed = hash_usize_fast(
                province_key.0 + (self.current_year * BATTLE_DURATION * 4.0) as usize,
            ) as u64;

            let awake_ratio = 12.0 / 24.0;
            let combat_ratio = 1.0 / 24.0; //0.2 hours of fighting a day
                                           // let reload_modifier = 0.1;
            let aim_accuracy = 0.05;
            let weather_factor = 0.8;
            let spotted_enemies_ratio = 0.01;
            let cover_modifier = 0.9;

            {
                let ammo_supply = self.organizations[second_combatant_key]
                    .military
                    .deployed_forces[province_key]
                    .ammo_needs_met;
                let fire_rate = self.organizations[second_combatant_key]
                    .military
                    .service_rifle
                    .firerate;
                let gun_accuracy = self.organizations[second_combatant_key]
                    .military
                    .service_rifle
                    .accuracy;
                let kill_chance = self.organizations[second_combatant_key]
                    .military
                    .service_rifle
                    .kill_probability;

                // let target_richness = second_army_targets / first_army_targets;

                let shape = 5.0;
                let rate = 2.5;

                let random_bonus = {
                    let mut rng = StdRng::seed_from_u64(seed);
                    let gamma = Gamma::new(shape, rate).unwrap();
                    gamma.sample(&mut rng)
                };

                let losses = second_army_targets
                    * fire_rate
                    * awake_ratio
                    * combat_ratio
                    // * reload_modifier
                    * gun_accuracy
                    * aim_accuracy
                    * weather_factor
                    * kill_chance
                    * spotted_enemies_ratio
                    * cover_modifier
                    * random_bonus
                    * ammo_supply
                    * delta_year;

                // let volume_of_fire = second_army_targets * fire_rate * delta_year * awake_ratio * combat_ratio;
                // let possible_targets = first_army_targets * gun_accuracy * aim_accuracy * weather_factor * spotted_enemies_ratio * cover_modifier;
                // let bullets_per_person = volume_of_fire / possible_targets;
                // let losses = volume_of_fire

                let first_army = &mut self.organizations[first_combatant_key].military;
                first_army.deployed_forces[province_key].num_troops -=
                    losses.min(first_army.deployed_forces[province_key].num_troops);
            }

            {
                let ammo_supply = self.organizations[first_combatant_key]
                    .military
                    .deployed_forces[province_key]
                    .ammo_needs_met;
                let fire_rate = self.organizations[first_combatant_key]
                    .military
                    .service_rifle
                    .firerate;
                let gun_accuracy = self.organizations[first_combatant_key]
                    .military
                    .service_rifle
                    .accuracy;
                let kill_chance = self.organizations[first_combatant_key]
                    .military
                    .service_rifle
                    .kill_probability;

                // let target_richness = first_army_targets / second_army_targets;

                let shape = 5.0;
                let rate = 2.5;

                let random_bonus = {
                    let mut rng = StdRng::seed_from_u64(seed);
                    let gamma = Gamma::new(shape, rate).unwrap();
                    gamma.sample(&mut rng)
                };
                let losses = first_army_targets
                    * fire_rate
                    * awake_ratio
                    * combat_ratio
                    // * reload_modifier
                    * gun_accuracy
                    * aim_accuracy
                    * weather_factor
                    * kill_chance
                    * spotted_enemies_ratio
                    * cover_modifier
                    // * target_richness
                    * random_bonus
                    * ammo_supply
                    * delta_year;

                let second_army = &mut self.organizations[second_combatant_key].military;
                second_army.deployed_forces[province_key].num_troops -=
                    losses.min(second_army.deployed_forces[province_key].num_troops);
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
