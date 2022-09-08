use std::{
    collections::HashMap,
    hash::Hash,
    ops::{Index, IndexMut},
};

use bitflags::bitflags;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use variant_count::VariantCount;

use super::{
    ideology::{Ideology, PoliticalPartyKey},
    Good, ProvinceKey, ProvinceMap, equipment::guns::{ServiceFirearm, INTERMEDIATE_AUTO_RIFLE, SUBMACHINE_GUN, AUTO_RIFLE, LARGE_CALIBER_BOLT_RIFLE},
};

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct OrganizationKey(pub usize);

#[derive(Serialize, Deserialize)]
pub struct Organization {
    pub name: String,
    pub money: f64,
    pub owned_goods: [f64; Good::VARIANT_COUNT],
    pub supply: [f64; Good::VARIANT_COUNT],
    pub demand: [f64; Good::VARIANT_COUNT],
    pub province_control: ProvinceMap<f64>,
    pub province_approval: ProvinceMap<f64>,
    pub military: Military,
    pub branches: Vec<BranchKey>,
    pub decisions: [Decision; DecisionCategory::VARIANT_COUNT],
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum Decision {
    None,
    Vetoed,
    SetTaxes(ProvinceKey, f64),
    DeclareWar(OrganizationKey),
    MoveTroops(ProvinceKey, ProvinceKey, f64),
}

// Appointing a party to a branch
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum BranchControlDecision {
    None,
    Vetoed,
    Appoint(PoliticalPartyKey),
}

/*

    Decision Process
        First phase: Decisions are made
        Second phase: Decisons approved or veto'd

    Define branch,
    list powers for branch

    decisions
        - Set tax rate
        - Declare war
        - Move Troops
    decision control -- ctrl_decision
        - enacts
        - approves
    branch control -- ctrl_branch
        - appoints
        - approves

*/

#[repr(usize)]
#[derive(Clone, Copy, VariantCount, Debug, IntoPrimitive, TryFromPrimitive)]
pub enum DecisionCategory {
    SetTaxes,   //keyword: Taxes
    DeclareWar, //keyword: War
    MoveTroops, //keyword: MoveTroops
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct BranchKey(pub usize);

#[derive(Serialize, Deserialize)]
pub struct BranchMap<T>(pub Vec<T>);

impl<T> Index<BranchKey> for BranchMap<T> {
    type Output = T;

    fn index(&self, index: BranchKey) -> &Self::Output {
        &self.0[index.0]
    }
}

impl<T> IndexMut<BranchKey> for BranchMap<T> {
    fn index_mut(&mut self, index: BranchKey) -> &mut Self::Output {
        &mut self.0[index.0]
    }
}

impl<T> BranchMap<T> {
    pub fn new() -> Self {
        Self(vec![])
    }
    pub fn into_iter(self) -> impl Iterator<Item = (BranchKey, T)> {
        self.0
            .into_iter()
            .enumerate()
            .map(|(key, org)| (BranchKey(key), org))
    }
    pub fn iter(&self) -> impl Iterator<Item = (BranchKey, &T)> {
        self.0
            .iter()
            .enumerate()
            .map(|(key, org)| (BranchKey(key), org))
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (BranchKey, &mut T)> {
        self.0
            .iter_mut()
            .enumerate()
            .map(|(key, org)| (BranchKey(key), org))
    }
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }
    pub fn push(&mut self, item: T) -> BranchKey {
        self.0.push(item);
        BranchKey(self.0.len() - 1)
    }
}

bitflags! {
    #[derive(Serialize, Deserialize)]
    pub struct DecisionControlFlag: u32{
        const ENACT = 0b01; //keyword: enacts
        const APPROVE = 0b10; //keyword: approves
    }
}
bitflags! {
    #[derive(Serialize, Deserialize)]
    pub struct BranchControlFlag: u32{
        const APPOINT = 0b01; //keyword: appoints
        const APPROVE = 0b10; //keyword: approves
    }
}
#[derive(Serialize, Deserialize)]
pub struct Branch {
    pub name: String,
    pub decision_control: [DecisionControlFlag; DecisionCategory::VARIANT_COUNT],
    pub branch_control: HashMap<BranchKey, BranchControlFlag>,
    pub last_election: f64,
    pub term_length: f64,
    pub controlling_party: PoliticalPartyKey,
    pub elected: bool,
}

impl Organization {
    pub fn new(
        name: String,
        num_provinces: usize,
        branches: &mut BranchMap<Branch>,
        org_definition: Option<String>,
    ) -> Self {
        let branches = match org_definition {
            Some(org_definition) => {
                let mut identifier_to_branch = HashMap::new();
                for line in org_definition.lines() {
                    if line.is_empty() || line.chars().next() == Some('#') {
                        continue;
                    }
                    let mut words = line.split_whitespace();
                    match words.next().expect("No instruction given") {
                        "def" => {
                            let identifier = words.next().expect("No identifier given");
                            identifier_to_branch
                                .entry(identifier)
                                .or_insert(branches.push(Branch {
                                    name: identifier.to_string(),
                                    decision_control: [DecisionControlFlag::empty();
                                        DecisionCategory::VARIANT_COUNT],
                                    branch_control: HashMap::new(),
                                    last_election: -2.0,
                                    term_length: 2.0 + rand::random::<f64>(),
                                    controlling_party: PoliticalPartyKey(0),
                                    elected: false,
                                }));
                        }
                        "ctrl_decision" => {
                            let identifier = words.next().expect("No source identifier given");
                            let decision_control_type =
                                words.next().expect("No decision control type given");
                            let decision_control_type = match decision_control_type {
                                "enacts" => DecisionControlFlag::ENACT,
                                "approves" => DecisionControlFlag::APPROVE,
                                _ => panic!("Invalid control type"),
                            };
                            let decision_category =
                                match words.next().expect("No decision category given") {
                                    "Taxes" => DecisionCategory::SetTaxes,
                                    "War" => DecisionCategory::DeclareWar,
                                    "MoveTroops" => DecisionCategory::MoveTroops,
                                    _ => panic!("Invalid decision category"),
                                };
                            branches[*identifier_to_branch
                                .get(identifier)
                                .expect("Identifier not found")]
                            .decision_control[decision_category as usize]
                                .set(decision_control_type, true);
                        }
                        "ctrl_branch" => {
                            let identifier = words.next().expect("No source identifier given");
                            let branch_control_type =
                                match words.next().expect("No branch control type given") {
                                    "appoints" => BranchControlFlag::APPOINT,
                                    "approves" => BranchControlFlag::APPROVE,
                                    _ => panic!("Invalid control type"),
                                };
                            let target_branch = identifier_to_branch
                                .get(words.next().expect("No target branch given"))
                                .expect("Target branch not found");
                            if identifier == "$People" {
                                branches[*target_branch].elected = true;
                                continue;
                            }
                            branches[*identifier_to_branch
                                .get(identifier)
                                .expect("Identifier not found")]
                            .branch_control
                            .entry(*target_branch)
                            .or_insert(BranchControlFlag::empty())
                            .set(branch_control_type, true);
                        }
                        _ => panic!("Invalid instruction"),
                    }
                }
                identifier_to_branch.into_values().collect()
            }
            None => vec![],
        };

        Self {
            name: name.to_string(),
            money: 0.0,
            owned_goods: [0.0; Good::VARIANT_COUNT],
            supply: [0.0; Good::VARIANT_COUNT],
            demand: [0.0; Good::VARIANT_COUNT],
            province_control: ProvinceMap(vec![0.0; num_provinces].into_boxed_slice()),
            province_approval: ProvinceMap(vec![0.0; num_provinces].into_boxed_slice()),
            military: Military::new(num_provinces),
            branches,
            decisions: [Decision::None; DecisionCategory::VARIANT_COUNT],
        }
    }

    pub fn transfer_troops(
        &mut self,
        source_province: ProvinceKey,
        destination_province: ProvinceKey,
        ratio: f64,
    ) {
        let transfer_amount = self.military.deployed_forces[source_province].num_troops * ratio;
        self.military.deployed_forces[destination_province].num_troops += transfer_amount;
        self.military.deployed_forces[source_province].num_troops -= transfer_amount;
    }
}

// #[derive(Serialize, Deserialize, Debug)]
// pub enum MilitaryType {
//     Enforcer,
//     Army,
// }

#[derive(Serialize, Deserialize, Clone)]
pub struct MilitaryForce {
    pub num_troops: f64,
    pub survival_supply: f64,    //in person-years
    pub survival_needs_met: f64, //ratio
    pub ammo_supply: f64,        //in person-years of fighting
    pub ammo_needs_met: f64,
}
impl MilitaryForce {
    pub fn empty() -> Self {
        Self {
            num_troops: 0.0,
            survival_supply: 0.0,
            survival_needs_met: 1.0,
            ammo_supply: 0.0,
            ammo_needs_met: 1.0,
        }
    }
    pub fn supply_goods(&mut self, goods: &mut [f64; Good::VARIANT_COUNT], allocation_ratio: f64) {
        let target_survival_supply = 1.0 * self.num_troops * 1.0;
        let target_ammo_supply = 1.0 * self.num_troops;

        let survival_delta = (target_survival_supply - self.survival_supply).max(0.0);
        let survival_needs = {
            let mut needs = std::array::from_fn(|_| 0.0);
            Military::add_survival_needs(survival_delta, &mut needs);
            needs
        };
        let ammo_delta = (target_ammo_supply - self.ammo_supply).max(0.0);
        let ammo_needs = {
            let mut needs = std::array::from_fn(|_| 0.0);
            Military::add_ammo_needs(ammo_delta, &mut needs);
            needs
        };
        let mut min_survival_need = 1.0f64;
        let mut min_ammo_need = 1.0f64;
        for good in (0..Good::VARIANT_COUNT).map(|g| Good::try_from(g).unwrap()) {
            let survival_need = survival_needs[good as usize];
            let ammo_need = ammo_needs[good as usize];

            let total_need = survival_need + ammo_need;
            let available_good = goods[good as usize] * allocation_ratio;

            let supply_ratio = (available_good / total_need).min(1.0);

            if survival_need > 0.0 {
                min_survival_need =
                    min_survival_need.min((available_good * supply_ratio) / survival_need);
            }
            if ammo_need > 0.0 {
                min_ammo_need = min_ammo_need.min((available_good * supply_ratio) / ammo_need);
            }
        }
        for good in (0..Good::VARIANT_COUNT).map(|g| Good::try_from(g).unwrap()) {
            goods[good as usize] -= survival_needs[good as usize] * min_survival_need;
            goods[good as usize] -= ammo_needs[good as usize] * min_ammo_need;
        }
        self.survival_supply += min_survival_need * survival_delta;
        self.ammo_supply += min_ammo_need * ammo_delta;
    }
    pub fn build_troops(
        &mut self,
        goods: &mut [f64; Good::VARIANT_COUNT],
        num_troops: f64,
        allocation_ratio: f64,
    ) -> f64 {
        // if num_troops > 0.0{
        //     dbg!(num_troops);
        // }
        assert!(!num_troops.is_nan());
        let target_equipment_supply = num_troops;
        let equipment_needs = {
            let mut needs = std::array::from_fn(|_| 0.0);
            Military::add_equipment_needs(target_equipment_supply, &mut needs);
            needs
        };

        let mut min_equipment_need = 1.0f64;
        for good in (0..Good::VARIANT_COUNT).map(|g| Good::try_from(g).unwrap()) {
            let equipment_need = equipment_needs[good as usize];
            let available_good = goods[good as usize] * allocation_ratio;

            if equipment_need > 0.0 {
                min_equipment_need = min_equipment_need.min(available_good / equipment_need);
            }
        }
        // if num_troops > 0.0{
        //     dbg!(min_equipment_need);
        // }
        for good in (0..Good::VARIANT_COUNT).map(|g| Good::try_from(g).unwrap()) {
            goods[good as usize] -= equipment_needs[good as usize] * min_equipment_need;
        }
        let added_troops = min_equipment_need * target_equipment_supply;
        self.num_troops += added_troops;
        if added_troops.is_nan() {
            dbg!(goods);
            dbg!(min_equipment_need);
            dbg!(target_equipment_supply);
        }
        assert!(!added_troops.is_nan());
        added_troops
    }
    pub fn consume_supplies(&mut self, fighting: bool, delta_year: f64) {
        let survival_supplies_needed = delta_year * self.num_troops;
        self.survival_needs_met = (self.survival_supply / survival_supplies_needed).min(1.0);
        self.survival_supply -= survival_supplies_needed * self.survival_needs_met;

        let ammo_supplies_needed = if fighting {
            delta_year * self.num_troops
        } else {
            0.0
        };
        self.ammo_needs_met = (self.ammo_supply / ammo_supplies_needed).min(1.0);
        self.ammo_supply -= ammo_supplies_needed * self.ammo_needs_met;
    }
}

#[derive(Serialize, Deserialize)]
pub struct Military {
    pub recruit_ratio: ProvinceMap<f64>,
    pub deployed_forces: ProvinceMap<MilitaryForce>,
    pub equipment_supply: f64, //in persons
    pub service_rifle: ServiceFirearm,
}
impl Military {
    pub fn new(num_provinces: usize) -> Self {
        let mut rng =  rand::thread_rng();
        let service_rifles = &[
            INTERMEDIATE_AUTO_RIFLE,
            SUBMACHINE_GUN,
            AUTO_RIFLE,
            LARGE_CALIBER_BOLT_RIFLE,
        ];
        Self {
            recruit_ratio: ProvinceMap(vec![0.0; num_provinces].into_boxed_slice()),
            // military_type,
            deployed_forces: ProvinceMap(
                vec![MilitaryForce::empty(); num_provinces].into_boxed_slice(),
            ),
            equipment_supply: 0.0,
            service_rifle: service_rifles.choose(&mut rng).unwrap().clone(),
            
        }
    }
    pub fn get_total_needs(&self) -> [f64; Good::VARIANT_COUNT] {
        let num_troops_deployed = self
            .deployed_forces
            .0
            .iter()
            .map(|p| p.num_troops)
            .sum::<f64>();

        let mut needed_goods = std::array::from_fn(|_| 0.0);
        Military::add_survival_needs(num_troops_deployed, &mut needed_goods);
        Military::add_ammo_needs(num_troops_deployed, &mut needed_goods);
        Military::add_equipment_needs(10_000.0, &mut needed_goods);

        needed_goods
    }
    pub fn add_survival_needs(person_years: f64, needed_goods: &mut [f64; Good::VARIANT_COUNT]) {
        needed_goods[Good::ProcessedFood as usize] += person_years * 2.0;
    }
    pub fn add_ammo_needs(person_years: f64, needed_goods: &mut [f64; Good::VARIANT_COUNT]) {
        needed_goods[Good::Ammunition as usize] += person_years * 0.5;
    }
    pub fn add_equipment_needs(persons: f64, needed_goods: &mut [f64; Good::VARIANT_COUNT]) {
        needed_goods[Good::SmallArms as usize] += persons * 0.1;
        needed_goods[Good::HeavyWeaponry as usize] += persons * 0.01;
        needed_goods[Good::Armor as usize] += persons * 0.1;
    }
    pub fn get_met_needs_ratio(
        needs: [f64; Good::VARIANT_COUNT],
        owned_goods: [f64; Good::VARIANT_COUNT],
        allocation_ratio: f64,
        goods_spent: &mut [f64],
    ) -> f64 {
        let mut met_needs_ratio = 1.0;

        for good_index in 0..Good::VARIANT_COUNT {
            let ratio = ((owned_goods[good_index] * allocation_ratio) / needs[good_index]).min(1.0);
            met_needs_ratio *= ratio;
        }
        for good_index in 0..Good::VARIANT_COUNT {
            goods_spent[good_index] += met_needs_ratio * needs[good_index];
        }
        met_needs_ratio
    }
}
#[derive(Serialize, Deserialize, Debug)]
pub struct Relation {
    pub at_war: bool,
}
