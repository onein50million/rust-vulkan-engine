use std::{
    collections::HashMap,
    hash::Hash,
    ops::{Index, IndexMut},
};

use bitflags::bitflags;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use variant_count::VariantCount;

use super::{
    ideology::{Ideology, PoliticalPartyKey},
    Good, ProvinceKey, ProvinceMap,
};

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct OrganizationKey(pub usize);

#[derive(Serialize, Deserialize)]
pub struct Organization {
    pub name: String,
    pub money: f64,
    pub owned_goods: [f64; Good::VARIANT_COUNT],
    pub province_control: ProvinceMap<f64>,
    pub province_approval: ProvinceMap<f64>,
    // pub relations
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
                                    term_length: 2.0 + fastrand::f64(),
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
        let num_branches = branches.len();

        Self {
            name: name.to_string(),
            money: 0.0,
            owned_goods: [0.0; Good::VARIANT_COUNT],
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
        let transfer_amount = self.military.deployed_forces[source_province] * ratio;
        self.military.deployed_forces[destination_province] += transfer_amount;
        self.military.deployed_forces[source_province] -= transfer_amount;
    }
}

// #[derive(Serialize, Deserialize, Debug)]
// pub enum MilitaryType {
//     Enforcer,
//     Army,
// }

#[derive(Serialize, Deserialize)]
pub struct Military {
    pub recruit_ratio: ProvinceMap<f64>,
    // pub military_type: MilitaryType,
    pub deployed_forces: ProvinceMap<f64>,
}
impl Military {
    pub fn new(num_provinces: usize) -> Self {
        Self {
            recruit_ratio: ProvinceMap(vec![0.0; num_provinces].into_boxed_slice()),
            // military_type,
            deployed_forces: ProvinceMap(vec![0.0; num_provinces].into_boxed_slice()),
        }
    }
}
#[derive(Serialize, Deserialize, Debug)]
pub struct Relation {
    pub at_war: bool,
}
