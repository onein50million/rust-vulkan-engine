use std::{collections::HashMap, hash::Hash};

use nohash_hasher::{BuildNoHashHasher, IsEnabled};
use serde::{Deserialize, Serialize};

use super::{Good, ProvinceKey, ProvinceMap};

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct OrganizationKey(pub u32);
impl IsEnabled for OrganizationKey {}

#[derive(Serialize, Deserialize)]
pub struct Organization {
    pub key: OrganizationKey,
    pub name: String,
    pub money: f64,
    pub ideology: Ideology,
    pub owned_goods: [f64; Good::VARIANT_COUNT],
    pub province_control: HashMap<ProvinceKey, f64, BuildNoHashHasher<usize>>,
    pub province_approval: HashMap<ProvinceKey, f64, BuildNoHashHasher<usize>>,
    pub opinions_on_claims:
        HashMap<OrganizationKey, Vec<(usize, f64)>, BuildNoHashHasher<OrganizationKey>>,
    // pub relations
    pub militaries: Vec<Military>,
}

impl Organization {
    pub fn new(name: &str, num_provinces: usize) -> Self {
        Self {
            key: OrganizationKey(fastrand::u32(..)), //TODO: Actually make collisions impossible
            name: name.to_string(),
            money: 0.0,
            ideology: Ideology {},
            owned_goods: [0.0; Good::VARIANT_COUNT],
            province_control: HashMap::with_capacity_and_hasher(
                num_provinces,
                BuildNoHashHasher::default(),
            ),
            province_approval: HashMap::with_capacity_and_hasher(
                num_provinces,
                BuildNoHashHasher::default(),
            ),
            opinions_on_claims: HashMap::with_hasher(BuildNoHashHasher::default()),
            militaries: vec![],
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum MilitaryType {
    Enforcer,
    Army,
}

#[derive(Serialize, Deserialize)]
pub struct Military {
    pub recruit_ratio: ProvinceMap<f64>,
    pub military_type: MilitaryType,
    pub deployed_forces: ProvinceMap<f64>,
}
impl Military {
    pub fn new(military_type: MilitaryType, province_keys: &[&ProvinceKey]) -> Self {
        Self {
            recruit_ratio: province_keys.iter().map(|&&key| (key, 0.0)).collect(),
            military_type,
            deployed_forces: province_keys.iter().map(|&&key| (key, 0.0)).collect(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Ideology {}

#[derive(Serialize, Deserialize, Debug)]
pub struct Relation {
    pub at_war: bool,
}
