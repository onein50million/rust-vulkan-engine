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
    pub province_control: ProvinceMap<f64>,
    pub province_approval: ProvinceMap<f64>,
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
            province_control: ProvinceMap(vec![0.0; num_provinces].into_boxed_slice()),
            province_approval: ProvinceMap(vec![0.0; num_provinces].into_boxed_slice()),
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
    pub fn new(military_type: MilitaryType, num_provinces: usize) -> Self {
        Self {
            recruit_ratio: ProvinceMap(vec![0.0; num_provinces].into_boxed_slice()),
            military_type,
            deployed_forces: ProvinceMap(vec![0.0; num_provinces].into_boxed_slice()),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Ideology {}

#[derive(Serialize, Deserialize, Debug)]
pub struct Relation {
    pub at_war: bool,
}
