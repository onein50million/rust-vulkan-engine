use std::ops::{Index, IndexMut};

use serde::{Deserialize, Serialize};

use super::ProvinceMap;

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct LanguageKey(pub usize);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LanguageMap<T>(pub Vec<T>);

impl<T> Index<LanguageKey> for LanguageMap<T> {
    type Output = T;

    fn index(&self, index: LanguageKey) -> &Self::Output {
        &self.0[index.0]
    }
}

impl<T> IndexMut<LanguageKey> for LanguageMap<T> {
    fn index_mut(&mut self, index: LanguageKey) -> &mut Self::Output {
        &mut self.0[index.0]
    }
}

impl<T> LanguageMap<T> {
    pub fn into_iter(self) -> impl Iterator<Item = (LanguageKey, T)> {
        self.0
            .into_iter()
            .enumerate()
            .map(|(key, org)| (LanguageKey(key), org))
    }
    pub fn iter(&self) -> impl Iterator<Item = (LanguageKey, &T)> {
        self.0
            .iter()
            .enumerate()
            .map(|(key, org)| (LanguageKey(key), org))
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (LanguageKey, &mut T)> {
        self.0
            .iter_mut()
            .enumerate()
            .map(|(key, org)| (LanguageKey(key), org))
    }
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }
}
#[derive(Serialize, Deserialize, Debug)]
pub struct LanguageManager {
    pub languages: LanguageMap<Language>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Language {
    pub name: String,
    pub province_presence: ProvinceMap<f64>,
    // pub parent: LanguageKey,
    // pub children: Vec<LanguageKey>,
}
