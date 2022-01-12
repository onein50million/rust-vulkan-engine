
//Identifiers: Job, Education //Each combination is a separate index
//Data: Wealth, Population count,Language literacy,

use std::collections::HashMap;
//Organizations:
//ngos, governments, corporations, etc
use std::mem::size_of;
use std::ops::{Index, IndexMut};

use strum::{EnumCount, IntoEnumIterator};
use crate::market::{Good, GoodAmounts};
use crate::RatioMap;

enum Language{
    English,
    French,
    German,
    Russian,
    Mandarin,
    Australian,
}

// const BITS_PER_LANGAUGE: usize = 2;
// const NUM_BYTES_NEEDED: usize = ((BITS_PER_LANGAUGE as f64 * NUM_LANGUAGES as f64) / 8.0).ceil() as usize;
// struct Proficiency([u8;NUM_BYTES_NEEDED]);
// enum ProficiencyLevel{
//     None,
//     Intelligible,
//     Fluent,
//     Expert
// }
// impl Index<Language> for Proficiency{
//     type Output = ProficiencyLevel;
//
//     fn index(&self, index: Language) -> &Self::Output {
//         let (byte_index, bits_to_shift) = match index{
//             Language::English => (0,BITS_PER_LANGAUGE*0),
//             Language::French => (0,BITS_PER_LANGAUGE*1),
//             Language::German => (0,BITS_PER_LANGAUGE*2),
//             Language::Russian => (0,BITS_PER_LANGAUGE*3),
//             Language::Mandarin => (1,BITS_PER_LANGAUGE*0),
//             Language::Australian => (1,BITS_PER_LANGAUGE*1),
//         };
//         let byte = (self.0[byte_index] >> bits_to_shift) & 0b11;
//         &match byte {
//             0 => { ProficiencyLevel::None },
//             1 => { ProficiencyLevel::Intelligible },
//             2 => { ProficiencyLevel::Fluent },
//             3 => { ProficiencyLevel::Expert },
//             _ => {panic!("Byte doesn't match any proficiency level: {:b}", byte)}
//         }
//     }
// }



//North American equivalent education level

#[derive(Debug, Copy, Clone,strum_macros::EnumIter, strum_macros::EnumCount)]
pub(crate) enum Education{
    None,
    Primary, //Primary school
    Secondary, //High school
    Tertiary, //University
    PostTertiary //like post graduates and doctors and stuff
}

#[derive(strum_macros::EnumIter, strum_macros::EnumCount, Debug)]
pub(crate) enum Job {
    Unemployed,
    Educator,
    Researcher,
    ResourceAcquisition,
    Security,
    Bureaucrat,
}

#[derive(Debug)]
pub(crate)struct PopSlice{
    pub(crate) population: f64,
    pub(crate) money: f64,
    pub(crate) good_amounts: GoodAmounts,
    pub(crate) organization_ratios: RatioMap<usize>
}
impl PopSlice{
    pub(crate) fn new(population: f64, money: f64) -> Self{
        Self{
            population,
            money,
            good_amounts: GoodAmounts::new(),
            organization_ratios: RatioMap::new()
        }
    }
    pub(crate) fn get_need(&self, good: Good) -> f64{
        return match good{
            Good::Food => {self.population * 1.0}
            Good::RawMetal => {0.0}
            Good::RefinedMetal => {0.0}
            Good::Fuel => {0.0}
        }
    }
}

#[derive(Debug)]
pub(crate) struct Identifier{
    pub(crate) education: Education,
    pub(crate) job: Job,
}

impl Identifier{
    fn index(&self)-> usize{
        let education_index = match self.education{
            Education::None => {0}
            Education::Primary => {1}
            Education::Secondary => {2}
            Education::Tertiary => {3}
            Education::PostTertiary => {4}
        };
        let job_index = match self.job{
            Job::Unemployed => {0}
            Job::Educator => {1}
            Job::Researcher => {2}
            Job::ResourceAcquisition => {3}
            Job::Security => {4}
            Job::Bureaucrat => {5}
        };

        return education_index * Job::COUNT + job_index;
    }
}

const SLICE_COUNT: usize = Education::COUNT * Job::COUNT;

pub(crate) struct PopSlices{
     slices: [PopSlice; SLICE_COUNT]
}

impl PopSlices{
    pub(crate) fn new(total_population: f64, total_money: f64)-> Self{
        let per_slice_population = total_population / SLICE_COUNT as f64;
        let per_slice_money = total_money / SLICE_COUNT as f64;

        //Can't think of a better way to do this. Trying to create an array of something that doesn't implement copy
        //Macros don't work because I'm using a const
        //Maybe a build script might work
        let slices = [
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
            PopSlice::new(per_slice_population, per_slice_money),
        ];

        let output = Self{
            slices,
        };
        return output;
    }
    pub(crate) fn process(&mut self, delta: f64){
        for slice in &mut self.slices{
            let food_ratio = slice.good_amounts[Good::Food] / slice.get_need(Good::Food);
            let food_ratio = food_ratio.clamp(0.01, 2.0);

            slice.good_amounts[Good::Food] -= slice.good_amounts[Good::Food] * food_ratio;

            let curve =
                0.001/
                (
                    -f64::powf(food_ratio, 2.0)
                ) + 0.001;
            // dbg!(curve);
            let population_change = slice.population * curve * delta;
            let population_change = population_change.max(-slice.population);
            slice.population += population_change;
        }
    }
    pub(crate) fn get_good_amount(&self, good: Good) -> f64 {
        let mut output = 0.0;
        for slice in &self.slices{
            output += slice.good_amounts[good];
        }
        return output;
    }
    pub(crate) fn get_need_amount(&self, good: Good) -> f64 {
        let mut output = 0.0;
        for slice in &self.slices{
            output += slice.get_need(good);
        }
        return output;
    }
}

impl Index<Identifier> for PopSlices{
    type Output = PopSlice;

    fn index(&self, identifier: Identifier) -> &Self::Output {
        return &self.slices[identifier.index()];
    }
}

impl IndexMut<Identifier> for PopSlices{
    fn index_mut(&mut self, identifier: Identifier) -> &mut Self::Output {
        return &mut self.slices[identifier.index()];
    }
}