use std::ops::{Index, IndexMut};
use strum::{EnumCount, IntoEnumIterator};

#[derive(Debug, strum_macros::EnumCount, strum_macros::EnumIter, Clone, Copy)]
pub(crate) enum Good{
    Food,
    RawMetal,
    RefinedMetal,
    Fuel
}
#[derive(Copy,Clone,Debug)]
pub(crate) struct GoodAmounts{
    pub(crate) goods: [f64; Good::COUNT]
}
impl GoodAmounts{
    pub(crate) fn new() -> Self{
        let goods = [0.0; Good::COUNT];
        return Self{
            goods
        };

    }
}
impl Index<Good> for GoodAmounts{
    type Output = f64;

    fn index(&self, index: Good) -> &Self::Output {
        &self.goods[index as usize]
    }
}
impl IndexMut<Good> for GoodAmounts{
    fn index_mut(&mut self, index: Good) -> &mut Self::Output {
        &mut self.goods[index as usize]
    }
}

#[derive(Copy,Clone)]
pub(crate) struct MarketGood{
    pub(crate) supply: f64,
    pub(crate) demand: f64,
    pub(crate) price: f64,
}

pub(crate) struct Market{
    goods: [MarketGood; Good::COUNT]
}
impl Market{
    pub(crate) fn new()-> Self{
        Self{
            goods: [
                MarketGood{
                    supply: 0.0,
                    demand: 0.0,
                    price: 1.0,
                }; Good::COUNT
            ]
        }
    }
}

impl Index<Good> for Market{
    type Output = MarketGood;

    fn index(&self, index: Good) -> &Self::Output {
        &self.goods[index as usize]
    }
}
impl IndexMut<Good> for Market{
    fn index_mut(&mut self, index: Good) -> &mut Self::Output {
        &mut self.goods[index as usize]
    }
}
