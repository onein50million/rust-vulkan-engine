use super::{
    Building, Good, Industry, AGRICULTURE_BUILDINGS, MANUFACTURING_BUILDINGS, MINING_BUILDINGS,
};

pub enum RecipeGood {
    Anything(f64),
    Good {
        good: Good,
        amount: f64,
    },
    RandomRatioGood {
        good: Good,
        weight: f64, //sum of weights is how much is average production
    },
}
impl RecipeGood {
    const fn new(good: Good, amount: f64) -> Self {
        Self::Good { good, amount }
    }
    const fn new_random_good(good: Good, weight: f64) -> Self {
        Self::RandomRatioGood { good, weight }
    }
}

pub struct Recipe<'a> {
    pub name: &'a str,
    pub building: Building,
    //Inputs and outputs in goods per year
    pub inputs: &'a [RecipeGood],
    pub outputs: &'a [RecipeGood],
}
impl<'a> Recipe<'a> {
    const fn new(
        name: &'a str,
        building: Building,
        inputs: &'a [RecipeGood],
        outputs: &'a [RecipeGood],
    ) -> Self {
        Self {
            name,
            building,
            inputs,
            outputs,
        }
    }
}

pub const fn get_recipe_count(building: Building) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < RECIPES.len() {
        if RECIPES[i].building as usize == building as usize {
            count += 1;
        }
        i += 1;
    }
    count
}

pub const fn get_enabled_buildings<'a>(industry: Industry) -> &'a [Building] {
    match industry {
        Industry::Agriculture => AGRICULTURE_BUILDINGS,
        Industry::Mining => MINING_BUILDINGS,
        Industry::Manufacturing => MANUFACTURING_BUILDINGS,
        Industry::GeneralLabor => &[],
        Industry::Unemployed => &[],
    }
}

pub const RECIPES: &[Recipe] = &[
    Recipe::new(
        "Vegetable Farming",
        Building::ProduceFarm,
        &[],
        &[
            RecipeGood::new(Good::Produce, 100.0),
            RecipeGood::new(Good::ProcessedFood, 10.0),
        ],
    ),
    Recipe::new(
        "Livestock Farming",
        Building::LivestockFarm,
        &[],
        &[RecipeGood::new(Good::AnimalProducts, 25.0)],
    ),
    Recipe::new(
        "Iron Mining",
        Building::IronMine,
        &[],
        &[RecipeGood::new(Good::Iron, 1.0)],
    ),
    Recipe::new(
        "Non-ferrous Metal Mining",
        Building::NonFerrousMetalMine,
        &[],
        &[RecipeGood::new(Good::NonFerrousMetal, 1.0)],
    ),
    Recipe::new(
        "Silica Quarry",
        Building::SilicaQuarry,
        &[],
        &[RecipeGood::new(Good::Silica, 1.0)],
    ),
    Recipe::new(
        "Hydrocarbon Extraction",
        Building::FossilFuelExtraction,
        &[],
        &[RecipeGood::new(Good::RawHydrocarbons, 1.0),RecipeGood::new(Good::Carbon, 0.1),],
    ),
    Recipe::new(
        "Produce Processing",
        Building::FoodProcessor,
        &[RecipeGood::new(Good::Produce, 100.0)],
        &[RecipeGood::new(Good::ProcessedFood, 100.0)],
    ),
    Recipe::new(
        "Meat Processing",
        Building::FoodProcessor,
        &[RecipeGood::new(Good::AnimalProducts, 5.0)],
        &[RecipeGood::new(Good::ProcessedFood, 20.0)],
    ),
    Recipe::new(
        "Steel Production",
        Building::Metalworks,
        &[
            RecipeGood::new(Good::Iron, 5.0),
            RecipeGood::new(Good::Carbon, 0.1),
        ],
        &[RecipeGood::new(Good::Steel, 5.1)],
    ),
    Recipe::new(
        "Stable Superheavy Element Production",
        Building::ParticleCollider,
        &[RecipeGood::Anything(100.0)],
        &[RecipeGood::new(Good::StableSuperheavyElement, 1.0)],
    ),
    Recipe::new(
        "Composites Combination",
        Building::CompositesFacility,
        &[
            RecipeGood::new(Good::Steel, 1.0 / 4.0),
            RecipeGood::new(Good::NonFerrousMetal, 1.0 / 4.0),
            RecipeGood::new(Good::Plastics, 1.0 / 4.0),
            RecipeGood::new(Good::Silica, 1.0 / 4.0),
        ],
        &[RecipeGood::new(Good::Composites, 1.0)],
    ),
    Recipe::new(
        "Plastic Production",
        Building::ChemicalPlant,
        &[RecipeGood::new(Good::RawHydrocarbons, 1.0)],
        &[RecipeGood::new(Good::Plastics, 5.0)],
    ),
    Recipe::new(
        "Battery Production",
        Building::ChemicalPlant,
        &[
            RecipeGood::new(Good::NonFerrousMetal, 0.1),
            RecipeGood::new(Good::Carbon, 0.9),
        ],
        &[RecipeGood::new(Good::Batteries, 1.0)],
    ),
    Recipe::new(
        "Explosives Production",
        Building::ChemicalPlant,
        &[RecipeGood::new(Good::RawHydrocarbons, 1.0)],
        &[RecipeGood::new(Good::Explosives, 1.0)],
    ),
    Recipe::new(
        "Semiconductor Wafer",
        Building::ChemicalPlant,
        &[
            RecipeGood::new(Good::Silica, 10.0),
            RecipeGood::new(Good::NonFerrousMetal, 0.1),
        ],
        &[RecipeGood::new(Good::Semiconductor, 1.0)],
    ),
    Recipe::new(
        "Electronics production",
        Building::SemiconductorFab,
        &[RecipeGood::new(Good::Semiconductor, 1.0)],
        &[
            RecipeGood::new_random_good(Good::LowGradeElectronics, 2.0),
            RecipeGood::new_random_good(Good::HighGradeElectronics, 1.0),
        ],
    ),
    Recipe::new(
        "Rifle production",
        Building::ArmsFactory,
        &[
            RecipeGood::new(Good::Steel, 0.85),
            RecipeGood::new(Good::Plastics, 0.1),
            RecipeGood::new(Good::NonFerrousMetal, 0.05),
        ],
        &[RecipeGood::new(Good::SmallArms, 1.0)],
    ),
    Recipe::new(
        "Artillery production",
        Building::ArmsFactory,
        &[
            RecipeGood::new(Good::Steel, 0.085),
            RecipeGood::new(Good::Plastics, 0.01),
            RecipeGood::new(Good::NonFerrousMetal, 0.005),
        ],
        &[RecipeGood::new(Good::HeavyWeaponry, 0.1)],
    ),
    Recipe::new(
        "Ammo Production",
        Building::ArmsFactory,
        &[
            RecipeGood::new(Good::Steel, 10.0),
            RecipeGood::new(Good::Explosives, 20.0),
        ],
        &[RecipeGood::new(Good::Ammunition, 30.0)],
    ),
    Recipe::new(
        "Armor Production",
        Building::ArmsFactory,
        &[RecipeGood::new(Good::Composites, 1.0)],
        &[RecipeGood::new(Good::Armor, 1.0)],
    ),
];
