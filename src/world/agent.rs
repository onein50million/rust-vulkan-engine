use serde::{Deserialize, Serialize};

use super::{
    ideology::{
        positions::{IndustrySupport, TaxRate},
        PoliticalParty, PoliticalPartyMap,
    },
    organization::{
        Branch, BranchMap, DecisionCategory, DecisionControlFlag, DiplomaticAction,
        DiplomaticOfferType, Organization, OrganizationKey,
    },
    recipes::get_enabled_buildings,
    Building, OrganizationMap, Province, ProvinceKey, ProvinceMap, RelationMap, World,
};

//Agent controls an organization
//Could be a player agent or an AI agent
pub struct Agent {
    pub controlling_organization: Option<OrganizationKey>,
    pub political_power: f64,
    pub targeted_province: Option<ProvinceKey>,
}
impl Agent {
    pub fn new(controlling_organization: Option<OrganizationKey>) -> Self {
        Self {
            controlling_organization,
            political_power: 0.0,
            targeted_province: None,
            
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AgentAction {
    SetResearch, //TODO: Requires research system to be implemented
    ModifyBuilding(ProvinceKey, Building, f64),
    SetTaxes(ProvinceKey, f64),
    SetWelfare(f64),
    MoveTroops {
        source: ProvinceKey,
        dest: ProvinceKey,
        ratio: f64,
    },
    SetTroopWeight {
        province: ProvinceKey,
        weight: f64,
    },
    /// Target Org Key and action
    DiplomaticAction(OrganizationKey, DiplomaticAction),
    EventResponse,                         //TODO: Requires event system
    RespondToDiplomaticOffer(usize, bool), //index of offer and whether bool for acceptance/denial
}
impl AgentAction {
    pub fn get_cost(&self, controlling_organization: OrganizationKey, world: &World) -> f64 {
        let mut cost = 0.0f64;

        for branch in &world.organizations[controlling_organization].branches {
            let branch = &world.branches[*branch];
            let ideology = &world.political_parties[branch.controlling_party].ideology;
            match self {
                AgentAction::SetResearch => {}
                AgentAction::ModifyBuilding(_, _, delta) => {
                    if !branch.decision_control[DecisionCategory::Industry as usize]
                        .intersects(DecisionControlFlag::ENACT | DecisionControlFlag::APPROVE)
                    {
                        continue;
                    }
                    let delta_cost = if *delta > 0.0 { 10.0 } else { -10.0 };
                    if matches!(ideology.industry, IndustrySupport::Agrarian) {
                        cost += delta_cost * 1.0;
                    } else if matches!(ideology.industry, IndustrySupport::Industrial) {
                        cost -= delta_cost * 1.0;
                    } else if matches!(ideology.industry, IndustrySupport::FactoryMustGrow) {
                        cost -= delta_cost * 5.0;
                    }
                }
                AgentAction::SetTaxes(province, new_rate) => {
                    if !branch.decision_control[DecisionCategory::Economy as usize]
                        .intersects(DecisionControlFlag::ENACT | DecisionControlFlag::APPROVE)
                    {
                        continue;
                    }
                    let old_rate = world.provinces[*province].tax_rate;
                    let delta_cost = if *new_rate > old_rate { 10.0 } else { -10.0 };
                    match ideology.tax_rate {
                        TaxRate::Low => cost += delta_cost,
                        TaxRate::Medium => {}
                        TaxRate::High => cost -= delta_cost,
                    }
                }
                AgentAction::SetWelfare(welfare) => {
                    if !branch.decision_control[DecisionCategory::Economy as usize]
                        .intersects(DecisionControlFlag::ENACT | DecisionControlFlag::APPROVE)
                    {
                        continue;
                    }
                    let old_welfare = world.organizations[controlling_organization].welfare_rate;
                    let delta_cost = if *welfare > old_welfare { 10.0 } else { -10.0 };
                    match ideology.welfare_support {
                        super::ideology::positions::WelfareSupport::None => {
                            cost += delta_cost * 2.0
                        }
                        super::ideology::positions::WelfareSupport::PartialNeeds => {
                            cost += delta_cost * 1.0
                        }
                        super::ideology::positions::WelfareSupport::FullNeeds => {
                            cost -= delta_cost * 1.0
                        }
                        super::ideology::positions::WelfareSupport::LuxuryNeeds => {
                            cost -= delta_cost * 2.0
                        }
                    }
                }
                AgentAction::MoveTroops { .. } => {}
                AgentAction::SetTroopWeight { .. } => {}
                AgentAction::DiplomaticAction(_, diplo_action) => {
                    if !branch.decision_control[DecisionCategory::Diplomacy as usize]
                        .intersects(DecisionControlFlag::ENACT | DecisionControlFlag::APPROVE)
                    {
                        continue;
                    }
                    let war_cost = match diplo_action {
                        DiplomaticAction::DeclareWar => 10.0,
                        DiplomaticAction::OfferPeace => -10.0,
                    };
                    match ideology.war_support {
                        super::ideology::positions::WarSupport::Pacifistic => cost += war_cost,
                        super::ideology::positions::WarSupport::Weak => cost -= war_cost * 1.0,
                        super::ideology::positions::WarSupport::Medium => cost -= war_cost * 2.0,
                        super::ideology::positions::WarSupport::Strong => cost -= war_cost * 3.0,
                        super::ideology::positions::WarSupport::Jingoistic => {
                            cost -= war_cost * 10.0
                        }
                    }
                }
                AgentAction::EventResponse => {}
                AgentAction::RespondToDiplomaticOffer(_, _) => {}
            }
        }

        cost.max(0.0)
    }
}

impl Agent {
    pub fn attempt_action(
        &mut self,
        world: &mut World,
        action: AgentAction,
    ) -> Result<(), &'static str> {
        if let Some(controlling_organization) = self.controlling_organization {
            let cost = action.get_cost(controlling_organization, world);
            if cost <= self.political_power {
                self.political_power -= cost;
                // let org = &mut organizations[self.controlling_organization];
                match action {
                    AgentAction::SetResearch => todo!(),
                    AgentAction::ModifyBuilding(
                        province,
                        targeted_building,
                        building_size_delta,
                    ) => {
                        if world.organizations[controlling_organization].province_control[province]
                            < 0.5
                        {
                            return Err("You don't have control of the province");
                        }
                        let province = &mut world.provinces[province];
                        let industry = targeted_building.get_industry();
                        let industry_size = province.industry_data[industry as usize].size;
                        let new_industry_size = industry_size + building_size_delta;
                        for building in get_enabled_buildings(industry) {
                            let old_size = province.building_ratios[*building as usize]
                                * industry_size
                                + if *building == targeted_building {
                                    building_size_delta
                                } else {
                                    0.0
                                };
                            province.building_ratios[*building as usize] =
                                old_size / new_industry_size;
                        }
                        province.industry_data[industry as usize].size = new_industry_size;
                        assert!(
                            (get_enabled_buildings(industry)
                                .iter()
                                .map(|&b| province.building_ratios[b as usize])
                                .sum::<f64>()
                                - 1.0)
                                .abs()
                                < 0.01
                        );
                    }
                    AgentAction::SetTaxes(province, new_rate) => {
                        if world.organizations[controlling_organization].province_control[province]
                            < 0.5
                        {
                            return Err("You don't have control of the province");
                        }
                        let province = &mut world.provinces[province];
                        province.tax_rate = new_rate;
                    }
                    AgentAction::SetWelfare(new_welfare) => {
                        world.organizations[controlling_organization].welfare_rate = new_welfare;
                    }
                    AgentAction::MoveTroops {
                        source,
                        dest,
                        ratio,
                    } => {
                        world.organizations[controlling_organization]
                            .transfer_troops(source, dest, ratio);
                    }
                    AgentAction::SetTroopWeight { province, weight } => {
                        world.organizations[controlling_organization]
                            .military
                            .province_weights[province] = weight;
                    }
                    AgentAction::DiplomaticAction(target_org, diplo_action) => match diplo_action {
                        DiplomaticAction::DeclareWar => {
                            world
                                .relations
                                .get_relations_mut(controlling_organization, target_org)
                                .at_war = true
                        }
                        DiplomaticAction::OfferPeace => {
                            world.push_diplomatic_offer(
                                controlling_organization,
                                target_org,
                                DiplomaticOfferType::Peace,
                            );
                        }
                    },
                    AgentAction::EventResponse => todo!(),
                    AgentAction::RespondToDiplomaticOffer(index, response) => {
                        let offer = world.organizations[controlling_organization]
                            .diplomatic_offers
                            .remove(index);
                        match offer.offer_type {
                            DiplomaticOfferType::Peace => {
                                if response {
                                    world
                                        .relations
                                        .get_relations_mut(controlling_organization, offer.from)
                                        .at_war = false
                                }
                            }
                        }
                    }
                }
                Ok(())
            } else {
                Err("Out of Polical Power ⚖")
            }
        } else {
            Err("You are not in control of a organization")
        }
    }
}
