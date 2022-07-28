/*
    Economic Positions
        Tax rate
            Low, Medium, High
        Tax method,
            Regressive, Flat, Progressive
        Import tariffs
            Low, Medium, High
        Export tariffs
            Low, Medium, High
    Social Positions
        Immigration
            None, Only same language, Skilled Labor, Limited, Unlimited
        Assimilation
            None, Encouraged, Systematic, Genocide
        War support
            None, Weak, Medium, Strong, Jingoistic
        Irredentism
            None, Majority language, Minority language, Lebensraum
        Language Support
            Single Superior, Graduated, Equality, Revitalization
    Maybe some constitutional positions that decide how the government functions?


    What determines how people vote?
        material factors
            wealth, met needs
        social factors
            job, class
        beliefs


    Have a set of pseudo questions and pop's responses and weights


    Pop beliefs/hopes/goals/aspirations
        "I want a strong country"
        "I want equality"

        Maybe each one has an associated ideology with them

        Goal with flavor
            Selfish
            Strong Nation
                authoritarian flavor
            Strong economy
                libertarian flavor

            Equality
            Equity
            COMMUNISM
                Vangaurd authoritarian party
                Democracy


*/

use std::ops::{Index, IndexMut};

use serde::{Deserialize, Serialize};

use self::positions::*;

use super::{language::LanguageKey, organization::OrganizationKey, questions::*};

pub mod positions {
    use serde::{Deserialize, Serialize};

    //Economic Postions
    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Research {
        Evil,
        Unimportant,
        Priority,
        Technocratic,
    }
    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TaxRate {
        Low,
        Medium,
        High,
    }

    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TaxMethod {
        Regressive,
        Flat,
        Progressive,
    }
    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ImportDifficulty {
        Trivial,
        Inconvenient,
        Diffiicult,
        Impossible,
    }
    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ExportDifficulty {
        Trivial,
        Inconvenient,
        Diffiicult,
        Impossible,
    }
    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
    //Social Positions
    pub enum Immigration {
        None,
        SameLanguage,
        SkilledLabor,
        Limited,
        Unlimited,
    }
    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Assimilation {
        None,
        Encouraged,
        Systematic,
        Genocide,
    }
    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
    pub enum WarSupport {
        Pacifistic,
        Weak,
        Medium,
        Strong,
        Jingoistic,
    }
    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Irredentism {
        None,
        MajorityLanguage,
        MinorityLanguage,
        Lebensraum,
    }
    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
    pub enum LanguageSupport {
        SingleSuperior,
        Graduated,
        Equality,
        Revitalization,
    }
    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
    pub enum WelfareSupport {
        None,
        PartialNeeds,
        FullNeeds,
        LuxuryNeeds,
    }
    //Constitutional positions
    #[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SeparationOfPowers {
        OneBranch,
        ManyBranches,
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Ideology {
    pub tax_rate: TaxRate,
    pub tax_method: TaxMethod,
    pub import_difficulty: ImportDifficulty,
    pub export_difficulty: ExportDifficulty,
    pub immigration: Immigration,
    pub assimilation: Assimilation,
    pub war_support: WarSupport,
    pub irredentism: Irredentism,
    pub language_support: LanguageSupport,
    pub welfare_support: WelfareSupport,
    pub separation_of_powers: SeparationOfPowers,
    pub research: Research,
    pub primary_language: LanguageKey,
}

impl Ideology {
    pub fn new() -> Self {
        Self {
            tax_rate: TaxRate::Low,
            tax_method: TaxMethod::Flat,
            import_difficulty: ImportDifficulty::Diffiicult,
            export_difficulty: ExportDifficulty::Diffiicult,
            immigration: Immigration::Limited,
            assimilation: Assimilation::Encouraged,
            war_support: WarSupport::Jingoistic,
            irredentism: Irredentism::Lebensraum,
            language_support: LanguageSupport::Equality,
            welfare_support: WelfareSupport::FullNeeds,
            separation_of_powers: SeparationOfPowers::ManyBranches,
            research: Research::Evil,
            primary_language: LanguageKey(0),
        }
    }
    pub fn distance(&self, other: &Ideology) -> f64 {
        let mut out = 0.0;
        if self.tax_rate != other.tax_rate {
            out += 1.0;
        }
        if self.tax_method != other.tax_method {
            out += 1.0;
        }
        if self.import_difficulty != other.import_difficulty {
            out += 1.0;
        }
        if self.export_difficulty != other.export_difficulty {
            out += 1.0;
        }
        if self.immigration != other.immigration {
            out += 1.0;
        }
        if self.assimilation != other.assimilation {
            out += 1.0;
        }
        if self.war_support != other.war_support {
            out += 1.0;
        }
        if self.irredentism != other.irredentism {
            out += 1.0;
        }
        if self.language_support != other.language_support {
            out += 1.0;
        }
        if self.welfare_support != other.welfare_support {
            out += 1.0;
        }
        if self.separation_of_powers != other.separation_of_powers {
            out += 1.0;
        }
        if self.research != other.research {
            out += 1.0;
        }
        if self.primary_language != other.primary_language {
            out += 10.0;
        }
        out
    }
}
#[derive(Serialize, Deserialize, Debug)]
pub struct PoliticalParty {
    pub name: String,
    pub ideology: Ideology,
    pub home_org: OrganizationKey,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct PoliticalPartyKey(pub usize);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PoliticalPartyMap<T>(pub Vec<T>);

impl<T> Index<PoliticalPartyKey> for PoliticalPartyMap<T> {
    type Output = T;

    fn index(&self, index: PoliticalPartyKey) -> &Self::Output {
        &self.0[index.0]
    }
}

impl<T> IndexMut<PoliticalPartyKey> for PoliticalPartyMap<T> {
    fn index_mut(&mut self, index: PoliticalPartyKey) -> &mut Self::Output {
        &mut self.0[index.0]
    }
}

impl<T> PoliticalPartyMap<T> {
    pub fn new() -> Self {
        Self(vec![])
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }
    pub fn into_iter(self) -> impl Iterator<Item = (PoliticalPartyKey, T)> {
        self.0
            .into_iter()
            .enumerate()
            .map(|(key, org)| (PoliticalPartyKey(key), org))
    }
    pub fn iter(&self) -> impl Iterator<Item = (PoliticalPartyKey, &T)> {
        self.0
            .iter()
            .enumerate()
            .map(|(key, org)| (PoliticalPartyKey(key), org))
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (PoliticalPartyKey, &mut T)> {
        self.0
            .iter_mut()
            .enumerate()
            .map(|(key, org)| (PoliticalPartyKey(key), org))
    }
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

fn pick_variant<T: Copy>(value: f64, variants: &[T]) -> T {
    let value = (value + 1.0) / 2.0;
    let max_index = variants.len() as f64 - 1.0;
    let variant_index = (value * max_index).round() as usize;
    variants[variant_index]
}
fn pick_variant_bounded<T>(
    value: f64,
    importance: f64,
    lower_bound: T,
    upper_bound: T,
    max_importance: &mut f64,
) -> Option<T> {
    if importance > *max_importance {
        *max_importance = importance;
        Some(if value > 0.0 {
            upper_bound
        } else {
            lower_bound
        })
    } else {
        None
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Beliefs {
    pub responses: [Response; QUESTIONS.len()],
}
impl Beliefs {
    pub fn get_question<'a>(i: usize) -> &'a Question<'static> {
        &QUESTIONS[i]
    }
    pub fn new() -> Self {
        Self {
            responses: [Response {
                value: 0.0,
                importance: 0.0,
            }; QUESTIONS.len()],
        }
    }
    pub fn new_random() -> Self {
        let rng = fastrand::Rng::new();
        let mut questions = [Response {
            value: 0.0,
            importance: 0.0,
        }; QUESTIONS.len()];
        for response in &mut questions {
            response.value = rng.f64() * 2.0 - 1.0;
            response.importance = rng.f64();
            // response.value = rng.i16(..);
            // response.importance = rng.u16(..);
        }
        Self { responses: questions }
    }
    pub fn to_ideology(&self, majority_language: LanguageKey) -> Ideology {
        let tax_rate;
        let tax_method;
        let import_difficulty;
        let export_difficulty;
        let separation_of_powers;
        let research;

        let mut immigration = Immigration::Limited;
        let mut immigration_max_importance = 0.0;

        let mut assimilation = Assimilation::Encouraged;
        let mut assimilation_max_importance = 0.0;

        let mut war_support = WarSupport::Medium;
        let mut war_max_importance = 0.0;

        let mut irredentism = Irredentism::MajorityLanguage;
        let mut irredentism_max_importance = 0.0;

        let mut language_support = LanguageSupport::Equality;
        let mut language_max_importance = 0.0;

        let welfare_support;

        {
            let response = self.responses[TAX_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            tax_rate = pick_variant(value, &[TaxRate::Low, TaxRate::Medium, TaxRate::High])
        }
        {
            let response = self.responses[TAX_METHOD_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            tax_method = pick_variant(
                value,
                &[
                    TaxMethod::Regressive,
                    TaxMethod::Flat,
                    TaxMethod::Progressive,
                ],
            )
        }
        {
            let response = self.responses[TARIFF_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            (import_difficulty, export_difficulty) = (
                pick_variant(
                    value,
                    &[
                        ImportDifficulty::Trivial,
                        ImportDifficulty::Inconvenient,
                        ImportDifficulty::Diffiicult,
                        ImportDifficulty::Impossible,
                    ],
                ),
                pick_variant(
                    value,
                    &[
                        ExportDifficulty::Trivial,
                        ExportDifficulty::Inconvenient,
                        ExportDifficulty::Diffiicult,
                        ExportDifficulty::Impossible,
                    ],
                ),
            )
        }
        {
            let response = self.responses[SEP_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            separation_of_powers = pick_variant(
                value,
                &[
                    SeparationOfPowers::OneBranch,
                    SeparationOfPowers::ManyBranches,
                ],
            )
        }
        {
            let response = self.responses[WELF_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            welfare_support = pick_variant(
                value,
                &[
                    WelfareSupport::None,
                    WelfareSupport::PartialNeeds,
                    WelfareSupport::FullNeeds,
                    WelfareSupport::LuxuryNeeds,
                ],
            )
        }
        {
            let response = self.responses[RD_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            research = pick_variant(
                value,
                &[
                    Research::Evil,
                    Research::Unimportant,
                    Research::Priority,
                    Research::Technocratic,
                ],
            )
        }
        {
            let response = self.responses[IMM_ED_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            if let Some(bounded) = pick_variant_bounded(
                value,
                importance,
                Immigration::None,
                Immigration::SkilledLabor,
                &mut immigration_max_importance,
            ) {
                immigration = bounded
            }
        }
        {
            let response = self.responses[IMM_LANG_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            if let Some(bounded) = pick_variant_bounded(
                value,
                importance,
                Immigration::None,
                Immigration::SameLanguage,
                &mut immigration_max_importance,
            ) {
                immigration = bounded
            }
        }
        {
            let response = self.responses[IMM_UNL_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            if let Some(bounded) = pick_variant_bounded(
                value,
                importance,
                Immigration::None,
                Immigration::Unlimited,
                &mut immigration_max_importance,
            ) {
                immigration = bounded
            }
        }
        {
            let response = self.responses[ASSIM_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            if let Some(bounded) = pick_variant_bounded(
                value,
                importance,
                Assimilation::None,
                Assimilation::Encouraged,
                &mut assimilation_max_importance,
            ) {
                assimilation = bounded
            }
        }
        {
            let response = self.responses[ASSIM_SYST_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            if let Some(bounded) = pick_variant_bounded(
                value,
                importance,
                Assimilation::None,
                Assimilation::Systematic,
                &mut assimilation_max_importance,
            ) {
                assimilation = bounded
            }
        }
        {
            let response = self.responses[ASSIM_GENO_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            if let Some(bounded) = pick_variant_bounded(
                value,
                importance,
                Assimilation::None,
                Assimilation::Genocide,
                &mut assimilation_max_importance,
            ) {
                assimilation = bounded
            }
        }
        {
            let response = self.responses[WAR_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            if let Some(bounded) = pick_variant_bounded(
                value,
                importance,
                WarSupport::Pacifistic,
                WarSupport::Jingoistic,
                &mut war_max_importance,
            ) {
                war_support = bounded
            }
        }
        {
            let response = self.responses[WAR_LAST_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            if let Some(bounded) = pick_variant_bounded(
                value,
                importance,
                WarSupport::Strong,
                WarSupport::Weak,
                &mut war_max_importance,
            ) {
                war_support = bounded
            }
        }
        {
            let response = self.responses[WAR_NEVER_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            if let Some(bounded) = pick_variant_bounded(
                value,
                importance,
                WarSupport::Strong,
                WarSupport::Pacifistic,
                &mut war_max_importance,
            ) {
                war_support = bounded
            }
        }
        {
            let response = self.responses[IRRED_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            if let Some(bounded) = pick_variant_bounded(
                value,
                importance,
                Irredentism::None,
                Irredentism::MinorityLanguage,
                &mut irredentism_max_importance,
            ) {
                irredentism = bounded
            }
        }
        {
            let response = self.responses[IRRED_LEBEN_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            if let Some(bounded) = pick_variant_bounded(
                value,
                importance,
                Irredentism::None,
                Irredentism::Lebensraum,
                &mut irredentism_max_importance,
            ) {
                irredentism = bounded
            }
        }
        {
            let response = self.responses[LANG_SUP_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            if let Some(bounded) = pick_variant_bounded(
                value,
                importance,
                LanguageSupport::Equality,
                LanguageSupport::SingleSuperior,
                &mut language_max_importance,
            ) {
                language_support = bounded
            }
        }
        {
            let response = self.responses[LANG_EQ_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            if let Some(bounded) = pick_variant_bounded(
                value,
                importance,
                LanguageSupport::SingleSuperior,
                LanguageSupport::Equality,
                &mut language_max_importance,
            ) {
                language_support = bounded
            }
        }
        {
            let response = self.responses[LANG_REV_QUESTION];
            let value = response.get_value();
            let importance = response.get_importance();

            if let Some(bounded) = pick_variant_bounded(
                value,
                importance,
                LanguageSupport::SingleSuperior,
                LanguageSupport::Revitalization,
                &mut language_max_importance,
            ) {
                language_support = bounded
            }
        }
        Ideology {
            tax_rate,
            tax_method,
            import_difficulty,
            export_difficulty,
            immigration,
            assimilation,
            war_support,
            irredentism,
            language_support,
            welfare_support,
            separation_of_powers,
            research,
            primary_language: majority_language,
        }
    }
}
// trait FromBeliefs<T>{
//     fn from_beliefs(&self) -> T;
// }

// impl FromBeliefs<TaxRate> for Beliefs{
//     fn from_beliefs(&self) -> TaxRate{
//         let value = self.questions[TAX_RATE_QUESTION].get_value();
//         if value > 0.25{
//             TaxRate::High
//         }else if value < -0.25{
//             TaxRate::Low
//         }else{
//             TaxRate::Medium
//         }
//     }
// }

// #[derive(Debug, Deserialize, Serialize, Clone, Copy)]
// pub struct Response {
//     value: i16,
//     importance: u16,
// }
// impl Response {
//     pub fn get_value(&self) -> f64 {
//         self.value as f64 / i16::MAX as f64
//     }
//     pub fn get_importance(&self) -> f64 {
//         ((self.importance as f64 / u16::MAX as f64) * 4.0).powf(3.0)
//     }
// }


#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub struct Response {
    value: f64,
    importance: f64,
}
impl Response {
    pub fn get_value(&self) -> f64 {
        self.value
    }
    pub fn get_importance(&self) -> f64 {
        self.importance
    }
}