pub struct Question<'a> {
    pub question: &'a str,
    pub lower_bound_meaning: &'a str,
    pub upper_bound_meaning: &'a str,
}

pub const TAX_QUESTION: usize = 0;
pub const TAX_METHOD_QUESTION: usize = 1;
pub const TARIFF_QUESTION: usize = 2;
pub const IMM_ED_QUESTION: usize = 3;
pub const IMM_LANG_QUESTION: usize = 4;
pub const IMM_UNL_QUESTION: usize = 5;
pub const ASSIM_QUESTION: usize = 6;
pub const ASSIM_SYST_QUESTION: usize = 7;
pub const ASSIM_GENO_QUESTION: usize = 8;
pub const WAR_QUESTION: usize = 9;
pub const WAR_LAST_QUESTION: usize = 10;
pub const WAR_NEVER_QUESTION: usize = 11;
pub const IRRED_QUESTION: usize = 12;
pub const IRRED_LEBEN_QUESTION: usize = 13;
pub const LANG_SUP_QUESTION: usize = 14;
pub const LANG_EQ_QUESTION: usize = 15;
pub const LANG_REV_QUESTION: usize = 16;
pub const SEP_QUESTION: usize = 17;
pub const WELF_QUESTION: usize = 18;
pub const RD_QUESTION: usize = 19;
pub const INDUSTRY_QUESTION: usize = 20;

pub const QUESTIONS: &[Question] = &[
Question{question:"The tax rate should be...", lower_bound_meaning:"lower", upper_bound_meaning:"higher"},
Question{question:"Should the rich be taxed at a higher or lower rate than the poor?", lower_bound_meaning:"more regressive", upper_bound_meaning:"more progressive"},
Question{question:"Tariffs should be...", lower_bound_meaning:"lower", upper_bound_meaning:"higher"},
Question{question:"Educated people should be allowed to immigrate", lower_bound_meaning:"disagree", upper_bound_meaning:"agree"},
Question{question:"People who speak our language should be allowed to immigrate", lower_bound_meaning:"disagree", upper_bound_meaning:"agree"},
Question{question:"All people should be allowed to immigrate without limits", lower_bound_meaning:"decreased", upper_bound_meaning:"increased"},
Question{question:"Do you believe that people should learn the majority language?", lower_bound_meaning:"no", upper_bound_meaning:"yes"},
Question{question:"Do you think the government should benefit people who speak the majority language?", lower_bound_meaning:"no", upper_bound_meaning:"yes"},
Question{question:"Do you believe that people who refuse to learn the majority language should be punished?", lower_bound_meaning:"no", upper_bound_meaning:"yes"},
Question{question:"War is a valid method of gaining power", lower_bound_meaning:"disagree", upper_bound_meaning:"agree"},
Question{question:"War must be should be a last resort", lower_bound_meaning:"disagree", upper_bound_meaning:"agree"},
Question{question:"War must be avoided at all costs", lower_bound_meaning:"disagree", upper_bound_meaning:"agree"},
Question{question:"All people who speak our langauge should be a part of our country", lower_bound_meaning:"disagree", upper_bound_meaning:"agree"},
Question{question:"Our people need territory to expand into", lower_bound_meaning:"", upper_bound_meaning:""},
Question{question:"The language I speak is superior to others", lower_bound_meaning:"disagree", upper_bound_meaning:"agree"},
Question{question:"All languages are valid forms of communication", lower_bound_meaning:"disagree", upper_bound_meaning:"agree"},
Question{question:"We should work to revitalize dead and dying languages", lower_bound_meaning:"disagree", upper_bound_meaning:"agree"},
Question{question:"my leaders should be accountable", lower_bound_meaning:"disagree", upper_bound_meaning:"agree"},
Question{question:"Others basic needs are more imporant than my luxury needs", lower_bound_meaning:"disagree", upper_bound_meaning:"agree"},
Question{question:"How important is funding research and development", lower_bound_meaning:"not very", upper_bound_meaning:"very"},
Question{question:"How important is funding industry", lower_bound_meaning:"not very", upper_bound_meaning:"very"}];
