//Territory occupied
//Territory claimed w/claim strength
//Territory owned w/owned area

pub(crate) struct Claim{
    pub(crate) vertex_index: usize,
    pub(crate) strength: f64,
}

pub(crate) struct Owned{
    pub(crate) province_index:usize,
    pub(crate) area: f64,
}

pub(crate) struct Organization{
    pub(crate) name:  String,
    pub(crate) money: f64,
    pub(crate) occupied_territory: Vec<usize>,
    pub(crate) claimed_territory: Vec<Claim>,
}
impl Organization{
    pub(crate) fn new(name: &str, money: f64) -> Self{
         return Self{
             name: name.to_owned(),
             money,
             occupied_territory: vec![],
             claimed_territory: vec![],
         }
    }
}