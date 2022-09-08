


pub mod guns{
    use serde::{Serialize, Deserialize};


    const RPM_TO_RPY: f64 = 60.0 * 24.0 * 365.0;

    #[derive(Serialize, Deserialize, Clone)]
    pub struct ServiceFirearm{
        pub firerate: f64, //rounds per year
        pub kill_probability: f64,
        pub accuracy: f64
    }

    pub const SUBMACHINE_GUN: ServiceFirearm = ServiceFirearm{
        // name: "Submachine Gun",
        firerate: RPM_TO_RPY * 900.0,
        kill_probability: 0.1,
        accuracy: 0.8,
    };
    pub const INTERMEDIATE_AUTO_RIFLE: ServiceFirearm = ServiceFirearm{
        // name: "Intermediate Automatic Rifle",
        firerate: RPM_TO_RPY * 400.0,
        kill_probability: 0.5,
        accuracy: 0.95,
    };
    pub const AUTO_RIFLE: ServiceFirearm = ServiceFirearm{
        // name: "Automatic Rifle",
        firerate: RPM_TO_RPY * 300.0,
        kill_probability: 0.8,
        accuracy: 0.95,
    };
    pub const LARGE_CALIBER_BOLT_RIFLE: ServiceFirearm = ServiceFirearm{
        // name: "Large Caliber Bolt Action",
        firerate: RPM_TO_RPY * 100.0,
        kill_probability: 1.1,
        accuracy: 0.99,
    };

    // pub trait GunComponent{
    //     fn get_name(&self) -> String;
    //     fn get_firerate(&self) -> Firerate;
    //     fn get_kill_probability(&self) -> f64;
    //     fn get_accuracy(&self) -> f64;
    // }

    // impl GunComponent for ServiceFirearm{
    //     fn get_name(&self) -> String {
    //         self.name
    //     }
    //     fn get_firerate(&self) -> Firerate {
    //         self.firerate
    //     }
    //     fn get_kill_probability(&self) -> f64 {
    //         self.kill_probability
    //     }
    //     fn get_accuracy(&self) -> f64 {
    //         self.accuracy
    //     }
    // }

    // #[derive(Serialize, Deserialize)]
    // pub struct Sight<T: GunComponent> {
    //     magnification: f64,
    //     gun_component: T
    // }
    // impl<T: GunComponent> Sight<T>{
    //     pub const fn new(magnification: f64, gun_component: T) -> Self{
    //         Self { magnification, gun_component }
    //     }
    // }

    // impl<T: GunComponent> GunComponent for Sight<T>{
    //     fn get_name(&self) -> String {
    //         format!("{:} with {:.1}X Scope", self.gun_component.get_name(), self.magnification)
    //     }

    //     fn get_firerate(&self) -> Firerate {
    //         Firerate(self.gun_component.get_firerate().0 / self.magnification)
    //     }

    //     fn get_kill_probability(&self) -> f64 {
    //         self.gun_component.get_kill_probability()
    //     }

    //     fn get_accuracy(&self) -> f64 {
    //         let miss_chance = 1.0 - self.gun_component.get_accuracy();
    //         1.0 - miss_chance * self.magnification
    //     }
    // }

}