// Battle event log (replay/observation layer)

#[derive(Debug, Clone)]
pub enum Event {
    Switch {
        player: u8,
        from_name: Option<String>,
        to_name: String,
    },
    Move {
        player: u8,
        pokemon_name: String,
        move_name: String,
        damage: u16,
        effectiveness: f32,
        is_crit: bool,
        target_hp_remaining: u16,
    },
    Faint {
        player: u8,
        pokemon_name: String,
    },
    Struggle {
        player: u8,
        pokemon_name: String,
        damage: u16,
    },
    Miss {
        player: u8,
        pokemon_name: String,
        move_name: String,
    },
    StatusMove {
        player: u8,
        pokemon_name: String,
        move_name: String,
    },
    StatusApplied {
        player: u8,
        pokemon_name: String,
        status: String,
    },
    StatusCured {
        player: u8,
        pokemon_name: String,
        status: String,
    },
    StatusPrevented {
        player: u8,
        pokemon_name: String,
        status: String,
        reason: String,
    },
    ResidualDamage {
        player: u8,
        pokemon_name: String,
        status: String,
        damage: u16,
    },
    ConfusionApplied {
        player: u8,
        pokemon_name: String,
    },
    ConfusionHitSelf {
        player: u8,
        pokemon_name: String,
        damage: u16,
    },
    StatChange {
        player: u8,
        pokemon_name: String,
        stat: String,
        stages: i8,
    },
    Heal {
        player: u8,
        pokemon_name: String,
        amount: u16,
        source: String,
    },
    Flinch {
        player: u8,
        pokemon_name: String,
    },
    SpikesSet {
        player: u8,
    },
    SpikesDamage {
        player: u8,
        pokemon_name: String,
        damage: u16,
    },
    ScreenSet {
        player: u8,
        screen: String,
    },
    ScreenExpired {
        player: u8,
        screen: String,
    },
    Protect {
        player: u8,
        pokemon_name: String,
        success: bool,
    },
    LeechSeedApplied {
        player: u8,
        pokemon_name: String,
    },
    LeechSeedDrain {
        player: u8,
        pokemon_name: String,
        damage: u16,
    },
    Phaze {
        player: u8,
        pokemon_name: String,
        forced_in: String,
    },
    Haze {
        player: u8,
    },
    WeatherSet {
        player: u8,
        weather: String,
    },
    WeatherDamage {
        player: u8,
        pokemon_name: String,
        damage: u16,
    },
    WeatherExpired {
        weather: String,
    },
}
