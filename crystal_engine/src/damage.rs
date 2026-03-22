// Gen 2 damage formula (integer math)

use rand::Rng;

use crate::moves::{DamageClass, MoveTemplate};
use crate::player::SideConditions;
use crate::pokemon::Pokemon;
use crate::stat_stages::get_stage_multiplier;
use crate::status::Status;
use crate::types::{self, Type};

/// Weather state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weather {
    Sun,
    Rain,
    Sandstorm,
}

impl Weather {
    pub fn as_str(self) -> &'static str {
        match self {
            Weather::Sun => "sun",
            Weather::Rain => "rain",
            Weather::Sandstorm => "sandstorm",
        }
    }

    pub fn from_str(s: &str) -> Option<Weather> {
        match s {
            "sun" => Some(Weather::Sun),
            "rain" => Some(Weather::Rain),
            "sandstorm" => Some(Weather::Sandstorm),
            _ => None,
        }
    }
}

/// Result of damage calculation
pub struct DamageResult {
    pub damage: u16,
    pub effectiveness: f32,
    pub is_crit: bool,
}

/// Gen 2 damage formula. Returns (damage, effectiveness, is_crit).
pub fn calc_damage(
    attacker: &Pokemon,
    defender: &Pokemon,
    mv: &MoveTemplate,
    rng: &mut impl Rng,
    screens: Option<&SideConditions>,
    weather: Option<Weather>,
) -> DamageResult {
    if mv.damage_class == DamageClass::Status || mv.power == 0 {
        return DamageResult {
            damage: 0,
            effectiveness: 1.0,
            is_crit: false,
        };
    }

    let level: u32 = 100;

    // select stats based on damage class
    let (atk_idx, dfn_idx) = match mv.damage_class {
        DamageClass::Physical => (1usize, 2usize), // attack, defense
        DamageClass::Special => (3usize, 4usize),  // spa, spd
        DamageClass::Status => unreachable!(),
    };

    let mut atk = attacker.stats[atk_idx] as u32;
    let mut dfn = defender.stats[dfn_idx] as u32;

    // sandstorm: Rock types get 1.5x SpDef
    if weather == Some(Weather::Sandstorm)
        && dfn_idx == 4
        && defender.has_type(Type::Rock)
    {
        dfn = dfn * 3 / 2;
    }

    // burn halves physical attack
    if attacker.status == Status::Burn && mv.damage_class == DamageClass::Physical {
        atk /= 2;
    }

    // crit calculation
    let mut crit_threshold = attacker.stats[5] as u32 / 2; // speed / 2
    if mv.meta.crit_rate > 0 {
        crit_threshold *= 4;
    }
    crit_threshold = crit_threshold.min(255);
    let is_crit = rng.random_range(0..=255u32) < crit_threshold;

    // stat stage application
    let atk_stage_idx = match mv.damage_class {
        DamageClass::Physical => 0, // STAT_ATK
        DamageClass::Special => 2,  // STAT_SPA
        DamageClass::Status => unreachable!(),
    };
    let dfn_stage_idx = match mv.damage_class {
        DamageClass::Physical => 1, // STAT_DEF
        DamageClass::Special => 3,  // STAT_SPD
        DamageClass::Status => unreachable!(),
    };

    if is_crit {
        // crits: only apply positive atk stages and negative def stages
        let atk_stage = attacker.stat_stages[atk_stage_idx];
        if atk_stage > 0 {
            let (num, den) = get_stage_multiplier(atk_stage);
            atk = atk * num as u32 / den as u32;
        }
        let dfn_stage = defender.stat_stages[dfn_stage_idx];
        if dfn_stage < 0 {
            let (num, den) = get_stage_multiplier(dfn_stage);
            dfn = dfn * num as u32 / den as u32;
        }
    } else {
        let atk_stage = attacker.stat_stages[atk_stage_idx];
        if atk_stage != 0 {
            let (num, den) = get_stage_multiplier(atk_stage);
            atk = atk * num as u32 / den as u32;
        }
        let dfn_stage = defender.stat_stages[dfn_stage_idx];
        if dfn_stage != 0 {
            let (num, den) = get_stage_multiplier(dfn_stage);
            dfn = dfn * num as u32 / den as u32;
        }
    }

    // prevent division by zero
    if dfn == 0 {
        dfn = 1;
    }

    // base damage
    let crit_mult: u32 = if is_crit { 2 } else { 1 };
    let base = ((2 * level * crit_mult / 5 + 2) * mv.power as u32 * atk / dfn) / 50 + 2;

    // STAB
    let stab = attacker.has_type(mv.move_type);

    // weather modifier
    let weather_boost = match (weather, mv.move_type) {
        (Some(Weather::Sun), Type::Fire) | (Some(Weather::Rain), Type::Water) => 1, // 1.5x
        (Some(Weather::Sun), Type::Water) | (Some(Weather::Rain), Type::Fire) => -1, // 0.5x
        _ => 0,
    };

    // type effectiveness
    let effectiveness = types::combined_effectiveness(mv.move_type, defender.types());

    if effectiveness == 0.0 {
        return DamageResult {
            damage: 0,
            effectiveness: 0.0,
            is_crit,
        };
    }

    // random factor
    let rand_factor = rng.random_range(85..=100u32);

    // apply multipliers with integer math (matching Python order)
    let mut damage = base;

    if stab {
        damage = damage * 3 / 2;
    }

    match weather_boost {
        1 => damage = damage * 3 / 2,
        -1 => damage /= 2,
        _ => {}
    }

    // type effectiveness as integer operations
    if effectiveness == 0.25 {
        damage /= 4;
    } else if effectiveness == 0.5 {
        damage /= 2;
    } else if effectiveness == 2.0 {
        damage *= 2;
    } else if effectiveness == 4.0 {
        damage *= 4;
    }

    damage = damage * rand_factor / 100;

    // screens (crits bypass)
    if !is_crit {
        if let Some(screens) = screens {
            if mv.damage_class == DamageClass::Physical && screens.reflect_turns > 0 {
                damage /= 2;
            } else if mv.damage_class == DamageClass::Special && screens.light_screen_turns > 0 {
                damage /= 2;
            }
        }
    }

    // minimum 1
    damage = damage.max(1);

    DamageResult {
        damage: damage as u16,
        effectiveness,
        is_crit,
    }
}

/// Expected damage (deterministic, no crit, average random factor)
pub fn calc_expected_damage(
    attacker: &Pokemon,
    defender: &Pokemon,
    mv: &MoveTemplate,
    screens: Option<&SideConditions>,
    weather: Option<Weather>,
) -> f32 {
    if mv.damage_class == DamageClass::Status || mv.power == 0 {
        return 0.0;
    }

    let level: u32 = 100;
    let (atk_idx, dfn_idx) = match mv.damage_class {
        DamageClass::Physical => (1usize, 2usize),
        DamageClass::Special => (3usize, 4usize),
        DamageClass::Status => unreachable!(),
    };

    let mut atk = attacker.stats[atk_idx] as u32;
    let mut dfn = defender.stats[dfn_idx] as u32;

    if weather == Some(Weather::Sandstorm)
        && dfn_idx == 4
        && defender.has_type(Type::Rock)
    {
        dfn = dfn * 3 / 2;
    }

    // apply all stat stages
    let atk_stage_idx = if mv.damage_class == DamageClass::Physical { 0 } else { 2 };
    let dfn_stage_idx = if mv.damage_class == DamageClass::Physical { 1 } else { 3 };

    let atk_stage = attacker.stat_stages[atk_stage_idx];
    if atk_stage != 0 {
        let (num, den) = get_stage_multiplier(atk_stage);
        atk = atk * num as u32 / den as u32;
    }
    let dfn_stage = defender.stat_stages[dfn_stage_idx];
    if dfn_stage != 0 {
        let (num, den) = get_stage_multiplier(dfn_stage);
        dfn = dfn * num as u32 / den as u32;
    }

    if attacker.status == Status::Burn && mv.damage_class == DamageClass::Physical {
        atk /= 2;
    }

    if dfn == 0 {
        dfn = 1;
    }

    let base = ((2 * level / 5 + 2) * mv.power as u32 * atk / dfn) / 50 + 2;

    let stab: f32 = if attacker.has_type(mv.move_type) { 1.5 } else { 1.0 };

    let weather_mult: f32 = match (weather, mv.move_type) {
        (Some(Weather::Sun), Type::Fire) | (Some(Weather::Rain), Type::Water) => 1.5,
        (Some(Weather::Sun), Type::Water) | (Some(Weather::Rain), Type::Fire) => 0.5,
        _ => 1.0,
    };

    let effectiveness = types::combined_effectiveness(mv.move_type, defender.types());
    if effectiveness == 0.0 {
        return 0.0;
    }

    let mut damage = base as f32 * stab * effectiveness * weather_mult * 0.925;

    if let Some(screens) = screens {
        if mv.damage_class == DamageClass::Physical && screens.reflect_turns > 0 {
            damage *= 0.5;
        } else if mv.damage_class == DamageClass::Special && screens.light_screen_turns > 0 {
            damage *= 0.5;
        }
    }

    if let Some(acc) = mv.accuracy {
        damage *= acc as f32 / 100.0;
    }

    damage.max(1.0)
}
