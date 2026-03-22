// Stat stage mechanics and move-to-effect mapping

pub const MIN_STAGE: i8 = -6;
pub const MAX_STAGE: i8 = 6;

/// Gen 2 stage multiplier: (2 + max(0, stage)) / (2 + max(0, -stage))
pub fn get_stage_multiplier(stage: i8) -> (i32, i32) {
    let num = 2 + stage.max(0) as i32;
    let den = 2 + (-stage).max(0) as i32;
    (num, den)
}

/// Apply stat stage change, clamped to [-6, +6]. Returns actual change.
pub fn apply_stat_change(stages: &mut [i8; 7], stat_idx: usize, change: i8) -> i8 {
    let current = stages[stat_idx];
    let new = current.saturating_add(change).clamp(MIN_STAGE, MAX_STAGE);
    let actual = new - current;
    stages[stat_idx] = new;
    actual
}

// Stat indices into the stat_stages array
pub const STAT_ATK: usize = 0;
pub const STAT_DEF: usize = 1;
pub const STAT_SPA: usize = 2;
pub const STAT_SPD: usize = 3;
pub const STAT_SPE: usize = 4;
pub const STAT_ACC: usize = 5;
pub const STAT_EVA: usize = 6;

pub fn stat_idx_from_name(name: &str) -> Option<usize> {
    match name {
        "attack" => Some(STAT_ATK),
        "defense" => Some(STAT_DEF),
        "special_attack" => Some(STAT_SPA),
        "special_defense" => Some(STAT_SPD),
        "speed" => Some(STAT_SPE),
        "accuracy" => Some(STAT_ACC),
        "evasion" => Some(STAT_EVA),
        _ => None,
    }
}

pub fn stat_name_from_idx(idx: usize) -> &'static str {
    match idx {
        STAT_ATK => "attack",
        STAT_DEF => "defense",
        STAT_SPA => "special_attack",
        STAT_SPD => "special_defense",
        STAT_SPE => "speed",
        STAT_ACC => "accuracy",
        STAT_EVA => "evasion",
        _ => "unknown",
    }
}

// Stat effect: (stat_index, stages, targets_self)
pub type StatEffect = (usize, i8, bool);

/// Lookup move stat effects by move name. Returns None if no effects.
pub fn move_stat_effects(name: &str) -> Option<&'static [StatEffect]> {
    match name {
        // self-boosting status moves
        "Swords Dance" => Some(&[(STAT_ATK, 2, true)]),
        "Growth" => Some(&[(STAT_SPA, 1, true)]),
        "Meditate" => Some(&[(STAT_ATK, 1, true)]),
        "Sharpen" => Some(&[(STAT_ATK, 1, true)]),
        "Agility" => Some(&[(STAT_SPE, 2, true)]),
        "Amnesia" => Some(&[(STAT_SPD, 2, true)]),
        "Barrier" => Some(&[(STAT_DEF, 2, true)]),
        "Acid Armor" => Some(&[(STAT_DEF, 2, true)]),
        "Double Team" => Some(&[(STAT_EVA, 1, true)]),
        "Minimize" => Some(&[(STAT_EVA, 1, true)]),
        "Harden" => Some(&[(STAT_DEF, 1, true)]),
        "Withdraw" => Some(&[(STAT_DEF, 1, true)]),
        "Defense Curl" => Some(&[(STAT_DEF, 1, true)]),
        "Curse" => Some(&[(STAT_ATK, 1, true), (STAT_DEF, 1, true), (STAT_SPE, -1, true)]),
        "Belly Drum" => Some(&[(STAT_ATK, 6, true)]),
        // opponent-lowering status moves
        "Growl" => Some(&[(STAT_ATK, -1, false)]),
        "Leer" => Some(&[(STAT_DEF, -1, false)]),
        "Tail Whip" => Some(&[(STAT_DEF, -1, false)]),
        "String Shot" => Some(&[(STAT_SPE, -1, false)]),
        "Screech" => Some(&[(STAT_DEF, -2, false)]),
        "Charm" => Some(&[(STAT_ATK, -2, false)]),
        "Scary Face" => Some(&[(STAT_SPE, -2, false)]),
        "Sweet Scent" => Some(&[(STAT_EVA, -1, false)]),
        "Sand Attack" => Some(&[(STAT_ACC, -1, false)]),
        "Smokescreen" => Some(&[(STAT_ACC, -1, false)]),
        "Flash" => Some(&[(STAT_ACC, -1, false)]),
        "Cotton Spore" => Some(&[(STAT_SPE, -2, false)]),
        "Kinesis" => Some(&[(STAT_ACC, -1, false)]),
        // secondary effects on damaging moves
        "Acid" => Some(&[(STAT_DEF, -1, false)]),
        "Bubble" => Some(&[(STAT_SPE, -1, false)]),
        "Bubble Beam" => Some(&[(STAT_SPE, -1, false)]),
        "Aurora Beam" => Some(&[(STAT_ATK, -1, false)]),
        "Psychic" => Some(&[(STAT_SPD, -1, false)]),
        "Constrict" => Some(&[(STAT_SPE, -1, false)]),
        "Mud-Slap" => Some(&[(STAT_ACC, -1, false)]),
        "Octazooka" => Some(&[(STAT_ACC, -1, false)]),
        "Icy Wind" => Some(&[(STAT_SPE, -1, false)]),
        "Crunch" => Some(&[(STAT_SPD, -1, false)]),
        "Shadow Ball" => Some(&[(STAT_SPD, -1, false)]),
        "Rock Smash" => Some(&[(STAT_DEF, -1, false)]),
        "Iron Tail" => Some(&[(STAT_DEF, -1, false)]),
        "Steel Wing" => Some(&[(STAT_DEF, 1, true)]),
        "Metal Claw" => Some(&[(STAT_ATK, 1, true)]),
        "Ancient Power" => Some(&[
            (STAT_ATK, 1, true),
            (STAT_DEF, 1, true),
            (STAT_SPA, 1, true),
            (STAT_SPD, 1, true),
            (STAT_SPE, 1, true),
        ]),
        _ => None,
    }
}
