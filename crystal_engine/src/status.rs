// Status effects: application, prevention, and end-of-turn damage

use rand::Rng;

use crate::pokemon::Pokemon;
use crate::stat_stages::get_stage_multiplier;
use crate::types::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    None,
    Burn,
    Paralysis,
    Sleep,
    Freeze,
    Poison,
    Toxic,
}

impl Status {
    pub fn is_none(self) -> bool {
        self == Status::None
    }

    pub fn is_some(self) -> bool {
        self != Status::None
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Status::None => "",
            Status::Burn => "brn",
            Status::Paralysis => "par",
            Status::Sleep => "slp",
            Status::Freeze => "frz",
            Status::Poison => "psn",
            Status::Toxic => "tox",
        }
    }

    pub fn from_str(s: &str) -> Status {
        match s {
            "brn" => Status::Burn,
            "par" => Status::Paralysis,
            "slp" => Status::Sleep,
            "frz" => Status::Freeze,
            "psn" => Status::Poison,
            "tox" => Status::Toxic,
            _ => Status::None,
        }
    }
}

/// Check if a status can be applied to a pokemon. Returns (can_apply, reason).
pub fn can_apply_status(pokemon: &Pokemon, status: Status) -> (bool, &'static str) {
    if pokemon.status.is_some() {
        return (false, "already has status");
    }
    if pokemon.is_fainted() {
        return (false, "fainted");
    }
    // type immunities
    for &t in pokemon.types() {
        match t {
            Type::Fire if status == Status::Burn => return (false, "fire immune to burn"),
            Type::Electric if status == Status::Paralysis => {
                return (false, "electric immune to paralysis")
            }
            Type::Poison | Type::Steel
                if status == Status::Poison || status == Status::Toxic =>
            {
                return (false, "immune to poison")
            }
            Type::Ice if status == Status::Freeze => return (false, "ice immune to freeze"),
            _ => {}
        }
    }
    (true, "")
}

/// Apply a status to a pokemon. Returns true if applied.
pub fn apply_status(pokemon: &mut Pokemon, status: Status, rng: &mut impl Rng) -> bool {
    let (can, _) = can_apply_status(pokemon, status);
    if !can {
        return false;
    }
    pokemon.status = status;
    match status {
        Status::Sleep => pokemon.status_turns = rng.random_range(1..=7),
        Status::Toxic => pokemon.status_turns = 0,
        _ => pokemon.status_turns = 0,
    }
    true
}

/// Apply confusion. Returns true if applied.
pub fn apply_confusion(pokemon: &mut Pokemon, rng: &mut impl Rng) -> bool {
    if pokemon.confusion_turns > 0 || pokemon.is_fainted() {
        return false;
    }
    pokemon.confusion_turns = rng.random_range(2..=5);
    true
}

/// Result of checking whether a move is prevented by status
pub enum MovePreventionResult {
    CanAct,
    WokeUp,
    ThawedOut,
    Prevented(&'static str), // reason: "fast asleep", "frozen solid", "fully paralyzed"
}

/// Check if status prevents acting this turn. Mutates pokemon state.
pub fn check_move_prevention(pokemon: &mut Pokemon, rng: &mut impl Rng) -> MovePreventionResult {
    match pokemon.status {
        Status::Sleep => {
            pokemon.status_turns -= 1;
            if pokemon.status_turns == 0 {
                pokemon.clear_status();
                MovePreventionResult::WokeUp
            } else {
                MovePreventionResult::Prevented("fast asleep")
            }
        }
        Status::Freeze => {
            if rng.random_range(0..100) < 20 {
                pokemon.clear_status();
                MovePreventionResult::ThawedOut
            } else {
                MovePreventionResult::Prevented("frozen solid")
            }
        }
        Status::Paralysis => {
            if rng.random_range(0..100) < 25 {
                MovePreventionResult::Prevented("fully paralyzed")
            } else {
                MovePreventionResult::CanAct
            }
        }
        _ => MovePreventionResult::CanAct,
    }
}

/// Check confusion. Returns (hit_self, self_damage).
pub fn check_confusion(pokemon: &mut Pokemon, rng: &mut impl Rng) -> (bool, u16) {
    if pokemon.confusion_turns == 0 {
        return (false, 0);
    }
    pokemon.confusion_turns -= 1;
    // 50% chance to hit self
    if rng.random_range(0..2) == 0 {
        // typeless 40-power physical hit, no crits, no random roll
        let atk = pokemon.stats[1] as u32; // attack
        let dfn = pokemon.stats[2] as u32; // defense
        let damage = ((42 * 40 * atk / dfn) / 50 + 2) as u16;
        (true, damage)
    } else {
        (false, 0)
    }
}

/// Calculate end-of-turn residual damage. Returns damage amount.
/// For toxic, increments status_turns as a side effect.
pub fn end_of_turn_damage(pokemon: &mut Pokemon) -> u16 {
    let max_hp = pokemon.stats[0] as u16;
    match pokemon.status {
        Status::Burn | Status::Poison => max_hp / 8,
        Status::Toxic => {
            pokemon.status_turns += 1;
            (max_hp as u32 * pokemon.status_turns as u32 / 16) as u16
        }
        _ => 0,
    }
}

/// Determine status from move's ailment_id. ailment_id 5 + "Toxic" -> Toxic.
pub fn status_from_move(move_name: &str, ailment_id: u8) -> Option<Status> {
    match ailment_id {
        1 => Some(Status::Paralysis),
        2 => Some(Status::Sleep),
        3 => Some(Status::Freeze),
        4 => Some(Status::Burn),
        5 => {
            if move_name == "Toxic" {
                Some(Status::Toxic)
            } else {
                Some(Status::Poison)
            }
        }
        _ => None,
    }
}

/// Returns true if ailment_id represents confusion
pub fn confusion_from_move(ailment_id: u8) -> bool {
    ailment_id == 6
}

/// Effective speed accounting for stat stages and paralysis
pub fn effective_speed(pokemon: &Pokemon) -> u32 {
    let base_speed = pokemon.stats[5] as u32;
    let stage = pokemon.stat_stages[4]; // speed stage
    let (num, den) = get_stage_multiplier(stage);
    let mut speed = base_speed * num as u32 / den as u32;
    if pokemon.status == Status::Paralysis {
        speed /= 4;
    }
    speed
}
