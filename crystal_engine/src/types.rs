// Type system and effectiveness chart for Gen 2

use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pyclass(eq, eq_int, skip_from_py_object)]
pub enum Type {
    Bug = 0,
    Dark = 1,
    Dragon = 2,
    Electric = 3,
    Fighting = 4,
    Fire = 5,
    Flying = 6,
    Ghost = 7,
    Grass = 8,
    Ground = 9,
    Ice = 10,
    Normal = 11,
    Poison = 12,
    Psychic = 13,
    Rock = 14,
    Steel = 15,
    Water = 16,
}

pub const NUM_TYPES: usize = 17;

impl Type {
    pub fn from_str(s: &str) -> Option<Type> {
        match s {
            "bug" => Some(Type::Bug),
            "dark" => Some(Type::Dark),
            "dragon" => Some(Type::Dragon),
            "electric" => Some(Type::Electric),
            "fighting" => Some(Type::Fighting),
            "fire" => Some(Type::Fire),
            "flying" => Some(Type::Flying),
            "ghost" => Some(Type::Ghost),
            "grass" => Some(Type::Grass),
            "ground" => Some(Type::Ground),
            "ice" => Some(Type::Ice),
            "normal" => Some(Type::Normal),
            "poison" => Some(Type::Poison),
            "psychic" => Some(Type::Psychic),
            "rock" => Some(Type::Rock),
            "steel" => Some(Type::Steel),
            "water" => Some(Type::Water),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Type::Bug => "bug",
            Type::Dark => "dark",
            Type::Dragon => "dragon",
            Type::Electric => "electric",
            Type::Fighting => "fighting",
            Type::Fire => "fire",
            Type::Flying => "flying",
            Type::Ghost => "ghost",
            Type::Grass => "grass",
            Type::Ground => "ground",
            Type::Ice => "ice",
            Type::Normal => "normal",
            Type::Poison => "poison",
            Type::Psychic => "psychic",
            Type::Rock => "rock",
            Type::Steel => "steel",
            Type::Water => "water",
        }
    }

    pub fn idx(self) -> usize {
        self as usize
    }
}

// Effectiveness stored as u8: 0=immune, 2=0.5x, 4=1x, 8=2x
// Multiply: result = product / 4 for single type, product / 16 for dual type
// Or just use f32 directly -- the chart is small and f32 is fine for 100x speedup goals.

// Type chart: TYPE_CHART[atk][def] -> effectiveness multiplier
// Order matches Type enum: Bug, Dark, Dragon, Electric, Fighting, Fire, Flying,
//   Ghost, Grass, Ground, Ice, Normal, Poison, Psychic, Rock, Steel, Water
#[rustfmt::skip]
pub static TYPE_CHART: [[f32; NUM_TYPES]; NUM_TYPES] = [
    // Bug attacking ->
    [1.0, 2.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 2.0, 1.0, 1.0, 1.0, 0.5, 2.0, 1.0, 0.5, 1.0],
    // Dark attacking ->
    [1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0],
    // Dragon attacking ->
    [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0],
    // Electric attacking ->
    [1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 2.0, 1.0, 0.5, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
    // Fighting attacking ->
    [0.5, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0, 2.0, 2.0, 0.5, 0.5, 2.0, 2.0, 1.0],
    // Fire attacking ->
    [2.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.5, 2.0, 0.5],
    // Flying attacking ->
    [2.0, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0],
    // Ghost attacking ->
    [1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.5, 1.0],
    // Grass attacking ->
    [0.5, 1.0, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 2.0, 1.0, 1.0, 0.5, 1.0, 2.0, 0.5, 2.0],
    // Ground attacking ->
    [0.5, 1.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0],
    // Ice attacking ->
    [1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 2.0, 1.0, 2.0, 2.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    // Normal attacking ->
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0],
    // Poison attacking ->
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 2.0, 0.5, 1.0, 1.0, 0.5, 1.0, 0.5, 0.0, 1.0],
    // Psychic attacking ->
    [1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.5, 1.0],
    // Rock attacking ->
    [2.0, 1.0, 1.0, 1.0, 0.5, 2.0, 2.0, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0],
    // Steel attacking ->
    [1.0, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 0.5, 0.5],
    // Water attacking ->
    [1.0, 1.0, 0.5, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5],
];

/// Single-type effectiveness lookup
pub fn effectiveness(atk: Type, def: Type) -> f32 {
    TYPE_CHART[atk.idx()][def.idx()]
}

/// Combined effectiveness against one or two defender types
pub fn combined_effectiveness(atk: Type, def_types: &[Type]) -> f32 {
    let mut eff = 1.0f32;
    for &dt in def_types {
        eff *= TYPE_CHART[atk.idx()][dt.idx()];
    }
    eff
}

// Hidden Power type table (Gen 2)
const HP_TYPE_TABLE: [Type; 16] = [
    Type::Fighting, Type::Flying, Type::Poison, Type::Ground,
    Type::Rock, Type::Bug, Type::Ghost, Type::Steel,
    Type::Fire, Type::Water, Type::Grass, Type::Electric,
    Type::Psychic, Type::Ice, Type::Dragon, Type::Dark,
];

/// Gen 2 Hidden Power: returns (type, power) from DVs
pub fn calc_hidden_power(atk_dv: u8, def_dv: u8, spd_dv: u8, spc_dv: u8) -> (Type, u8) {
    let type_idx = (((atk_dv & 3) << 2) | (def_dv & 3)) as usize % 16;
    let hp_type = HP_TYPE_TABLE[type_idx];

    let bit3_sum = (if atk_dv & 8 != 0 { 8u16 } else { 0 })
        + (if def_dv & 8 != 0 { 4 } else { 0 })
        + (if spd_dv & 8 != 0 { 2 } else { 0 })
        + (if spc_dv & 8 != 0 { 1 } else { 0 });
    let power = ((5 * bit3_sum + (spc_dv as u16 & 3)) / 2 + 31) as u8;

    (hp_type, power)
}

/// Brute-force search for optimal DVs that produce a given Hidden Power type
/// with maximum power. Returns (atk_dv, def_dv, spd_dv, spc_dv).
pub fn dvs_for_hidden_power(hp_type: Type) -> (u8, u8, u8, u8) {
    let mut best = (15u8, 15u8, 15u8, 15u8);
    let mut best_power = 0u8;

    for atk in 0..16u8 {
        for def in 0..16u8 {
            for spd in 0..16u8 {
                for spc in 0..16u8 {
                    let (t, p) = calc_hidden_power(atk, def, spd, spc);
                    if t == hp_type && p > best_power {
                        best_power = p;
                        best = (atk, def, spd, spc);
                    }
                }
            }
        }
    }
    best
}
