// Pokemon species and battle instances

use crate::moves::MoveSlot;
use crate::status::Status;
use crate::types::Type;

// Stats array indices
pub const HP: usize = 0;
pub const ATK: usize = 1;
pub const DEF: usize = 2;
pub const SPA: usize = 3;
pub const SPD: usize = 4;
pub const SPE: usize = 5;

const PERFECT_DV: u16 = 15;
const MAX_STAT_EXP_BONUS: u16 = 64;
const LEVEL: u16 = 100;

/// Gen 2 stat formula at level 100, max stat exp, perfect DVs by default
pub fn calc_stat(base: u16, is_hp: bool, dv: u16) -> u16 {
    let core = (base + dv) * 2 + MAX_STAT_EXP_BONUS;
    // at level 100: core * 100 / 100 = core
    if is_hp {
        core + LEVEL + 10
    } else {
        core + 5
    }
}

#[derive(Debug, Clone)]
pub struct Pokemon {
    pub species_id: u16,
    pub name: String,
    pub types: [Option<Type>; 2],
    pub base_stats: [u16; 6], // raw base stats (for obs features)
    pub stats: [u16; 6],      // computed battle stats
    pub current_hp: u16,
    pub move_slots: Vec<MoveSlot>,

    // status
    pub status: Status,
    pub status_turns: u8,
    pub confusion_turns: u8,

    // stat stages: atk, def, spa, spd, spe, accuracy, evasion
    pub stat_stages: [i8; 7],

    // volatile flags
    pub flinched: bool,
    pub leech_seeded: bool,
    pub protected: bool,
    pub protect_consecutive: u8,
    pub recharging: bool,
    pub charging_move_id: Option<u16>,
    pub semi_invulnerable: Option<SemiInvuln>,
    pub locked_move_id: Option<u16>,
    pub locked_turns: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemiInvuln {
    Fly,
    Dig,
}

impl SemiInvuln {
    pub fn as_str(self) -> &'static str {
        match self {
            SemiInvuln::Fly => "fly",
            SemiInvuln::Dig => "dig",
        }
    }
}

impl Pokemon {
    pub fn new(
        species_id: u16,
        name: String,
        types: [Option<Type>; 2],
        base_stats: [u16; 6],
        dvs: [u16; 4], // atk, def, speed, special
        move_slots: Vec<MoveSlot>,
    ) -> Self {
        let stats = [
            calc_stat(base_stats[0], true, dvs[3].min(PERFECT_DV)),  // HP uses special DV? No -- HP DV is derived in Gen 2
            // Actually in the Python code, all DVs default to 15 and special is used for both spa and spd
            // Let's follow the Python: all DVs = 15 by default, calc_stat with dv=15 for everything
            calc_stat(base_stats[1], false, dvs[0].min(PERFECT_DV)),
            calc_stat(base_stats[2], false, dvs[1].min(PERFECT_DV)),
            calc_stat(base_stats[3], false, dvs[3].min(PERFECT_DV)), // spa uses special dv
            calc_stat(base_stats[4], false, dvs[3].min(PERFECT_DV)), // spd uses special dv
            calc_stat(base_stats[5], false, dvs[2].min(PERFECT_DV)),
        ];
        // HP DV in Gen 2 is derived from other DVs but the Python code just uses 15
        let hp = calc_stat(base_stats[0], true, PERFECT_DV);
        let mut stats_arr = stats;
        stats_arr[0] = hp;

        // resolve Hidden Power (move id 237): compute type/power from DVs
        let mut move_slots = move_slots;
        for slot in &mut move_slots {
            if slot.template.id == 237 {
                let (hp_type, hp_power) = crate::types::calc_hidden_power(
                    dvs[0] as u8, dvs[1] as u8, dvs[2] as u8, dvs[3] as u8,
                );
                slot.template.move_type = hp_type;
                slot.template.power = hp_power;
            }
        }

        Pokemon {
            species_id,
            name,
            types,
            base_stats,
            stats: stats_arr,
            current_hp: stats_arr[0],
            move_slots,
            status: Status::None,
            status_turns: 0,
            confusion_turns: 0,
            stat_stages: [0; 7],
            flinched: false,
            leech_seeded: false,
            protected: false,
            protect_consecutive: 0,
            recharging: false,
            charging_move_id: None,
            semi_invulnerable: None,
            locked_move_id: None,
            locked_turns: 0,
        }
    }

    /// Simplified constructor: all DVs = 15 (matches Python default)
    pub fn with_perfect_dvs(
        species_id: u16,
        name: String,
        types: [Option<Type>; 2],
        base_stats: [u16; 6],
        move_slots: Vec<MoveSlot>,
    ) -> Self {
        Self::new(species_id, name, types, base_stats, [15, 15, 15, 15], move_slots)
    }

    pub fn max_hp(&self) -> u16 {
        self.stats[HP]
    }

    pub fn hp_frac(&self) -> f32 {
        self.current_hp as f32 / self.max_hp() as f32
    }

    pub fn is_fainted(&self) -> bool {
        self.current_hp == 0
    }

    pub fn types(&self) -> &[Type] {
        if self.types[1].is_some() {
            // safe: both are Some
            unsafe {
                std::slice::from_raw_parts(
                    &self.types[0] as *const Option<Type> as *const Type,
                    2,
                )
            }
        } else if self.types[0].is_some() {
            unsafe {
                std::slice::from_raw_parts(
                    &self.types[0] as *const Option<Type> as *const Type,
                    1,
                )
            }
        } else {
            &[]
        }
    }

    pub fn has_type(&self, t: Type) -> bool {
        self.types[0] == Some(t) || self.types[1] == Some(t)
    }

    pub fn take_damage(&mut self, amount: u16) -> u16 {
        let actual = amount.min(self.current_hp);
        self.current_hp -= actual;
        actual
    }

    pub fn heal(&mut self, amount: u16) -> u16 {
        let max = self.max_hp();
        let actual = amount.min(max - self.current_hp);
        self.current_hp += actual;
        actual
    }

    pub fn clear_status(&mut self) {
        self.status = Status::None;
        self.status_turns = 0;
    }

    pub fn clear_confusion(&mut self) {
        self.confusion_turns = 0;
    }

    pub fn has_any_pp(&self) -> bool {
        self.move_slots.iter().any(|s| s.current_pp > 0)
    }

    /// Clear all volatile state (called on switch-out)
    pub fn clear_volatiles(&mut self) {
        self.confusion_turns = 0;
        self.flinched = false;
        self.leech_seeded = false;
        self.protected = false;
        self.protect_consecutive = 0;
        self.recharging = false;
        self.charging_move_id = None;
        self.semi_invulnerable = None;
        self.locked_move_id = None;
        self.locked_turns = 0;
        self.stat_stages = [0; 7];
    }

    /// Find move slot index by move id
    pub fn find_move_slot(&self, move_id: u16) -> Option<usize> {
        self.move_slots.iter().position(|s| s.template.id == move_id)
    }
}
