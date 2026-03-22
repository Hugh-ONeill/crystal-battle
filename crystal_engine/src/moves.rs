// Move templates and move slots

use crate::types::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DamageClass {
    Physical,
    Special,
    Status,
}

impl DamageClass {
    pub fn from_str(s: &str) -> Option<DamageClass> {
        match s {
            "physical" => Some(DamageClass::Physical),
            "special" => Some(DamageClass::Special),
            "status" => Some(DamageClass::Status),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MoveMeta {
    pub ailment_id: u8,
    pub min_hits: Option<u8>,
    pub max_hits: Option<u8>,
    pub drain: i8,
    pub healing: u8,
    pub crit_rate: u8,
    pub ailment_chance: u8,
    pub flinch_chance: u8,
    pub stat_chance: u8,
}

impl Default for MoveMeta {
    fn default() -> Self {
        MoveMeta {
            ailment_id: 0,
            min_hits: None,
            max_hits: None,
            drain: 0,
            healing: 0,
            crit_rate: 0,
            ailment_chance: 0,
            flinch_chance: 0,
            stat_chance: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MoveTemplate {
    pub id: u16,
    pub name: String,
    pub move_type: Type,
    pub power: u8,
    pub accuracy: Option<u8>, // None = always hits
    pub pp: u8,
    pub priority: i8,
    pub damage_class: DamageClass,
    pub meta: MoveMeta,
}

pub fn struggle() -> MoveTemplate {
    MoveTemplate {
        id: 165,
        name: "Struggle".to_string(),
        move_type: Type::Normal,
        power: 50,
        accuracy: None,
        pp: 255, // sentinel, never depleted
        priority: 0,
        damage_class: DamageClass::Physical,
        meta: MoveMeta::default(),
    }
}

pub const STRUGGLE_ID: u16 = 165;

#[derive(Debug, Clone)]
pub struct MoveSlot {
    pub template: MoveTemplate,
    pub current_pp: u8,
}

impl MoveSlot {
    pub fn new(template: MoveTemplate) -> Self {
        let pp = template.pp;
        MoveSlot {
            template,
            current_pp: pp,
        }
    }

    pub fn has_pp(&self) -> bool {
        self.current_pp > 0
    }

    pub fn use_pp(&mut self) {
        if self.current_pp > 0 {
            self.current_pp -= 1;
        }
    }
}
