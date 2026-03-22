// JSON data loading for pokemon and moves

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::moves::{DamageClass, MoveMeta, MoveSlot, MoveTemplate};
use crate::pokemon::Pokemon;
use crate::types::Type;

#[derive(Deserialize)]
struct RawPokemon {
    id: u16,
    name: String,
    base_stats: HashMap<String, u16>,
    types: Vec<String>,
    #[allow(dead_code)]
    learnset: Vec<u16>,
}

#[derive(Deserialize)]
struct RawMoveMeta {
    ailment_id: Option<i8>,
    min_hits: Option<u8>,
    max_hits: Option<u8>,
    drain: Option<i8>,
    healing: Option<i8>,
    crit_rate: Option<u8>,
    ailment_chance: Option<u8>,
    flinch_chance: Option<u8>,
    stat_chance: Option<u8>,
}

#[derive(Deserialize)]
struct RawMove {
    id: u16,
    name: String,
    #[serde(rename = "type")]
    move_type: String,
    power: Option<u8>,
    accuracy: Option<u8>,
    pp: u8,
    priority: Option<i8>,
    damage_class: String,
    meta: Option<RawMoveMeta>,
}

pub struct DataStore {
    pokemon: HashMap<u16, RawPokemon>,
    pub moves: HashMap<u16, MoveTemplate>,
}

impl DataStore {
    pub fn load(data_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let pokemon_json = std::fs::read_to_string(data_dir.join("pokemon.json"))?;
        let raw_pokemon: Vec<RawPokemon> = serde_json::from_str(&pokemon_json)?;
        let pokemon: HashMap<u16, RawPokemon> =
            raw_pokemon.into_iter().map(|p| (p.id, p)).collect();

        let moves_json = std::fs::read_to_string(data_dir.join("moves.json"))?;
        let raw_moves: Vec<RawMove> = serde_json::from_str(&moves_json)?;
        let moves: HashMap<u16, MoveTemplate> = raw_moves
            .into_iter()
            .map(|m| {
                let meta = m.meta.map(|rm| MoveMeta {
                    ailment_id: rm.ailment_id.unwrap_or(0).max(0) as u8,
                    min_hits: rm.min_hits,
                    max_hits: rm.max_hits,
                    drain: rm.drain.unwrap_or(0),
                    healing: rm.healing.unwrap_or(0).max(0) as u8,
                    crit_rate: rm.crit_rate.unwrap_or(0),
                    ailment_chance: rm.ailment_chance.unwrap_or(0),
                    flinch_chance: rm.flinch_chance.unwrap_or(0),
                    stat_chance: rm.stat_chance.unwrap_or(0),
                }).unwrap_or_default();

                let template = MoveTemplate {
                    id: m.id,
                    name: m.name,
                    move_type: Type::from_str(&m.move_type).unwrap_or(Type::Normal),
                    power: m.power.unwrap_or(0),
                    accuracy: m.accuracy,
                    pp: m.pp,
                    priority: m.priority.unwrap_or(0),
                    damage_class: DamageClass::from_str(&m.damage_class)
                        .unwrap_or(DamageClass::Status),
                    meta,
                };
                (template.id, template)
            })
            .collect();

        Ok(DataStore { pokemon, moves })
    }

    pub fn get_move(&self, id: u16) -> Option<&MoveTemplate> {
        self.moves.get(&id)
    }

    pub fn build_pokemon(&self, species_id: u16, move_ids: &[u16]) -> Option<Pokemon> {
        let raw = self.pokemon.get(&species_id)?;
        let types = {
            let mut t = [None; 2];
            for (i, type_name) in raw.types.iter().enumerate().take(2) {
                t[i] = Type::from_str(type_name);
            }
            t
        };
        let base_stats = [
            *raw.base_stats.get("hp").unwrap_or(&50),
            *raw.base_stats.get("attack").unwrap_or(&50),
            *raw.base_stats.get("defense").unwrap_or(&50),
            *raw.base_stats.get("special_attack").unwrap_or(&50),
            *raw.base_stats.get("special_defense").unwrap_or(&50),
            *raw.base_stats.get("speed").unwrap_or(&50),
        ];

        let move_slots: Vec<MoveSlot> = move_ids
            .iter()
            .filter_map(|&mid| self.moves.get(&mid))
            .take(4)
            .map(|t| MoveSlot::new(t.clone()))
            .collect();

        Some(Pokemon::with_perfect_dvs(
            species_id,
            raw.name.clone(),
            types,
            base_stats,
            move_slots,
        ))
    }

    /// Build a Pokemon with explicit DVs (for Hidden Power type resolution)
    pub fn build_pokemon_with_dvs(
        &self,
        species_id: u16,
        move_ids: &[u16],
        dvs: [u16; 4], // atk, def, speed, special
    ) -> Option<Pokemon> {
        let raw = self.pokemon.get(&species_id)?;
        let types = {
            let mut t = [None; 2];
            for (i, type_name) in raw.types.iter().enumerate().take(2) {
                t[i] = Type::from_str(type_name);
            }
            t
        };
        let base_stats = [
            *raw.base_stats.get("hp").unwrap_or(&50),
            *raw.base_stats.get("attack").unwrap_or(&50),
            *raw.base_stats.get("defense").unwrap_or(&50),
            *raw.base_stats.get("special_attack").unwrap_or(&50),
            *raw.base_stats.get("special_defense").unwrap_or(&50),
            *raw.base_stats.get("speed").unwrap_or(&50),
        ];

        let move_slots: Vec<MoveSlot> = move_ids
            .iter()
            .filter_map(|&mid| self.moves.get(&mid))
            .take(4)
            .map(|t| MoveSlot::new(t.clone()))
            .collect();

        Some(Pokemon::new(
            species_id,
            raw.name.clone(),
            types,
            base_stats,
            dvs,
            move_slots,
        ))
    }
}
