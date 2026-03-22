// Player state: team, active mon, side conditions, valid actions

use crate::actions::Action;
use crate::moves::DamageClass;
use crate::pokemon::Pokemon;
use crate::status::Status;
use crate::types;

// Move IDs with special gating
const DREAM_EATER: u16 = 138;
const SNORE: u16 = 173;
const SLEEP_TALK: u16 = 214;
const REST: u16 = 156;

#[derive(Debug, Clone)]
pub struct SideConditions {
    pub spikes: bool,
    pub reflect_turns: u8,
    pub light_screen_turns: u8,
}

impl SideConditions {
    pub fn new() -> Self {
        SideConditions {
            spikes: false,
            reflect_turns: 0,
            light_screen_turns: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlayerState {
    pub team: Vec<Pokemon>,
    pub active_index: u8,
    pub side: SideConditions,
    pub active_turns: u16,
}

impl PlayerState {
    pub fn new(team: Vec<Pokemon>) -> Self {
        PlayerState {
            team,
            active_index: 0,
            side: SideConditions::new(),
            active_turns: 1,
        }
    }

    pub fn active(&self) -> &Pokemon {
        &self.team[self.active_index as usize]
    }

    pub fn active_mut(&mut self) -> &mut Pokemon {
        &mut self.team[self.active_index as usize]
    }

    pub fn is_defeated(&self) -> bool {
        self.team.iter().all(|p| p.is_fainted())
    }

    pub fn must_switch(&self) -> bool {
        self.active().is_fainted() && !self.is_defeated()
    }

    pub fn alive_count(&self) -> u8 {
        self.team.iter().filter(|p| !p.is_fainted()).count() as u8
    }

    pub fn total_hp_frac(&self) -> f32 {
        let (sum_cur, sum_max) = self.team.iter().fold((0u32, 0u32), |(c, m), p| {
            (c + p.current_hp as u32, m + p.max_hp() as u32)
        });
        if sum_max == 0 {
            0.0
        } else {
            sum_cur as f32 / sum_max as f32
        }
    }

    /// Execute a switch. Clears volatiles on outgoing mon.
    /// Returns the old pokemon's name (or None if same index).
    pub fn switch_to(&mut self, team_index: u8) -> Option<String> {
        let old_idx = self.active_index as usize;
        if team_index as usize == old_idx {
            return None;
        }
        let old_name = self.team[old_idx].name.clone();
        self.team[old_idx].clear_volatiles();
        self.active_index = team_index;
        self.active_turns = 1;
        Some(old_name)
    }

    /// Compute valid actions for this player.
    /// When `filter_immune` is true, soft-filters moves that are type-immune
    /// against the opponent (damaging moves with 0x effectiveness and status
    /// moves with an ailment that can't land through immunity). Falls back
    /// to the unfiltered list if every move would be filtered.
    pub fn valid_actions(&self, opponent: Option<&PlayerState>) -> Vec<Action> {
        self.valid_actions_filtered(opponent, false)
    }

    pub fn valid_actions_filtered(
        &self,
        opponent: Option<&PlayerState>,
        filter_immune: bool,
    ) -> Vec<Action> {
        let active = self.active();

        // forced switch after faint
        if self.must_switch() {
            let mut actions = Vec::new();
            for (i, p) in self.team.iter().enumerate() {
                if i != self.active_index as usize && !p.is_fainted() {
                    actions.push(Action::Switch(i as u8));
                }
            }
            return actions;
        }

        // locked states: no switching
        if active.recharging || active.charging_move_id.is_some() {
            return vec![Action::UseMove(0)]; // dummy, engine overrides
        }
        if let Some(locked_id) = active.locked_move_id {
            if let Some(idx) = active.find_move_slot(locked_id) {
                return vec![Action::UseMove(idx as u8)];
            }
            return vec![Action::UseMove(0)];
        }

        // normal: collect usable moves
        if !active.has_any_pp() {
            return vec![Action::Struggle];
        }

        let opp_active = opponent.map(|o| o.active());
        let opp_asleep = opp_active.map_or(false, |p| p.status == Status::Sleep);
        let user_asleep = active.status == Status::Sleep;

        let mut usable = Vec::new();
        let mut immune_filtered = Vec::new();

        for (i, slot) in active.move_slots.iter().enumerate() {
            if slot.current_pp == 0 {
                continue;
            }
            let mid = slot.template.id;
            if mid == DREAM_EATER && !opp_asleep {
                continue;
            }
            if (mid == SNORE || mid == SLEEP_TALK) && !user_asleep {
                continue;
            }
            if mid == REST && user_asleep {
                continue;
            }

            let action = Action::UseMove(i as u8);
            usable.push(action);

            // soft-filter type-immune moves when requested
            if filter_immune {
                if let Some(opp) = opp_active {
                    let opp_types = opp.types();
                    let eff = types::combined_effectiveness(slot.template.move_type, opp_types);
                    if eff == 0.0 {
                        // damaging move with 0x effectiveness
                        if slot.template.power > 0 {
                            continue;
                        }
                        // status move with an ailment component
                        if slot.template.damage_class == DamageClass::Status
                            && slot.template.meta.ailment_id > 0
                        {
                            continue;
                        }
                    }
                }
            }
            immune_filtered.push(action);
        }

        // prefer immune-filtered list, fall back to all usable
        let moves = if !immune_filtered.is_empty() {
            immune_filtered
        } else {
            usable
        };

        if moves.is_empty() {
            return vec![Action::Struggle];
        }

        let mut actions = moves;

        // switch actions
        for (i, p) in self.team.iter().enumerate() {
            if i != self.active_index as usize && !p.is_fainted() {
                actions.push(Action::Switch(i as u8));
            }
        }

        actions
    }

    /// 10-element action mask: [move0..move3, switch0..switch5]
    pub fn valid_action_mask(&self, opponent: Option<&PlayerState>) -> [bool; 10] {
        self.valid_action_mask_filtered(opponent, false)
    }

    pub fn valid_action_mask_filtered(
        &self,
        opponent: Option<&PlayerState>,
        filter_immune: bool,
    ) -> [bool; 10] {
        let mut mask = [false; 10];
        for action in self.valid_actions_filtered(opponent, filter_immune) {
            match action {
                Action::UseMove(i) => mask[i as usize] = true,
                Action::Switch(i) => mask[4 + i as usize] = true,
                Action::Struggle => mask[0] = true,
                Action::Forfeit => {}
            }
        }
        mask
    }
}
