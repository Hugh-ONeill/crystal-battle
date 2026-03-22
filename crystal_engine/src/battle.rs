// Top-level battle state

use crate::damage::Weather;
use crate::player::PlayerState;

pub const WEATHER_DURATION: u8 = 5;

#[derive(Debug, Clone)]
pub struct BattleState {
    pub p1: PlayerState,
    pub p2: PlayerState,
    pub turn: u16,
    pub winner: Option<u8>, // Some(1) or Some(2)
    pub rng: rand::rngs::SmallRng,
    pub weather: Option<Weather>,
    pub weather_turns: u8,
}

impl BattleState {
    pub fn new(p1: PlayerState, p2: PlayerState, rng: rand::rngs::SmallRng) -> Self {
        BattleState {
            p1,
            p2,
            turn: 0,
            winner: None,
            rng,
            weather: None,
            weather_turns: 0,
        }
    }

    pub fn is_over(&self) -> bool {
        self.winner.is_some()
    }

    pub fn check_winner(&mut self) -> Option<u8> {
        if self.p1.is_defeated() {
            self.winner = Some(2);
        } else if self.p2.is_defeated() {
            self.winner = Some(1);
        }
        self.winner
    }

    pub fn get_player(&self, player_num: u8) -> &PlayerState {
        if player_num == 1 { &self.p1 } else { &self.p2 }
    }

    pub fn get_player_mut(&mut self, player_num: u8) -> &mut PlayerState {
        if player_num == 1 { &mut self.p1 } else { &mut self.p2 }
    }

    /// Get mutable references to both players (attacker, defender) by player number.
    /// Panics if player_num is not 1 or 2.
    pub fn players_mut(&mut self, attacker_num: u8) -> (&mut PlayerState, &mut PlayerState) {
        if attacker_num == 1 {
            (&mut self.p1, &mut self.p2)
        } else {
            (&mut self.p2, &mut self.p1)
        }
    }
}
