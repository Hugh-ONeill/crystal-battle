// Quick benchmark: simulate N turns in pure Rust (no PyO3 overhead)

use crystal_engine_rs::actions::Action;
use crystal_engine_rs::battle::BattleState;
use crystal_engine_rs::moves::{DamageClass, MoveMeta, MoveSlot, MoveTemplate};
use crystal_engine_rs::player::PlayerState;
use crystal_engine_rs::pokemon::Pokemon;
use crystal_engine_rs::turn_engine::resolve_turn;
use crystal_engine_rs::types::Type;

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::Rng;
use std::time::Instant;

fn make_move(id: u16, name: &str, mt: Type, power: u8, dc: DamageClass) -> MoveTemplate {
    MoveTemplate {
        id, name: name.into(), move_type: mt, power, accuracy: Some(100),
        pp: 15, priority: 0, damage_class: dc, meta: MoveMeta::default(),
    }
}

fn make_team(rng: &mut SmallRng) -> Vec<Pokemon> {
    let types_pool = [Type::Normal, Type::Fire, Type::Water, Type::Electric, Type::Grass,
        Type::Ice, Type::Fighting, Type::Poison, Type::Ground, Type::Flying,
        Type::Psychic, Type::Bug, Type::Rock, Type::Ghost, Type::Dragon, Type::Dark, Type::Steel];
    let dc = [DamageClass::Physical, DamageClass::Special];

    (0..6).map(|i| {
        let t1 = types_pool[rng.random_range(0..types_pool.len())];
        let t2 = if rng.random_bool(0.5) { Some(types_pool[rng.random_range(0..types_pool.len())]) } else { None };
        let base = [
            rng.random_range(60..120u16), rng.random_range(60..130),
            rng.random_range(60..130), rng.random_range(60..130),
            rng.random_range(60..130), rng.random_range(60..130),
        ];
        let moves: Vec<MoveSlot> = (0..4).map(|j| {
            let mt = types_pool[rng.random_range(0..types_pool.len())];
            MoveSlot::new(make_move(
                (i * 4 + j) as u16 + 100, "Move", mt,
                rng.random_range(40..120), dc[rng.random_range(0..2)],
            ))
        }).collect();
        Pokemon::with_perfect_dvs(i as u16 + 1, format!("Mon{}", i), [Some(t1), t2], base, moves)
    }).collect()
}

fn main() {
    let n_games = 10000;
    let mut total_turns = 0u64;
    let mut rng = SmallRng::seed_from_u64(42);

    let start = Instant::now();

    for _ in 0..n_games {
        let t1 = make_team(&mut rng);
        let t2 = make_team(&mut rng);
        let game_rng = SmallRng::seed_from_u64(rng.random());
        let mut state = BattleState::new(
            PlayerState::new(t1), PlayerState::new(t2), game_rng,
        );

        for _ in 0..200 {
            if state.is_over() { break; }

            let p1_mask = state.p1.valid_action_mask(Some(&state.p2));
            let p2_mask = state.p2.valid_action_mask(Some(&state.p1));
            let p1_valid: Vec<u8> = (0..10).filter(|&i| p1_mask[i as usize]).collect();
            let p2_valid: Vec<u8> = (0..10).filter(|&i| p2_mask[i as usize]).collect();

            if p1_valid.is_empty() || p2_valid.is_empty() { break; }

            let a1_idx = p1_valid[rng.random_range(0..p1_valid.len())];
            let a2_idx = p2_valid[rng.random_range(0..p2_valid.len())];

            let a1 = match a1_idx {
                0..=3 => Action::UseMove(a1_idx),
                4..=9 => Action::Switch(a1_idx - 4),
                _ => Action::Struggle,
            };
            let a2 = match a2_idx {
                0..=3 => Action::UseMove(a2_idx),
                4..=9 => Action::Switch(a2_idx - 4),
                _ => Action::Struggle,
            };

            resolve_turn(&mut state, a1, a2);
            total_turns += 1;

            // forced switches
            for pnum in [1u8, 2] {
                let ps = state.get_player(pnum);
                if ps.must_switch() {
                    let idx = ps.team.iter().enumerate()
                        .find(|(i, p)| *i != ps.active_index as usize && !p.is_fainted())
                        .map(|(i, _)| i as u8);
                    if let Some(idx) = idx {
                        let sw1 = if pnum == 1 { Some(idx) } else { None };
                        let sw2 = if pnum == 2 { Some(idx) } else { None };
                        crystal_engine_rs::turn_engine::resolve_forced_switches(&mut state, sw1, sw2);
                    }
                }
            }
        }
    }

    let elapsed = start.elapsed();
    let turns_per_sec = total_turns as f64 / elapsed.as_secs_f64();

    println!("Games: {}", n_games);
    println!("Total turns: {}", total_turns);
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    println!("Turns/sec: {:.0}", turns_per_sec);
    println!("Games/sec: {:.0}", n_games as f64 / elapsed.as_secs_f64());
}
