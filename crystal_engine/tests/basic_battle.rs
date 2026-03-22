// Integration tests for the crystal engine

use crystal_engine_rs::actions::Action;
use crystal_engine_rs::battle::BattleState;
use crystal_engine_rs::damage::Weather;
use crystal_engine_rs::events::Event;
use crystal_engine_rs::moves::{DamageClass, MoveMeta, MoveSlot, MoveTemplate};
use crystal_engine_rs::player::PlayerState;
use crystal_engine_rs::pokemon::Pokemon;
use crystal_engine_rs::stat_stages::get_stage_multiplier;
use crystal_engine_rs::status::Status;
use crystal_engine_rs::turn_engine::{resolve_forced_switches, resolve_turn};
use crystal_engine_rs::types::Type;

use rand::SeedableRng;
use rand::rngs::SmallRng;

// ---- helpers ----

fn make_tackle() -> MoveTemplate {
    MoveTemplate {
        id: 33,
        name: "Tackle".into(),
        move_type: Type::Normal,
        power: 40,
        accuracy: Some(100),
        pp: 35,
        priority: 0,
        damage_class: DamageClass::Physical,
        meta: MoveMeta::default(),
    }
}

fn make_thunderbolt() -> MoveTemplate {
    MoveTemplate {
        id: 85,
        name: "Thunderbolt".into(),
        move_type: Type::Electric,
        power: 95,
        accuracy: Some(100),
        pp: 15,
        priority: 0,
        damage_class: DamageClass::Special,
        meta: MoveMeta {
            ailment_id: 1, // paralysis
            ailment_chance: 10,
            ..MoveMeta::default()
        },
    }
}

fn make_swords_dance() -> MoveTemplate {
    MoveTemplate {
        id: 14,
        name: "Swords Dance".into(),
        move_type: Type::Normal,
        power: 0,
        accuracy: None,
        pp: 20,
        priority: 0,
        damage_class: DamageClass::Status,
        meta: MoveMeta::default(),
    }
}

fn make_quick_attack() -> MoveTemplate {
    MoveTemplate {
        id: 98,
        name: "Quick Attack".into(),
        move_type: Type::Normal,
        power: 40,
        accuracy: Some(100),
        pp: 30,
        priority: 1,
        damage_class: DamageClass::Physical,
        meta: MoveMeta::default(),
    }
}

fn make_test_pokemon(name: &str, types: [Option<Type>; 2], base_stats: [u16; 6], moves: Vec<MoveTemplate>) -> Pokemon {
    let slots = moves.into_iter().map(MoveSlot::new).collect();
    Pokemon::with_perfect_dvs(1, name.into(), types, base_stats, slots)
}

fn make_simple_battle(seed: u64) -> BattleState {
    let p1_mon = make_test_pokemon(
        "Pikachu",
        [Some(Type::Electric), None],
        [35, 55, 30, 50, 40, 90],
        vec![make_thunderbolt(), make_quick_attack()],
    );
    let p2_mon = make_test_pokemon(
        "Geodude",
        [Some(Type::Rock), Some(Type::Ground)],
        [40, 80, 100, 30, 30, 20],
        vec![make_tackle()],
    );
    let p1 = PlayerState::new(vec![p1_mon]);
    let p2 = PlayerState::new(vec![p2_mon]);
    BattleState::new(p1, p2, SmallRng::seed_from_u64(seed))
}

// ============================================================
// TESTS
// ============================================================

#[test]
fn test_stat_calculation() {
    // Pikachu: HP base 35, with DV 15 and max stat exp
    // HP = (35+15)*2 + 64 + 100 + 10 = 100 + 64 + 110 = 274
    let pika = make_test_pokemon(
        "Pikachu", [Some(Type::Electric), None],
        [35, 55, 30, 50, 40, 90], vec![make_tackle()],
    );
    assert_eq!(pika.stats[0], 274); // HP
    assert_eq!(pika.current_hp, 274);
    // Attack: (55+15)*2 + 64 + 5 = 140 + 64 + 5 = 209
    assert_eq!(pika.stats[1], 209);
    // Speed: (90+15)*2 + 64 + 5 = 210 + 64 + 5 = 279
    assert_eq!(pika.stats[5], 279);
}

#[test]
fn test_stage_multiplier() {
    assert_eq!(get_stage_multiplier(0), (2, 2));
    assert_eq!(get_stage_multiplier(1), (3, 2));
    assert_eq!(get_stage_multiplier(-1), (2, 3));
    assert_eq!(get_stage_multiplier(6), (8, 2));
    assert_eq!(get_stage_multiplier(-6), (2, 8));
}

#[test]
fn test_type_effectiveness() {
    use crystal_engine_rs::types::{effectiveness, combined_effectiveness};
    // Electric vs Water = 2x
    assert_eq!(effectiveness(Type::Electric, Type::Water), 2.0);
    // Electric vs Ground = 0x (immune)
    assert_eq!(effectiveness(Type::Electric, Type::Ground), 0.0);
    // Electric vs Rock/Ground = 0x (ground immunity)
    assert_eq!(combined_effectiveness(Type::Electric, &[Type::Rock, Type::Ground]), 0.0);
    // Fire vs Grass = 2x
    assert_eq!(effectiveness(Type::Fire, Type::Grass), 2.0);
    // Normal vs Ghost = 0x
    assert_eq!(effectiveness(Type::Normal, Type::Ghost), 0.0);
}

#[test]
fn test_basic_turn() {
    let mut state = make_simple_battle(42);
    // both use move 0
    let events = resolve_turn(&mut state, Action::UseMove(0), Action::UseMove(0));
    // should have at least one move event
    let has_move_event = events.iter().any(|e| matches!(e, Event::Move { .. }));
    assert!(has_move_event, "Expected at least one Move event");
    assert_eq!(state.turn, 1);
}

#[test]
fn test_electric_vs_ground_immune() {
    let mut state = make_simple_battle(42);
    // Pikachu uses Thunderbolt vs Geodude (Rock/Ground) -- should be immune
    let events = resolve_turn(&mut state, Action::UseMove(0), Action::UseMove(0));
    // thunderbolt should deal 0 damage
    let tb_event = events.iter().find(|e| matches!(e, Event::Move { move_name, .. } if move_name == "Thunderbolt"));
    if let Some(Event::Move { damage, effectiveness, .. }) = tb_event {
        assert_eq!(*damage, 0);
        assert_eq!(*effectiveness, 0.0);
    } else {
        panic!("Expected Thunderbolt Move event");
    }
}

#[test]
fn test_priority_moves() {
    // Quick Attack (+1 priority) should go before Tackle (0 priority),
    // even if the Quick Attack user is slower
    let slow_mon = make_test_pokemon(
        "SlowMon", [Some(Type::Normal), None],
        [100, 100, 100, 100, 100, 10], // very slow
        vec![make_quick_attack()],
    );
    let fast_mon = make_test_pokemon(
        "FastMon", [Some(Type::Normal), None],
        [100, 100, 100, 100, 100, 150], // very fast
        vec![make_tackle()],
    );
    let p1 = PlayerState::new(vec![slow_mon]);
    let p2 = PlayerState::new(vec![fast_mon]);
    let mut state = BattleState::new(p1, p2, SmallRng::seed_from_u64(0));
    let events = resolve_turn(&mut state, Action::UseMove(0), Action::UseMove(0));
    // first move event should be from SlowMon (Quick Attack has priority)
    let first_move = events.iter().find(|e| matches!(e, Event::Move { .. }));
    if let Some(Event::Move { pokemon_name, move_name, .. }) = first_move {
        assert_eq!(pokemon_name, "SlowMon");
        assert_eq!(move_name, "Quick Attack");
    }
}

#[test]
fn test_switch() {
    let mon1 = make_test_pokemon("Mon1", [Some(Type::Normal), None], [100; 6], vec![make_tackle()]);
    let mon2 = make_test_pokemon("Mon2", [Some(Type::Fire), None], [100; 6], vec![make_tackle()]);
    let opp = make_test_pokemon("Opp", [Some(Type::Normal), None], [100; 6], vec![make_tackle()]);
    let p1 = PlayerState::new(vec![mon1, mon2]);
    let p2 = PlayerState::new(vec![opp]);
    let mut state = BattleState::new(p1, p2, SmallRng::seed_from_u64(0));

    // P1 switches to Mon2
    let events = resolve_turn(&mut state, Action::Switch(1), Action::UseMove(0));
    // switch should happen first
    let first_event = &events[0];
    assert!(matches!(first_event, Event::Switch { player: 1, to_name, .. } if to_name == "Mon2"));
    assert_eq!(state.p1.active_index, 1);
}

#[test]
fn test_forfeit() {
    let mon = make_test_pokemon("Mon", [Some(Type::Normal), None], [100; 6], vec![make_tackle()]);
    let opp = make_test_pokemon("Opp", [Some(Type::Normal), None], [100; 6], vec![make_tackle()]);
    let p1 = PlayerState::new(vec![mon]);
    let p2 = PlayerState::new(vec![opp]);
    let mut state = BattleState::new(p1, p2, SmallRng::seed_from_u64(0));
    let _ = resolve_turn(&mut state, Action::Forfeit, Action::UseMove(0));
    assert_eq!(state.winner, Some(2));
}

#[test]
fn test_swords_dance_boost() {
    let mon = make_test_pokemon("Mon", [Some(Type::Normal), None], [100; 6], vec![make_swords_dance(), make_tackle()]);
    let opp = make_test_pokemon("Opp", [Some(Type::Normal), None], [100; 6], vec![make_tackle()]);
    let p1 = PlayerState::new(vec![mon]);
    let p2 = PlayerState::new(vec![opp]);
    let mut state = BattleState::new(p1, p2, SmallRng::seed_from_u64(0));
    let events = resolve_turn(&mut state, Action::UseMove(0), Action::UseMove(0));
    // should have stat change event for attack +2
    let has_boost = events.iter().any(|e| matches!(e, Event::StatChange { stat, stages, .. } if stat == "attack" && *stages == 2));
    assert!(has_boost);
    assert_eq!(state.p1.active().stat_stages[0], 2); // atk stage
}

#[test]
fn test_hidden_power_calc() {
    use crystal_engine_rs::types::calc_hidden_power;
    // All 15 DVs: type = ((15&3)<<2 | (15&3)) % 16 = (3<<2|3) % 16 = 15 % 16 = 15 = Dark
    // power: bit3_sum = 8+4+2+1=15, ((5*15+3)/2)+31 = (78/2)+31 = 39+31 = 70
    let (hp_type, power) = calc_hidden_power(15, 15, 15, 15);
    assert_eq!(hp_type, Type::Dark);
    assert_eq!(power, 70);
}

#[test]
fn test_forced_switch() {
    let mon1 = make_test_pokemon("Mon1", [Some(Type::Normal), None], [1, 200, 100, 100, 100, 100], vec![make_tackle()]);
    let mon2 = make_test_pokemon("Mon2", [Some(Type::Normal), None], [100; 6], vec![make_tackle()]);
    let opp = make_test_pokemon("Opp", [Some(Type::Normal), None], [100; 6], vec![make_tackle()]);
    let p1 = PlayerState::new(vec![mon1, mon2]);
    let p2 = PlayerState::new(vec![opp]);
    let mut state = BattleState::new(p1, p2, SmallRng::seed_from_u64(42));

    // kill mon1
    let hp = state.p1.active().current_hp;
    state.p1.active_mut().take_damage(hp);
    assert!(state.p1.active().is_fainted());
    assert!(state.p1.must_switch());

    let events = resolve_forced_switches(&mut state, Some(1), None);
    assert!(events.iter().any(|e| matches!(e, Event::Switch { player: 1, to_name, .. } if to_name == "Mon2")));
    assert_eq!(state.p1.active_index, 1);
}
