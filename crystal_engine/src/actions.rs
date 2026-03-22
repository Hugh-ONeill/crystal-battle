// Battle actions

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    UseMove(u8),   // slot_index 0-3
    Switch(u8),    // team_index 0-5
    Struggle,
    Forfeit,
}

impl Action {
    pub fn is_switch(&self) -> bool {
        matches!(self, Action::Switch(_))
    }
}
