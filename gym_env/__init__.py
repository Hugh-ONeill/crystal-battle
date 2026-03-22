# registers CrystalBattle-v1 with gymnasium

import gymnasium

gymnasium.register(
    id="CrystalBattle-v1",
    entry_point="gym_env.battle_env:CrystalBattleEnv",
)
