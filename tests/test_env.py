import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_env_reset():
    import gym_env  # registers the env
    import gymnasium
    env = gymnasium.make("CrystalBattle-v1")
    obs, info = env.reset(seed=42)
    assert obs.shape == (1052,)
    assert obs.dtype == np.float32
    assert "action_mask" in info
    assert len(info["action_mask"]) == 10
    env.close()


def test_env_step():
    import gym_env
    import gymnasium
    env = gymnasium.make("CrystalBattle-v1")
    obs, info = env.reset(seed=42)
    mask = info["action_mask"]

    # pick first valid action
    action = 0
    for i in range(10):
        if mask[i]:
            action = i
            break

    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert obs2.shape == (1052,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


def test_env_plays_to_completion():
    import gym_env
    import gymnasium
    env = gymnasium.make("CrystalBattle-v1")
    obs, info = env.reset(seed=123)

    for _ in range(300):
        mask = info["action_mask"]
        valid = [i for i in range(10) if mask[i]]
        action = valid[0]
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    assert terminated or truncated
    env.close()


def test_env_render():
    import gym_env
    import gymnasium
    env = gymnasium.make("CrystalBattle-v1", render_mode="ansi")
    env.reset(seed=42)
    output = env.render()
    assert output is not None
    assert "Turn" in output
    env.close()


def test_obs_values_in_range():
    import gym_env
    import gymnasium
    env = gymnasium.make("CrystalBattle-v1")
    obs, _ = env.reset(seed=42)
    # most values should be in [0, 1] but some effectiveness can go up to ~1.0 normalized
    assert np.all(obs >= -1.1)  # stat stages can be negative (-6/6 = -1.0)
    assert np.all(obs <= 2.1)   # damage fracs capped at 2.0
    env.close()
