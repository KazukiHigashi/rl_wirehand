from gymnasium.envs.registration import register


register(
    id='WireHand-v0',
    entry_point='gym_stir.envs.wire_hand_env:WireHandEnv',
    max_episode_steps=2000,
)