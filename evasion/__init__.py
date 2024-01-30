from gymnasium.envs.registration import register


register(
    id='evasion/MissionGym-v0',
    entry_point='evasion.envs:MissionGym',
    max_episode_steps=300,
)
