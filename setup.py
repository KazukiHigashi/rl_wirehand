from setuptools import setup

# setup(name='gym_stir',
#       version='0.0.1',
#       install_requires=['gymnasium>=1.0.0',
#                         'mujoco>=3.2.3',
#                         'scipy>=1.13.1',
#                         'opencv-python>=4.10.0.84',
#                         'numpy>=1.25.0',
#                         'stable-baselines3>=2.4.0'],
#       package_data={'gym_stir' : [
#       ]}
# )

setup(name='gym_wire_hand',
      version='0.0.1',
      install_requires=['gymnasium>=1.0.0',
                        'mujoco>=3.2.5',
                        'scipy>=1.13.1',
                        'numpy>=1.25.0',
                        'stable-baselines3>=2.4.0',
                        'tensorboard>=2.10.1'],
      # entry_points={
      #     "gymnasium.envs": [
      #         "WireHand-v0 = gym_wire_hand.envs.wire_hand_env:WireHandEnv",
      #     ],
      # },
)