from gridWorld import *
import gymnasium as gym
from gym import spaces


env = GridWorldEnv(n_width=10, n_height=10, u_size=60, default_reward=-1, default_type=0, windy=False)
env.action_space = spaces.Discrete(4)
env.start = (0,0)
env.ends =[(9,9)]
env.refresh_setting()

env.render()