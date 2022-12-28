from stable_baselines3 import PPO
from quartoenv import RandomOpponentEnv
from policies import QuartoPPO, RandomPolicy, MaskedRandomPolicy
from sb3_contrib.common.wrappers import ActionMasker
from itertools import product
import gym
import numpy as np

def mask_function(env: gym.Env) -> np.ndarray:
    """This function returns the encoding of the valid moves given the actual
    """
    # work out all actions: [(0, 0), ..., (15, 15)]
    all_actions = product(range(16), range(16))
    # find all legal actions
    legal_actions = list(env.legal_actions())
  
    # return masking
    for action in all_actions:
        yield action in legal_actions

env = RandomOpponentEnv()

model = PPO.load("trainedmodels/A2Cv0_1e6.mdl")

testingenv = QuartoPPO(env=env, model=model)
controltest = RandomPolicy(env=env)

masked_env = ActionMasker(env = env, action_mask_fn = mask_function)
maskedcontrol = MaskedRandomPolicy(env=masked_env)

episodes = 2000

testingenv.test_policy(n_episodes=episodes)
controltest.train(n_episodes=episodes)
maskedcontrol.train(n_episodes=episodes)