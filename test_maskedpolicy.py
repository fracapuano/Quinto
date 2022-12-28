import gym
import numpy as np
from policies import MaskedRandomPolicy
from quartoenv import RandomOpponentEnv
from sb3_contrib.common.wrappers import ActionMasker
from itertools import product

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
env = ActionMasker(env = env, action_mask_fn = mask_function)

policy = MaskedRandomPolicy(env=env)
policy.train(n_episodes = 1_000)
