import gym
import numpy as np
from quartoenv import *
from quartoenv.policies import *

from sb3_contrib.common.wrappers import ActionMasker

def mask_function(env: gym.Env) -> np.ndarray:
    """This function returns the encoding of the valid moves given the actual
    """
    return env.legal_actions()

env = RandomOpponentEnv()  # initialize env

#policy = RandomPolicy(env=env)
#policy.train(n_episodes=1000)

env = ActionMasker(env=env, action_mask_fn=mask_function)  # Wrap to enable masking
policy = MaskedRandomPolicy(env=env)

policy.train()