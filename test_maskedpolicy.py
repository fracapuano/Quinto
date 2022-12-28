import gym
import numpy as np
from policies import MaskedRandomPolicy
from quartoenv import RandomOpponentEnv
from sb3_contrib.common.wrappers import ActionMasker

def mask_function(env: gym.Env) -> np.ndarray:
    """This function returns the encoding of the valid moves given the actual
    """
    return env.legal_actions()

env = RandomOpponentEnv()
env = ActionMasker(env = env, action_mask_fn=mask_function)

policy = MaskedRandomPolicy(env=env)
policy.train(n_episodes=5_000)
