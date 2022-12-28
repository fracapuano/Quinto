import gym
import numpy as np
from quartoenv import RandomOpponentEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

def mask_function(env: gym.Env) -> np.ndarray:
    """This function returns the encoding of the valid moves given the actual
    """
    return env.legal_actions()

env = RandomOpponentEnv()
env = ActionMasker(env = env, action_mask_fn=mask_function)

model = MaskablePPO(policy="MlpPolicy", env=env, verbose=1)
model.learn(total_timesteps=10_000, progress_bar=True)
