import gym
import numpy as np
import random

from quartoenv import *

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO


def mask_function(env: gym.Env) -> np.ndarray:
    """This function returns the encoding of the valid moves given the actual
    """
    return env.legal_actions()

env = RandomOpponentEnv()  # initialize env
env = ActionMasker(env=env, action_mask_fn=mask_function)  # Wrap to enable masking
env.reset()

done = False
while not done:
    action = random.choice(list(env.action_masks()))
    
    print(f"Pieces still available: {len(list(env.game.available_pieces))}")
    print(f"Next piece chosen: {action[1]}")
    print(f"Pieces still available: {'/'.join([str(p.index) for p in env.game.available_pieces])}'")

    obs, reward, done, info = env.step(action)
    
    print(done)
    
    print("*"*50)
    if done: 
        print(info)


    