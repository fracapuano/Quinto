import gym
import numpy as np
from quartoenv import RandomOpponentEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from itertools import product

# def mask_function(env: gym.Env) -> np.ndarray:
#     """This function returns the encoding of the valid moves given the actual
#     """
#     # work out all actions: [(0, 0), ..., (15, 15)]
#     all_actions = product(range(16), range(16))
#     # find all legal actions
#     legal_actions = list(env.legal_actions())
  
#     return [action in legal_actions for action in all_actions]
#     # # return masking
#     # for action in all_actions:
#     #     yield action in legal_actions

def mask_function(env: gym.Env) -> np.ndarray:
    """This function returns the encoding of the valid moves given the actual.
    """
    # unpack legal_actions() in legal_positions and in legal_pieces
    legal_positions = [action[0] for action in env.legal_actions()]
    legal_pieces = [action[1] for action in env.legal_actions()]

    # convert into masking

    # for each pos in range(16), check if pos is in legal_positions.
    masked_positions = [pos in legal_positions for pos in range(16)]
    # for each piece in range(16), check if piece is in legal pieces
    masked_pieces = [piece in legal_pieces for piece in range(16)]

    # we now have two masking lists of booleans.
    # Put them together into a single numpy array.
    return np.array([masked_positions, masked_pieces], dtype = bool)

env = RandomOpponentEnv()
env = ActionMasker(env = env, action_mask_fn = mask_function)

model = MaskablePPO(policy="MlpPolicy", env=env, verbose=1)
model.learn(total_timesteps=10_000, progress_bar=True)
