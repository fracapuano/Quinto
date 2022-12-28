import gym
import numpy as np
from quartoenv import RandomOpponentEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

def mask_function(env: gym.Env) -> np.ndarray:
    """This function returns the encoding of the valid moves given the actual.
    """
    # unpack legal_actions() in legal_positions and in legal_pieces
    legal_positions = set([action[0] for action in env.legal_actions()])
    legal_pieces = set([action[1] for action in env.legal_actions()])

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

model = MaskablePPO(MaskableActorCriticPolicy, env=env, verbose=2)
timesteps = 1_000_000
model.learn(total_timesteps=timesteps, progress_bar=True)
model.save(f'trainedmodels/maskedPPO_1e6.mdl')