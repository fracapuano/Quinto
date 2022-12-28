from stable_baselines3 import A2C 
from quartoenv import RandomOpponentEnv
from policies import RandomPolicy
from sb3_contrib.ppo_mask import MaskablePPO
env = RandomOpponentEnv()
import gym
from tqdm import tqdm
import numpy as np

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

class QuartoPPO: 
    def __init__(self, env:gym.Env, model:A2C, verbose:int=1):
        self.env = env
        self.model = model
        self.verbose = verbose
    
    def test_policy(self, n_episodes:int): 
        """Test trained policy in `model` for `n_episodes`"""
        wincounter, losscounter, drawcounter, invalidcounter = 0, 0, 0, 0

        for episode in tqdm(range(n_episodes)): 
            obs = self.env.reset()
            done = False
            while not done: 
                action, _ = self.model.predict(obs, action_masks = mask_function(self.env))
                obs, _, done, info = self.env.step(action=action)

            if info["win"]: 
                wincounter += 1
            elif info["draw"]: 
                drawcounter += 1
            elif info["invalid"]: 
                invalidcounter += 1
            elif info.get("loss", None):
                losscounter += 1
        
        if self.verbose: 
            print(f"Out of {n_episodes} testing episodes:")
            print("\t (%) games ended for an invalid move: {:.4f}".format(100 * invalidcounter/n_episodes))
            print("\t (%) games won by the agent: {:.4f}".format(100*wincounter/n_episodes))
            print("\t (%) games drawn: {:.4f}".format(100*drawcounter/n_episodes))
            print("\t (%) games lost by the agent: {:.4f}".format(100*losscounter/n_episodes))

model = MaskablePPO.load("trainedmodels/maskedPPO_1e6.mdl")

testingenv = QuartoPPO(env=env, model=model)
controltest = RandomPolicy(env=env)

episodes = 1000

testingenv.test_policy(n_episodes=episodes)
# controltest.train(n_episodes=episodes)