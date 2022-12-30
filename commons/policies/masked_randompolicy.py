import random
from sb3_contrib.common.wrappers import ActionMasker
from rich.progress import track
from itertools import product, compress
import gym
import numpy as np

def mask_function(env: gym.Env) -> np.ndarray:
    """This function returns the encoding of the valid moves given the actual
    """
    # work out all actions: [(0, 0), ..., (15, 15)]
    all_actions = product(range(16), range(16))
    # find all legal actions
    legal_actions = list(env.legal_actions())
  
    return [action in legal_actions for action in all_actions]

class MaskedRandomPolicy: 
	def __init__(self, env:ActionMasker): 
		self.env = env
	
	def test_policy(self, n_episodes:int=1_000, render:bool=False, verbose:int=1): 
		"""This function emulates the usual episodic-training framework for the agent. 
		However, in this case no policy is actually trained (that is, no learning actually takes 
		place) since the agent selects actions randomly from the subset of valid actions given each
		board configuration. 
		
		Args: 
			n_episodes(int): Number of training episodes to use.
			render (bool, optional): Whether or not to render the environment at each step.
		"""
		wincounter, drawcounter, losscounter, invalidcounter = 0, 0, 0, 0
		
		for episode in range(n_episodes):
			done = False
			_ = self.env.reset()
			while not done:
				# mask all actions
				possible_actions = list(compress(product(range(16), range(16)), mask_function(self.env)))
				# edge case: when we are left with only one position on the board. The move is "forced"
				if len(possible_actions) == 0:
					# the only available move can be found in the environment legal actions
					possible_actions = list(self.env.legal_actions())
				# choose one legal move at random
				action = random.choice(possible_actions)
				
				if verbose > 1:
					print(f"Pieces still available: {len(list(self.env.available_pieces()))}")
					print(f"Next piece chosen: {action[1]}")
					print(f"Pieces still available: {'/'.join(sorted([str(p.index) for p in self.env.available_pieces()]))}")

				_, _, done, info = self.env.step(action)
			
			if info["win"]: 
				wincounter += 1
			elif info["draw"]: 
				drawcounter += 1
			elif info["loss"]: 
				losscounter += 1
			elif info["invalid"]: 
				invalidcounter += 1
		
		if verbose:
			print(f"Out of {n_episodes} testing episodes:")
			print("Playing against a random opponent:")
			print("\t(%) games ended for an invalid move: {:.4f}".format(100 * invalidcounter/n_episodes))
			print("\t(%) won games: {:.4f}".format(100 * wincounter/n_episodes))
			print("\t(%) drawn games: {:.4f}".format(100 * drawcounter/n_episodes))
			print("\t(%) lost games: {:.4f}".format(100 * losscounter/n_episodes))
