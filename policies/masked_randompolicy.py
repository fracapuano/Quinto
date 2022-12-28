import random
from sb3_contrib.common.wrappers import ActionMasker
from tqdm import tqdm

class MaskedRandomPolicy: 
	def __init__(self, env:ActionMasker, verbose:int=1): 
		self.env = env
		self.verbose = verbose
	
	def train(self, n_episodes:int=1_000, render:bool=False): 
		"""This function emulates the usual episodic-training framework for the agent. 
		However, in this case no policy is actually trained (that is, no learning actually takes 
		place) since the agent selects actions randomly from the subset of valid actions given each
		board configuration. 
		
		Args: 
			n_episodes(int): Number of training episodes to use.
			render (bool, optional): Whether or not to render the environment at each step.
		"""
		wincounter, drawcounter, losscounter = 0, 0, 0
		
		for episode in tqdm(range(n_episodes)):
			done = False
			obs = self.env.reset()
			while not done:
				possible_actions = list(self.env.action_masks())
				action = random.choice(possible_actions)
				
				if self.verbose > 1: 
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
		
		if self.verbose: 
			print("Playing against a random opponent:")

			print("\t(%) won games: {:.4f}".format(100 * wincounter/n_episodes))
			print("\t(%) drawn games: {:.4f}".format(100 * drawcounter/n_episodes))
			print("\t(%) lost games: {:.4f}".format(100 * losscounter/n_episodes))
