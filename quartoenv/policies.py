import gym
from .env import *
from sb3_contrib.common.wrappers import ActionMasker

class RandomPolicy: 
	def __init__(self, env: gym.Env, verbose:int=1): 
		self.env = env
		self.verbose = 1

	def train(self, n_episodes:int, render:bool=False): 
		"""This function emulates the usual episodic-training framework for the agent. 
		However, in this case no policy is actually trained (that is, no learning actually takes 
		place) since the agent selects actions randomly. 
		
		Args: 
			n_episodes(int): Number of training episodes to use.
			render (bool, optional): Whether or not to render the environment at each step.
		"""
		invalid_counter = list()

		for _ in range(n_episodes): 
			done = False
			_ = self.env.reset()

			while not done:
				action = self.env.action_space.sample()	# Sample random action - non masked on valid actions. 
				_, _, done, info = self.env.step(action)	# Step the simulator to the next timestep

			invalid_counter.append(+1 if info["invalid"] else 0)
			
			if render:
				self.env.render()
		
		if self.verbose: 
			print("Percentage of episodes ending because an " + 
			"invalid action has been chosen: {:.4f} %".format(100 * sum(invalid_counter)/n_episodes)
			)

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
		
		for episode in range(n_episodes):
			done = False
			obs = self.env.reset()
			while not done:
				possible_actions = list(self.env.action_masks())
				action = random.choice(possible_actions)
				
				if self.verbose: 
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
		