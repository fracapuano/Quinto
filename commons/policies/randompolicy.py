import gym
from rich.progress import track

class RandomPolicy: 
	def __init__(self, env: gym.Env): 
		self.env = env

	def test_policy(self, n_episodes:int, render:bool=False, verbose:int=1): 
		"""This function emulates the usual episodic-training framework for the agent. 
		However, in this case no policy is actually trained (that is, no learning actually takes 
		place) since the agent selects actions randomly. 
		
		Args: 
			n_episodes(int): Number of training episodes to use.
			render (bool, optional): Whether or not to render the environment at each step.
		"""
		wincounter, drawcounter, losscounter, invalidcounter = 0, 0, 0, 0
		for _ in track(range(n_episodes)):
			done = False
			_ = self.env.reset()

			while not done:
				action = self.env.action_space.sample()	# Sample random action - non masked on valid actions. 
				_, _, done, info = self.env.step(action)	# Step the simulator to the next timestep

			if info["win"]: 
				wincounter += 1
			elif info["draw"]: 
				drawcounter += 1
			elif info.get("loss", None):
				losscounter += 1
			elif info["invalid"]: 
				invalidcounter += 1
			
			if render:
				self.env.render()
		
		if verbose: 
			print(f"Out of {n_episodes} testing episodes:")
			print("Playing against a random opponent:")
			print("\t(%) games ended for an invalid move: {:.4f}".format(100 * invalidcounter/n_episodes))
			print("\t(%) won games: {:.4f}".format(100 * wincounter/n_episodes))
			print("\t(%) drawn games: {:.4f}".format(100 * drawcounter/n_episodes))
			print("\t(%) lost games: {:.4f}".format(100 * losscounter/n_episodes))
