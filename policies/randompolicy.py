import gym

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
