from stable_baselines3 import PPO
import gym
from tqdm import tqdm

class QuartoPPO:
    def __init__(self, env:gym.Env, model:PPO, verbose:int=1):
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
                action, _ = self.model.predict(obs)
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
