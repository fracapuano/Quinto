import random
from gym import Wrapper
from .env import *
from .player import Player

class OnePlayerWrapper(Wrapper):
    """The environment reaction to agent's move is emulated with a 
    second player so that the agent can retrieve signal related to the
    consequences of its actions.

    """
    def __init__(self, env:QuartoEnv, other_player:Player):
        super(OnePlayerWrapper, self).__init__(env)
        self.other_player = other_player

    def reset(self):
        obs = self.env.reset()
        self.other_player.reset()
        self.other_first = random.choice([True, False])

        if self.other_first:
            # Make the first step now
            action, _ = self.other_player.predict(obs)
            obs, _, _, _ = self.env.step(action)

        return obs

    def step(self, action:Tuple[int, int])->Tuple:
        obs, self_rew, done, info = self.env.step(action)

        if done:
            if info['invalid']:
                # We just disqualified ourself
                info['winner'] = 'Env'
            else:
                info['winner'] = 'Agent'
            return obs, self_rew, done, info

        # Let other play
        action, _ = self.other_player.predict(obs)
        obs, rew, done, info = self.env.step(action)
        
        if done:
            if info['invalid']:
                # Other player made a bad move, don't reward the Agent
                reward = 0
                info['winner'] = 'Agent'
            elif info['draw']:
                # Same reward for both
                reward = rew
                info['winner'] = 'Draw'
            else:
                # If the second won the game, give negative reward to the agent
                reward = -rew
                info['winner'] = 'Env'
        else:
            reward = self_rew
            info['winner'] = None
        
        return obs, reward, done, info

    def seed(self, seed):
        self.other_player.seed(seed)
        return [seed]
