from .env import RandomOpponentEnv
from .player import RandomPlayer
from .wrapper import OnePlayerWrapper

def randomplayer_env():
    """Returns environment in which an agent playing random moves only plays"""
    env = RandomOpponentEnv()
    player = RandomPlayer(env)
    env = OnePlayerWrapper(env, player)
    
    return env
