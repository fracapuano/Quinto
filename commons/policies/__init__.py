from .randompolicy import * 
from .masked_randompolicy import *
from .onpolicy_wrapper import *

from stable_baselines3.ppo import PPO
from stable_baselines3.a2c import A2C
from sb3_contrib.ppo_mask import MaskablePPO

Algorithms = [PPO, A2C, MaskablePPO]

AlgoDict = {
    PPO: "PPO", 
    A2C: "A2C", 
    MaskablePPO: "MASKEDPPO"
}

reverseAlgoDict = {
    "PPO": PPO,
    "A2C": A2C,
    "MASKEDPPO": MaskablePPO
}