from commons.policies import reverseAlgoDict, AlgoDict, mask_function, ActionMasker
from commons.quartoenv import RandomOpponentEnv, RandomOpponentEnv_V1
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
import numpy as np
import random
from commons import *
import argparse

# reproducibility - random seed setted
seed = 777
np.random.seed(seed)
random.seed(seed)

# not saving 5e6 along the way to present results with reduced randomicity
trainsteps_dict = {
    3000: "3e3",
    1e6: "1e6",
    3e6: "3e6", 
    5e6: "5e6"
}

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def parse_args()->object: 
    """Args function. 
    Returns:
        (object): args parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="PPO", type=str, help="RL Algorithm. One in ['PPO', 'A2C', 'maskedPPO']")
    parser.add_argument("--verbose", default=1, type=int, help="Verbosity value")
    parser.add_argument("--train-timesteps", default=1e5, type=float, help="Number of timesteps to train the RL algorithm with")
    parser.add_argument("--test-episodes", default=500, type=int, help="Number of test matches the agent plays during periodic evaluation")
    parser.add_argument("--action-masking", default=False, type=boolean_string, help="Whether or not to perform action masking during training")
    parser.add_argument("--losing-penalty", default=True, type=boolean_string, help="Whether or not to enforce a penalty (negative reward) for losing")
    
    # TO BE REMOVED
    parser.add_argument("--debug", default=True, type=boolean_string, help="Debug mode, ignore all configurations")
    return parser.parse_args()

args = parse_args()

algorithm=args.algorithm
verbose=args.verbose
train_timesteps=args.train_timesteps
test_episodes=args.test_episodes
action_masking=args.action_masking
losing_penalty=args.losing_penalty

if args.debug: 
    algorithm = "PPO"
    verbose=1
    train_timesteps=1e6
    test_episodes=500
    action_masking=True
    losing_penalty=True

# input sanity check
if not action_masking:
    if algorithm.upper() not in ["PPO", "A2C"]:
        print(f"Prompted algorithm (upper): {algorithm.upper()}")
        raise ValueError("Non-action masking algorithm currently supported are ['PPO', 'A2C'] only!")

# create environment in which agent plays against random-playing agent
if losing_penalty:
    env = RandomOpponentEnv_V1()
    version = "v1"
else: 
    env = RandomOpponentEnv()
    version = "v0"

if action_masking:
    # masking action space to those actually available
    env = ActionMasker(env, mask_function)
    # maskable PPO object
    model = MaskablePPO(
        MaskableActorCriticPolicy, 
        env=env, 
        verbose=verbose, 
        seed=seed
    )
else:
    model_function = reverseAlgoDict[algorithm.upper()]
    model = model_function("MlpPolicy", env=env, verbose=verbose, seed=seed)

model = select_model(algorithm=algorithm, version=version, training_timesteps=train_timesteps)

episodes = test_episodes

testingenv = OnPolicy(env=env, model=model)
print("*"*25 + " Policy Randomly sampling action from the whole action space " + "*"*25)
controltest = RandomPolicy(env=env)
controltest.test_policy(n_episodes=episodes, verbose=verbose)

if action_masking:
    print("*"*25 + " Policy Randomly sampling action from masked action space " + "*"*25)
    secondcontrol_test = MaskedRandomPolicy(env=env)
    secondcontrol_test.test_policy(n_episodes=episodes, verbose=verbose)

print("-"*25 + f" Policy acting according to {algorithm} trained over {train_timesteps} timesteps " + "-"*25)
testingenv.test_policy(n_episodes=episodes, verbose=verbose)
