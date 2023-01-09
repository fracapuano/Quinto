from commons.policies import reverseAlgoDict, AlgoDict, mask_function, ActionMasker
from commons.quartoenv import RandomOpponentEnv, RandomOpponentEnv_V1, RandomOpponentEnv_V2
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from commons.utils import WinPercentageCallback
from itertools import compress
import numpy as np
import random

import argparse


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
    parser.add_argument("--verbose", default=0, type=int, help="Verbosity value")
    parser.add_argument("--train-timesteps", default=1e5, type=float, help="Number of timesteps to train the RL algorithm with")
    parser.add_argument("--evaluate_while_training", default=True, type=boolean_string, help="Whether or not to evaluate the RL algorithm while training")
    parser.add_argument("--store-checkpoints", default=True, type = boolean_string, help="Whether or not to store partially-trained models. Recommended True for long trainings (>1e6 ts)")
    parser.add_argument("--evaluation-frequency", default=1e3, type = float, help="Frequency with which to evaluate policy against random fair opponent")
    parser.add_argument("--test-episodes", default=50, type=int, help="Number of test matches the agent plays during periodic evaluation")
    parser.add_argument("--action-masking", default=False, type=boolean_string, help="Whether or not to perform action masking during training")
    parser.add_argument("--losing-penalty", default=True, type=boolean_string, help="Whether or not to enforce a penalty (negative reward) for losing")
    parser.add_argument("--duration-penalty", default=True, type=boolean_string, help="Whether or not to enforce a penalty (negative reward) on long games")
    parser.add_argument("--show-progressbar", default=True, type=boolean_string, help="Whether or not to display a progressbar during training")
    parser.add_argument("--save-model", default=False, type=boolean_string, help="Whether or not save the model currently trained")

    # TO BE REMOVED
    parser.add_argument("--debug", default=True, type=boolean_string, help="Debug mode, ignore all configurations")
    return parser.parse_args()

args = parse_args()

algorithm=args.algorithm
verbose=args.verbose
train_timesteps=args.train_timesteps
evaluate_while_training=args.evaluate_while_training
store_checkpoints=args.store_checkpoints
evaluation_frequency=args.evaluation_frequency
test_episodes=args.test_episodes
action_masking=args.action_masking
losing_penalty=args.losing_penalty
duration_penalty=args.duration_penalty
show_progressbar=args.show_progressbar
save_model=args.save_model

if args.debug: 
    algorithm = "maskedPPO"
    verbose=0
    train_timesteps=3000
    evaluate_while_training=True
    store_checkpoints=True
    evaluation_frequency=10
    test_episodes=5
    action_masking=True
    losing_penalty=True
    duration_penalty=True
    show_progressbar=True
    save_model=True

def main(): 
    # reproducibility - random seed setted
    seed = 777
    np.random.seed(seed)
    random.seed(seed)

    # input sanity check
    if not action_masking:
        if algorithm.upper() not in ["PPO", "A2C"]:
            print(f"Prompted algorithm (upper): {algorithm.upper()}")
            raise ValueError("Non-action masking algorithm currently supported are ['PPO', 'A2C'] only!")

    # create environment in which agent plays against random-playing agent
    if losing_penalty:
        env = RandomOpponentEnv_V1()
        version = "v1"
        if duration_penalty: 
            env = RandomOpponentEnv_V2()
            version = "v2"
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

    model_name = algorithm.upper() + version + "_" + trainsteps_dict[train_timesteps]

    # saving model every 5e5 timesteps
    checkpoint_save = CheckpointCallback(
        save_freq=500_000, save_path="checkpoints/", name_prefix=f"{algorithm}"
    )
    # saving the percentage of wins a model can achieve in n_episodes
    winpercentage = WinPercentageCallback(env=env, n_episodes=test_episodes, logfile=f"logs/{model_name}_logfile.txt")
    # evaluating the environment periodically every evaluation_frequency timesteps
    evaluation_callback = EveryNTimesteps(n_steps=evaluation_frequency, callback=winpercentage)

    callback_list = [
        checkpoint_save, 
        evaluation_callback
    ]
    callback_mask = [store_checkpoints, evaluate_while_training]
    # masking callbacks considering script input
    callback_list = list(compress(callback_list, callback_mask))

    # training the model with train_timesteps
    model.learn(total_timesteps=train_timesteps, callback=callback_list, progress_bar=show_progressbar)
    if save_model: 
        model.save(f"commons/trainedmodels/{model_name}.mdl")

if __name__ == "__main__": 
    main()
