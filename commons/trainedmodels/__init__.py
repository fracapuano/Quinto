from commons.policies import Algorithms, AlgoDict
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
import os

versions = ["v0", "v1"]
# not saving 5e6 along the way to present results with reduced randomicity
steps_dict = {
    3000: "3e3",
    1e6: "1e6",
    3e6: "3e6", 
    5e6: "5e6"
}

def select_model(algorithm:str, version:str="v0", training_timesteps:float=1e6)->OnPolicyAlgorithm: 
    """This function returns a loaded algorithm with a given set of configurations"""
    # turn everything to upper case
    algorithm = algorithm.upper()

    # sanity check on input arguments
    if algorithm not in AlgoDict.values(): 
        raise ValueError(f"{algorithm} not in {Algorithms}!")
    if version not in versions: 
        raise ValueError(f"Version {version} not in {versions}!")
    if training_timesteps not in steps_dict.keys(): 
        raise ValueError(f"Training timesteps {training_timesteps} not among considered ones!")
    
    # usually trained models reside in this folder
    trained_model = "commons/trainedmodels/"
    # accessing the algorithm
    trained_model += algorithm
    # selecting the version
    trained_model += version
    # select particular number of steps
    trained_model += "_" + steps_dict[training_timesteps] + ".mdl"
    try: 
        # accessing the algorithm object given the string
        algo = list(AlgoDict.keys())[list(AlgoDict.values()).index(algorithm)]
        # return trained model using stable_baselines3 API
        return algo.load(trained_model)
        
    except FileNotFoundError: 
        print(f"Attempted at loading {trained_model}")
        available_models = " / ".join(filter(lambda filename: filename.endswith(".mdl"), os.listdir("commons/trainedmodels")))
        raise ValueError(f"With available models: {available_models}")
