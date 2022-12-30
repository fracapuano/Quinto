from .policies import *
from typing import Union
from stable_baselines3.common.callbacks import BaseCallback

class WinPercentageCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, env:ActionMasker, verbose=0, n_episodes:int=1000, logfile:str="logtraining.txt"):
        super(WinPercentageCallback, self).__init__(verbose)
        self._env = env
        self.n_episodes = n_episodes
        self.logfile = logfile
        with open(self.logfile, "w") as training_file: 
            training_file.write("# timesteps,(%) wins,(%) losses,(%) draws,(%) invalid\n")
        
        self._env.reset()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `_env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        wincounter, losscounter, drawcounter, invalidcounter = 0, 0, 0, 0
        for episode in range(self.n_episodes):
            obs = self._env.reset()
            done = False
            while not done:
                # either performing a masked action or not
                if isinstance(self.model, MaskablePPO):
                    action, _ = self.model.predict(obs, action_masks = mask_function(self._env))
                else:
                    action, _ = self.model.predict(obs)
                # stepping the environment with the considered action 
                obs, _, done, info = self._env.step(action=action)

            if info["win"]: 
                wincounter += 1
            elif info.get("loss", None):
                losscounter += 1
            elif info["draw"]: 
                drawcounter += 1
            elif info["invalid"]: 
                invalidcounter += 1

        # resettin the environment at the end of testing phase
        self._env.reset()
            
        with open(self.logfile, "a") as training_file: 
            training_file.write(
                "\n{}, {}, {}, {}, {}".format(
                    self.num_timesteps, 
                    100 * wincounter / self.n_episodes, 
                    100 * losscounter / self.n_episodes, 
                    100 * drawcounter / self.n_episodes, 
                    100 * invalidcounter / self.n_episodes
                )
            )
        return True

def preprocess_policy(_env:gym.Env, policy:Union[OnPolicyAlgorithm, str])->Union[MaskedRandomPolicy, RandomPolicy, OnPolicy]:
        """This function pre-processes the input policy returning an instance of the policy that
        exposes a `train_policy` method.

        Args: 
            policy (Union[OnPolicyAlgorithm, str]): Either an OnPolicyAlgorithm (from stable_baselines3) or a string
                                                    identifying one of the dummy policies here implemented for testing.
        
        Raises: 
            ValueError: When policy is a string not in ["fair-random", "pure-random"].
            Warning: When policy is an OnPolicyAlgorithm (from stable_baselines3) that is not among the trained algorithms here. 
                     Implemented algorithms are in __init__ function of policies module.

        Returns: 
            Union[MaskedRandomPolicy, RandomPolicy, QuartoOnPolicy]: Preprocessed Policy Object.
        """
        if isinstance(policy, str): 
            if policy.lower() not in ["fair-random", "pure-random"]:
                raise ValueError(f"Policy {policy} not in ['fair-random', 'pure-random']!")
            else:
                if policy.lower() == "fair-random": 
                    actual_policy = MaskedRandomPolicy(_env=_env)
                elif policy.lower() == "pure-random": 
                    actual_policy = RandomPolicy(_env=_env)
        
        elif isinstance(policy, OnPolicyAlgorithm):
            if type(policy) not in Algorithms:
                print("Policy algorithm is not among already implemented ones!")
                raise Warning(f"Implemented algorithms are: {'/'.join([AlgoDict[algo] for algo in Algorithms])}")
            actual_policy = OnPolicy(_env=_env, model=policy)
        
        return actual_policy

class QuartoPolicy:
    "Base class wrapping various policies for model evaluation"
    def __init__(self, _env:gym.Env, policy:Union[OnPolicyAlgorithm, str]="pure-random", verbose:int=1):
        self._env = _env
        self._policy = preprocess_policy(_env=self._env, policy=policy)
        self.verbose = verbose

    @property
    def _env(self): 
        return self._env
    
    def test_policy(self, n_episodes:int=1_000, verbose:int=1)->None: 
        """Tests considered policy over `n_episodes`"""
        return self._policy.test_policy(n_episodes=n_episodes, verbose=verbose)
