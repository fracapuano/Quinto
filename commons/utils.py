from .policies import *
from typing import Union
from stable_baselines3.common.callbacks import BaseCallback
import os
from typing import Tuple
import wandb

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
        wincounter, losscounter, drawcounter, invalidcounter, matchduration = 0, 0, 0, 0, 0
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
            
            parent_obs = self._env.env._observation
            
            # unpacking parent observation
            board = 16*parent_obs[:-1].reshape((4,4))
            board_image = wandb.Image(board, caption="Board in Terminal State")

            matchduration += info["turn"]/self.n_episodes

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
        wandb.log({
            "Win(%)": 100 * wincounter / self.n_episodes,
            "Loss(%)": 100 * losscounter / self.n_episodes,
            "Draw(%)": 100 * drawcounter / self.n_episodes, 
            "Invalid(%)": 100 * invalidcounter / self.n_episodes,
            "Game Turns": matchduration, 
            "Board": board_image
        })
        
        return True

class UpdateOpponentCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, checkpoints_dir:str="checkpoints/", verbose:int=0):
        super().__init__(verbose)
        self.checkpoint_dir = checkpoints_dir

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `_env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # access last element in listdir - usually mostly trained agent ~ best available
        candidate_model = sorted(os.listdir(self.checkpoint_dir), key=lambda name: int(name.split(".")[0].split("_")[1]))[-1]
        candidate = MaskablePPO.load("checkpoints/" + candidate_model)
        # updating the current opponent with new candidate
        self.model.env.envs[0].update_opponent(candidate)

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

def anti90(x:int, y:int)->tuple:
    """Maps points after a 90-degrees rotation to their original representation."""
    return y, 3-x

def clock_rotation(x:int, y:int, degree:float)->tuple: 
    """Applies clock-wise rotation on the vector whose components are x and y.
    This is done in the sake of restoring the points obtained in the original space.
    
    Args:
        x (int): x-coordinate of the point considered.
        y (int): y-coordinate of the point considered.
        degree (float): degree in which to perform the rotation. Since the aim of this function is reconstruction, 
                        rotation is `clockwise`.
    
    Returns:
        (np.array): point in the old, non-rotated, space.
    """
    if degree==90.:
        return anti90(x,y)
    elif degree==180.:
        return anti90(*clock_rotation(x,y, degree=90))
    elif degree==270.:
        return anti90(*clock_rotation(x,y, degree=180))
    else:
        raise ValueError("Rotations for angles >270 are not yet implemented!")

def antiflip(x,y, verse:str):
    """Applies anti-flipping operation on the vector whose components are x and y.
    This is done in the sake of restoring the points obtained in the original space.
    
    Args:
        x (float): x-coordinate of the point considered.
        y (float): y-coordinate of the point considered.
        verse (str): verse in which to rotate once more the point considered.
    
    Returns:
        (np.array): point in the new, un-flipped, space.
    """
    if verse.lower()=="horizontal" or verse.lower()=="h":
        return np.array([3-x,y])
    elif verse.lower()=="vertical" or verse.lower()=="v":
        return np.array([x,3-y])
    else:
        raise ValueError("Anti-Flip operations are diagonally and horizontally only!")

class QuartoSymmetries:
    def __init__(self):
        # Define a list of all possible symmetries in a game of Quarto
        self.symmetries = {
            # identity
            "identity": lambda x: x,
            # Rotate 90 degrees
            "rot90": lambda x: np.rot90(x, k=1),
            # Rotate 180 degrees
            "rot180": lambda x: np.rot90(x, k=2),
            # Rotate 270 degrees
            "rot270": lambda x: np.rot90(x, k=3)
            # Reflect horizontally
            # "h_flip": lambda x: np.fliplr(x),
            # Reflect vertically
            # "v_flip": lambda x: np.flipud(x)
        }
        # Define inverse symmetries
        self.inverse_symmetries = {
            # identity
            "identity": lambda x,y: (x,y), 
            # Rotate -90 degrees (270 degrees)
            "rot90": lambda x,y : clock_rotation(x,y, degree=90),
            # Rotate -180 degrees (180 degrees)
            "rot180": lambda x,y: clock_rotation(x,y, degree=180),
            # Rotate -270 degrees (90 degrees)
            "rot270": lambda x,y: clock_rotation(x,y, degree=270)
            # Reflect horizontally, once more
            # "h_flip": lambda x,y: antiflip(x,y, verse="horizontal"),
            # Reflect vertically, once more
            # "v_flip": lambda x,y: antiflip(x,y, verse="vertical")
        }

    def apply_symmetries(self, board:np.array)->Tuple[np.array, list, list]:
        """This function applies the Quarto game symmetries to a given board, returning the board considered
        and the function that can be used to re-obtain original form (i.e., the function that when applied on apply_symmetries output
        would output board input once more)

        Args: 
            board (np.array): original board to translate to its canonical form
        
        Returns: 
            Tuple[np.array, list, list]: board in the canonical form, list of functions used for canonization and list of functions that 
            can be used to revert canonization
        """
        inv_symmetries = []
        applied_symmetries = []
        # applies symmetries
        canonical_board = board
        for name, symmetry in self.symmetries.items():
            # applies the symmetry
            sym_board = symmetry(board)
            # induces an order among alternative representations
            if sym_board[0,0] < canonical_board[0,0]:
                canonical_board = sym_board
                # stores the new symmetrical representation
                applied_symmetries.append(name)
                inv_symmetries.append(self.inverse_symmetries[name])

        if not inv_symmetries:
            inv_symmetries.append(self.inverse_symmetries["identity"])
        return canonical_board, [inv_symmetries[-1]]
