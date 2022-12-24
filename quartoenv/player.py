import logging
import random
from .env import QuartoEnv

logger = logging.getLogger(__name__)

class Player: 
    def reset(self): 
        ...

    def seed(self, seed:int):
        random.seed(seed)

class RandomPlayer(Player):
    def __init__(self, env:QuartoEnv):
        if not isinstance(env, QuartoEnv):
            raise ValueError(f"Env {env} is not a valid env! Submit a `Quarto Env`!")
        
        self.env = env

    def reset(self):
        pass

    def predict(self):
        possible_actions = list(self.env.legal_actions)
        return random.choice(possible_actions), None

class HumanPlayer(Player):

    def __init__(self, env:QuartoEnv): 
        if not isinstance(env, QuartoEnv):
            raise ValueError(f"Env {env} is not a valid env! Submit a `Quarto Env`!")
        
        self.env = env

    def reset(self):
        pass

    def predict(self):
        print("--- YOUR TURN ---")
        # print the board.
        print(self.env.game.board)
        # print available pieces. Print the integer and not the QuartoInstance, which is more informative.
        available_pieces = [piece.index for piece in self.env.game.available_pieces]
        print(available_pieces)

        print(f"Where would you like to place piece {self.env.piece}?")
        input_str = input("Please indicate X and Y, divided by a blank space -> e.g. 0 3")
        input_x, input_y = input_str.split(" ")  # split input command on white space

        position = int(input_x) * 4 + int(input_y)
        input_piece = input("What piece do you want me to place next?")
        
        next_piece = int(input_piece)
    
        return (position, next_piece), None

    def seed(self, seed):
        pass