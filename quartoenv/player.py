import logging
import random
from .game import QuartoPiece, QuartoGame
from .env import QuartoEnv

logger = logging.getLogger(__name__)

class RandomPlayer:
    def __init__(self, env):
        assert isinstance(env.unwrapped, QuartoEnv), env
        self.env = env

    def reset(self):
        pass

    def predict(self):
        possible_actions = list(self.env.legal_actions)
        return random.choice(possible_actions), None

    def seed(self, seed):
        random.seed(seed)

class HumanPlayer:
    def reset(self):
        pass

    def predict(self):
        print("--- YOUR TURN ---")
        # print the board.
        print(self.env.game.board)
        # print available pieces. Print the integer and not the QuartoInstance, which is more informative.
        available_pieces = [piece.index for piece in self.env.game.available_pieces]
        print(available_pieces)
        # TODO: WE NEED TO FIND A WAY TO INCLUDE THE OLD PIECE, SO THE PLAYER KNOWS WHAT THEY HAVE TO PLACE.
        input_x, input_y = input("Where would you like to place the piece? Please indicate X and Y, divided by a blank space -> e.g. 0 3")
        position = int(input_x) * 4 + int(input_y)
        input_piece = input("What piece do you want me to place next?")
        next_piece = input_piece.index
    
        return (position, next_piece), None

    def seed(self, seed):
        pass