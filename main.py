# Free for personal or classroom use; see 'LICENSE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
import argparse
import random
from commons.quarto.objects import Player, Quarto
from commons.quartoenv.env import RandomOpponentEnv
from commons.quartoenv.env_v2 import RandomOpponentEnv_V2
from commons.quartoenv.env_v3 import CustomOpponentEnv_V3
from commons.quartoenv.game import QuartoPiece
from sb3_contrib import MaskablePPO
import numpy as np
import time
from tqdm import tqdm
import gym
from sb3_contrib.common.wrappers import ActionMasker
from typing import Tuple

def mask_function(env: gym.Env) -> np.ndarray:
    """This function returns the encoding of the valid moves given the actual.
    """
    # unpack legal_actions() in legal_positions and in legal_pieces
    legal_positions = set([action[0] for action in env.legal_actions()])
    legal_pieces = set([action[1] for action in env.legal_actions()])

    # convert into masking

    # for each pos in range(16), check if pos is in legal_positions.
    masked_positions = [pos in legal_positions for pos in range(16)]
    # for each piece in range(16), check if piece is in legal pieces
    masked_pieces = [piece in legal_pieces for piece in range(16)]

    # we now have two masking lists of booleans.
    # Put them together into a single numpy array.
    return np.array([masked_positions, masked_pieces], dtype = bool)

class RandomPlayer(Player):
    """Random player"""

    def __init__(self, quarto: Quarto) -> None:
        super().__init__(quarto)

    def choose_piece(self) -> int:
        return random.randint(0, 15)

    def place_piece(self) -> Tuple[int, int]:
        return random.randint(0, 3), random.randint(0, 3)

class RLPlayer(Player):
    def __init__(self, quarto: Quarto, model = None) -> None:
        super().__init__(quarto)
        self.env = RandomOpponentEnv_V2()  # interface is used for prediction only, nothing really changes between envs here.

        if model:
            self.model = model
            if isinstance(self.model, MaskablePPO):
                self.uses_masking = True
            else: 
                self.uses_masking = False
        else:
            raise ValueError('Please, pass a valid model')
        self.model.set_env(env = ActionMasker(self.env, mask_function))
        self.action = None
        # dictionary to convert ThePseudo pieces into ours.
        self.indices_dict = {
            0: 0, 
            1: 8, 
            2: 4, 
            3: 12, 
            4: 2, 
            5: 10, 
            6: 6, 
            7: 14,
            8: 1, 
            9: 9, 
            10: 5, 
            11: 13,
            12: 3, 
            13: 11,
            14: 7,
            15: 15,
            -1: -1
        }
        
    def choose_piece(self) -> int:
        # we are just choosing a piece, so we don't care what we have in hand.
        if not self.action:
            self.env.game.board, self.env.piece = self.encode()
            if self.uses_masking:
                action, _ = self.model.predict(self.env._observation, action_masks = mask_function(self.env))
            else:
                action, _ = self.model.predict(self.env._observation)
            self.decode(action)

        return self.action[1]


    def place_piece(self) -> Tuple[int, int]:
        self.env.game.board, self.env.piece = self.encode()
        if self.uses_masking:
                action, _ = self.model.predict(self.env._observation, action_masks = mask_function(self.env))
        else:
            action, _ = self.model.predict(self.env._observation)
        self.decode(action)

        return self.action[0]

    def encode(self):
        # find board and selected_piece with Pseudo's encoding
        board, selected_piece = (self.get_game()._board.T, self.get_game()._Quarto__selected_piece_index)
        # create function to change values of numpy array based on the dictionary
        # first let's change the integers on the board
        changeint_func = np.vectorize(self.indices_dict.get)
        board = changeint_func(board)
        # then, let's turn the board of integers into a board of QuartoPieces
        selected_piece = QuartoPiece(self.indices_dict[selected_piece]) if selected_piece >= 0 else None

        return board, selected_piece  

    def decode(self, action):
        position = (action[0] // 4, action[0] % 4)
        next_piece = [key for key, val in self.indices_dict.items() if val == action[1]][0]

        self.action = (position, next_piece)


def main():
    palmares = {0 : 0, -1 : 0, 1 : 0}
    for _ in tqdm(range(50)):
        # create game
        game = Quarto()
        # create player
        playerRL = RLPlayer(game, MaskablePPO.load(
            'commons/trainedmodels/MASKEDPPOv3_130e6.zip',
            custom_objects= {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0, 
                "clip_range": lambda _: 0.0
            }
        ))
        # set players
        game.set_players((playerRL, RandomPlayer(game)))
        # run a match
        winner = game.run()
        # score results
        palmares[winner] += 1

        del game, playerRL
    
    print(palmares)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help='increase log verbosity')
    parser.add_argument('-d',
                        '--debug',
                        action='store_const',
                        dest='verbose',
                        const=2,
                        help='log debug messages (same as -vv)')
    args = parser.parse_args()

    if args.verbose == 0:
        logging.getLogger().setLevel(level=logging.WARNING)
    elif args.verbose == 1:
        logging.getLogger().setLevel(level=logging.INFO)
    elif args.verbose == 2:
        logging.getLogger().setLevel(level=logging.DEBUG)

    main()
