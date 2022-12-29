from .base_env import QuartoBase
from .encoder import MoveEncoder
from gym.spaces import MultiDiscrete
import logging
import numpy as np
from typing import Tuple
from .game import QUARTO_DICT
from itertools import product
import random

logger = logging.getLogger(__name__)

class RandomOpponentEnv_V1(QuartoBase):
    """
    Quarto Env supporting state and action encoding
    That's a subclass and not a wrapper.
    Moreover, this class also models a random opponent always playing valid moves.
    """
    def __init__(self):
        """
        State space: describes board + player hand 
                           board described as 16 cells in which one can find a piece (0-15) or nothing (16)
                           same goes with hand, either a piece (0-15) or nothing (16)
        
        Actions space: described as move played (cell played) and piece chosen from still
                       availables. 
                       16: moves that can be chosen; 16: pieces that can be played
        """
        super().__init__()
        # observation space ~ state space
        self.observation_space = MultiDiscrete([16+1] * (16+1))
        # action space
        self.action_space = MultiDiscrete([16, 16])
        # move encoder to take care of turning tuples/objects into integers
        self.move_encoder = MoveEncoder()

    @property
    def _observation(self):
        """Observation is returned in the observed space composed of integers"""
        # accessing parent observation (board, current_piece)
        parent_obs = super()._observation
        # unpacking parent observation
        board, current_piece = parent_obs
        # turning parent observation into a point of the observation space here defined
        board_pieces = np.fromiter(map(lambda el: 16 if el == -1 else el, board.flatten()), dtype=int)
        hand_piece = current_piece.index if current_piece else 16
        
        return np.append(arr=board_pieces, values=hand_piece)
    
    def available_pieces(self)->list:
        """This function returns the pieces currently available. Those are defined as all the pieces
        available but the one each player has in hand and the ones on the board.
        
        Returns: 
            list: List of integers representing (through QUARTO_DICT) QuartoPiece(s)."""

        # retrieve the available pieces as difference between all pieces and pieces on the board
        all_pieces = set(range(16))
        current_board, current_piece = self._observation[:-1], self._observation[-1]
        # available pieces are all pieces but the ones on the board and in hand
        nonavailable_pieces = set(current_board) | {current_piece}
        available_pieces = all_pieces - nonavailable_pieces
        
        return available_pieces

    def legal_actions(self): 
        """This function returns all the legal actions given the present state encoded as int-int tuples.
        
        Yields: 
            (int, int): Tuple encoding position and piece in their integer version.
        """
        # freecells are cells with no piece inside
        freecells = self.game.free_spots
        # available pieces are those that have not been put on the board
        available_pieces = list(
            map(lambda el: QUARTO_DICT[el], self.available_pieces())) \
                if len(self.available_pieces()) > 0 \
                else [None]
        
        # a legal action is of the kind ((x,y), QuartoPiece)
        for legal_action in product(freecells, available_pieces): 
            yield self.move_encoder.encode(legal_action)

    def get_observation(self): 
        return self._observation
    
    def reset(self): 
        """Resets env"""
        super().reset_state()
        return self._observation
    
    def step(self, action:Tuple[int,int]):
        """Steps the environment given an action"""
        # decoding and unpacking action
        position, next = self.move_encoder.decode(action=action)
        # performing agent's action on env
        
        # agent_ply
        _, reward, _, info = super().step((position, next))
        if not self.done:
            # opponent's reply
            random_move = self.move_encoder.decode(random.choice(list(self.legal_actions())))
            # stepping env with random player move - not interested in opponent's perspective
            super().step(random_move)
            
            if self.done: 
                info["loss"] = True
                reward = -1

        return self._observation, reward, self.done, info
