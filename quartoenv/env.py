import gym
from gym.spaces import MultiDiscrete
import logging
import numpy as np
from typing import Tuple
from .game import QuartoGame, QuartoPiece, QUARTO_DICT

logger = logging.getLogger(__name__)

class QuartoEnv(gym.Env):
    EMPTY = 0
    metadata = {'render.modes':['terminal']}

    action_space, observation_space = None, None

    @property
    def _observation(self):
        """ State of the game after the move.
        """
        return (self.game.board, self.piece)

    @property
    def done(self):
        return self.broken or self.game.game_over or self.game.draw

    def reset(self):
        """This function takes care of resetting the environment to initial state, 
        i.e. empty board with no pieces on."""
        self.game = QuartoGame()
        self.turns = 0
        self.piece = None
        self.broken = False
        return self._observation

    def step(self, action:Tuple[tuple, QuartoPiece])->Tuple:
        """This function steps the environment considering the given action"""
        # sparse rewards - no reward but in terminal states
        # TODO: Might be interesting to reward negatively the agent at each turn to encourage
        #       winning early. Idea
        reward = 0
        # increment number of turns
        self.turns += 1
        info = {'turn': self.turns,
                'invalid': False,
                'win': False,
                'draw': False}
        
        if self.done:
            logger.warn("Actually already done")
            return self._observation, reward, self.done, info

        position, next = action
        logger.debug(f"Received: position: {position}, next: {next}")

        # self.piece stores the piece that has to be positioned on the board. 
        # self.piece is None at first turn, i.e. at the beginning of the game.
        
        if self.piece is not None:  # current move is not move 0
            # play the current piece on the board
            valid = self.game.play(piece=self.piece, position=position, next_piece=next)

            if not valid:
                # Invalid move
                reward = -200
                self.broken = True  # boolean indicator indicating when invalid action is performed
                info['invalid'] = True
            
            # check if played move makes the agent win
            elif self.game.game_over:
                # We just won!
                reward = 100
                info['win'] = True
            
            # check draw
            elif self.game.draw:
                reward = 20
                info['draw'] = True
            else:
                # a valid move was played
                reward = 0

        # Process the next piece
        self.piece = next

        return self._observation, reward, self.done, info

    def render(self, mode, **kwargs):
        for row in self.game.board:
            s = ""
            for piece in row:
                if piece is None:
                    s += ". "
                else:
                    s += str(piece) + " "
            print(s)
        print(f"Next: {self.piece}, Free: {''.join(str(p) for p in self.game.available_pieces)}")
        print()

    @property
    def legal_actions(self):
        return self.game.get_valid_actions()

    def __del__(self):
        self.close()

class MoveEncoderV0(gym.ActionWrapper):
    """Move Encoding wrapping
    Action is [pos, next]
    """

    def __init__(self, env:gym.Env) -> None:
        super(MoveEncoderV0, self).__init__(env)
        # action is [pos, next]
        # both are not null, they are just ignored when irrelevant
        self.action_space = MultiDiscrete([16, 16])

    def action(self, action:Tuple[int, int])->Tuple[tuple, QuartoPiece]:
        """Decode an action of the type (position :int, piece :int) into an action of the type ((x, y), QuartoPiece).

        Args:
            action (Tuple[int, int]): Tuple of two integer entries. Refer to `decode` method for full documentation
                              
        Returns:
            Tuple[tuple, QuartoPiece]: decoded action, as per `decode` function.
        """
        return self.decode(action)

    @property
    def legal_actions(self):
        """Encode the valid actions.

        Yields:
            Iterable: encoded valid actions.
        """
        for action in self.game.get_valid_actions():
            yield self.encode(action)

    def action_masks(self)->tuple:
        """This function returns masks for the currently available actions given the board state
        
        Returns: 
            tuple: Masked positions and pieces for the given board configuration
        """

        # get valid positions
        valid_positions = list(set([pos for pos, _ in self.legal_actions()]))
        # convert valid positions (in their integer representation) to True
        mask_pos = [i in valid_positions for i in range(16)]  # True when move is valid, else oth.

        # get unique pieces
        valid_pieces = list(set([piece for _, piece in self.legal_actions()]))
        # convert valid pieces (in their integer representation) to True
        mask_piece = [i in valid_pieces for i in range(16)]
        
        return (mask_pos, mask_piece)

    def encode(self, action:Tuple[tuple, QuartoPiece])->Tuple[int, int]:
        """Encode an action. Here, an action is a Tuple of ((x, y), QuartoPiece).
        Converts the (x, y) coordinates into the integer position and the QuartoPiece into the corresponding
        integer representation.

        Args:
            action (Tuple[tuple, QuartoPiece]): (Position to be played, Piece chosen for other player).
                                                First entry are the (x, y) coordinates of the position on the board where 
                                                the player wishes to place the piece chosen by the previous player.
                                                Second entry is the QuartoPiece that the next player should place.

        Returns:
            Tuple[tuple, int]: Iterable of 2 entries of the type (integer position, integer piece).
        """
        # unpack the action.
        position, piece = action
        # convert (x, y) coordinates into a single integer
        position_index = position[0] * 4 + position[1]
        # piece is None during the last move, where there is no pieces left to choose.
        if piece is not None:
            # convert the Quarto piece into an integer.
            piece_index = piece.index
        
        return position_index, piece_index


    def decode(self, action:Tuple[int, int])->Tuple[tuple, QuartoPiece]:
        """Decode an action from the actual MultiDiscrete action space. Here, action is an iterable of the type (position, piece),
        where both position and piece are integers. Converts the integer position into (x, y) coordinates and the integer piece 
        into the corresponding Quarto piece.

        Args:
            action (Tuple[int, int]): Action to be decoded.
                                      First entry is an integer referring to the position on the board where the player wishes to 
                                      place the piece chosen by the previous player.
                                      Second entry is an integer referring to the piece that the next player should place.

        Returns:
            tuple: Iterable of 2 entries of the type ((x, y), QuartoPiece)
        """
        # unpack action
        position, piece = action
        # convert integer position into (x, y) on the grid
        position = (position // 4, position % 4)
        # piece is None during the last move, where there is no pieces left to choose.
        if piece is not None:
            # convert the integer into Quarto piece
            piece = QUARTO_DICT[piece]
        return position, piece

class QuartoEnvV0(QuartoEnv):
    """
    Quarto Env supporting move encoding
    That's a subclass and not a wrapper.
    """

    def __init__(self):
        super(QuartoEnvV0, self).__init__()
        """
        Observation space observed at various 
        """
        # observation space: describes board + player hand 
        # board described as 16 cells in which one can find a piece (0-15) or nothing (16)
        # same goes with hand, either a piece (0-15) or nothing (16)
        self.observation_space = MultiDiscrete([16+1] * 16+1)
        
        # actions space: described as move played (cell played) and piece chosen from still
        #                availables.
        # 16: moves that can be chosen; 16: pieces that can be played
        self.action_space = MultiDiscrete([16, 16])

        # move encoder to take care of turning tuples/objects into integers
        self.move_encoder = MoveEncoderV0()

    def step(self, action:Tuple[int,int]):
        """Steps the environment given an action"""
        # decoding and unpacking action
        position, next = self.move_encoder.decode(action=action)
        # performing action on env
        return super(QuartoEnvV0, self).step((position, next))

    @property
    def legal_actions(self):
        for game_action in super(QuartoEnvV0, self).legal_actions:
            yield self.move_encoder.encode(game_action)
