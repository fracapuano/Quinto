import logging
from typing import Union, Iterable
import collections.abc
import numpy as np

logger = logging.getLogger(__name__)

class QuartoPiece:
    def __init__(self, number:int):
        """Every piece is described by a set of four binary attributes. 
        Used to create the properties describing each single piece"""
        self.index = number
        # each piece is either tall or short...
        self.tall = bool(number & 1)
        # ... is hollow or full...
        self.hollow = bool(number & 2)
        # ... is dark or light...
        self.dark = bool(number & 4)
        # ...and it is round or square
        self.round = bool(number & 8)

    def __eq__(self, other):
        if not isinstance(other, QuartoPiece):
            raise ValueError(f"Other {other} is not an instance of QuartoPiece!")
        
        # checks if the two pieces have the same set of binary properties
        are_equal = self.tall == other.tall and self.hollow == other.hollow and self.dark == other.dark and self.round == other.round
        return are_equal
    
    """TODO: Turn meaningless letters into Tall-Hollow-Dark-Round..."""
    def __str__(self):
        return {
            (False, False, False, False): "A",
            (False, False, False, True): "B",
            (False, False, True, False): "C",
            (False, False, True, True): "D",
            (False, True, False, False): "E",
            (False, True, False, True): "F",
            (False, True, True, False): "G",
            (False, True, True, True): "H",
            (True, False, False, False): "a",
            (True, False, False, True): "b",
            (True, False, True, False): "c",
            (True, False, True, True): "d",
            (True, True, False, False): "e",
            (True, True, False, True): "f",
            (True, True, True, False): "g",
            (True, True, True, True): "h"
        }[self.tall, self.hollow, self.dark, self.round]

QUARTO_DICT = {
    idx: QuartoPiece(idx) for idx in range(16)
}


class QuartoGame:
    def __init__(self, init_board:Union[bool, Iterable[QuartoPiece]]=True):
        """Init function.
        
        Args: 
            init_board (Union[bool, Iterable[QuartoPiece]], optional): Whether to initialize the board from scratch or take init_board as 
                                                                       starting conditon. Defaults to True (initialize the board from scratch).
        """
        "TODO: Add check on initboard iterable containing only QuartoPiece objects"

        self.all_pieces = [QuartoPiece(i) for i in range(16)]

        if isinstance(init_board, collections.abc.Iterable):  # start from a given configuration
            # accept board as input
            self.board = init_board 
            self.available = [piece for piece in self.all_pieces if piece not in self.board]
        else:  
            # initialize board from scratch
            self.board = np.zeros((4,4))
            self.available = self.all_pieces

    def play(self, piece: QuartoPiece, position: int, next_piece: QuartoPiece) -> bool:
        """This function plays a move given piece and position as per game rules. Moreover, it also removes 
        a selected piece from the ones still available.
        
        Args: 
            piece (QuartoPiece): Object representing the piece that has to be moved. 
            position (tuple): Tuple of indices corresponding to the cell in the board in which `piece` has to be moved.
            next_piece (QuartoPiece): Piece chosen by the player which plays this move. 
            
        Returns: 
            bool: Boolean value corresponding to whether or not the game has ended. False when not in terminal state
                  (i.e., game continues), True otherwise.
        """
        if piece is None or not isinstance(piece, QuartoPiece): 
            raise ValueError(f"Piece {piece} is not an instance of QuartoPiece!")
        
        # check if piece is available
        if piece not in self.available:
            logger.warn(f"Not placing a free piece, {piece}, {''.join(str(p) for p in self.free)}")
            # still playing
            return False
        
        # turning position into single coordinates
        x, y = position

        # check if position free
        logger.debug(f"Willing to play at ({x},{y})")
        if self.board[x, y] != 0:
            # cell is not empty
            logger.warn(f"Not on a free spot {self.board[x, y]} at {x}, {y}")
            return False
        
        # play piece in position
        self.board[x, y] = piece.index
        # remove chosen piece from available ones
        self.available.remove(piece)
        
        # check if game is in terminal state
        if not (self.game_over or self.draw) and next_piece not in self.available:
            # either pieces is empty (no more pieces from which to chose) or game is over
            logger.warn(f"Next piece invalid, {next_piece}, {''.join(str(p) for p in self.free)}")
            return False
        
        return True  # terminal state

    @property
    def game_over(self):
        """This checks if the game is in terminal state because a terminal configuration
        (one exibing one common traits on any row-column or main diagonal) has been reached.
        """
        # checking over possible rows
        for i in range(4):
            if QuartoGame.common(*self.board[i, :]):
                logger.info(f" {i}-th row: common trait")
                return True
            if QuartoGame.common(*self.board[:, i]):
                logger.info(f" {i}-th columns: common trait")
                return True
        
        # checking over the two diagonals
        main_diagonal = [self.board[idx, idx] for idx in range(4)]
        inverse_main_diagonal = [self.board[idx, 3-idx] for idx in range(4)]

        if QuartoGame.common(*main_diagonal):
            logger.info("main diagonal: common trait")
            return True

        if QuartoGame.common(*inverse_main_diagonal):
            logger.info("inverse main diagonal: common trait")
            return True
        
        # board is not exhibiting any common trait
        return False

    @property
    def draw(self):
        """ Game is finished but no one won
        """
        # board is full of pieces not exhibiting any common trait
        return (not self.game_over) and (0 not in self.board)

    @property
    def free_spots(self):
        """Return free spots on the board"""
        return np.argwhere(self.board == 0)  # 0 identifies an empty spot
    
    def common(a, b, c, d):
        """ The four piece have a common property
        """
        if 0 in (a, b, c, d):  # one of the considered position is empty
            return False

        """TODO: Turn getattr(a, attr) == ... == getattr(d, attr) into something one could be proud of"""
        piece_a, piece_b, piece_c, piece_d = [QUARTO_DICT[idx] for idx in [a,b,c,d]]
        for attribute in ["tall", "hollow", "dark", "round"]: 
            common_trait = getattr(piece_a, attribute) == getattr(piece_b, attribute) == getattr(piece_c, attribute) == getattr(piece_d, attribute)
            if common_trait: 
                return True
        
        # no common trait has been found among the given pieces
        return False