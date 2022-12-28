import logging
from typing import Union, Iterable
import collections.abc
import numpy as np
from itertools import product

logger = logging.getLogger(__name__)

class QuartoPiece:
    def __init__(self, number:int):
        """Every piece is described by a set of four binary attributes. 
        Used to create the properties describing each single piece"""
        # to each quarto piece, an integer is associated
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
    
    def __str__(self): 
        piece_string = "-".join(
            [
                "Tall" if self.tall else "Short", 
                "Hollow" if self.hollow else "Full", 
                "Dark" if self.dark else "Light", 
                "Round" if self.round else "Square"
            ]
        )
        return piece_string
    
    def __hash__(self): 
        return hash(str(self))

# to each Quarto piece, we can access its index.
# However, given an index, we cannot access the corresponding Quarto piece.
# Hence, we create a dictionary which maps each integer into a Quarto piece.
# In this way, we can always turn one into the other without having to create multiple
# useless instances of the same QuartoPiece.

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

        # all possible pieces, from QuartoPiece(0) to QuartoPiece(15)
        self.all_pieces = [QuartoPiece(i) for i in range(16)]

        if isinstance(init_board, collections.abc.Iterable):  # start from a given configuration
            # accept board as input
            self.board = init_board
            # if board is given, onnly some pieces will be available 
            self.available_pieces = [piece for piece in self.all_pieces if piece not in self.board]
        else:  
            # initialize board from scratch. By default, an empty spot will be associated with value -1.
            self.board = -1 * np.ones((4,4))
            # all pieces are available
            self.available_pieces = self.all_pieces

    def pieces_from_board(self)->list: 
        """This function returns the pieces currently on the board"""
        flatten_board = self.board.flatten()
        return list(map(lambda ind: None if ind==-1 else QUARTO_DICT[ind], flatten_board))

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
        if piece not in self.available_pieces:
            logger.warn(f"Placing piece {piece.index} with available pieces {'/'.join(str(p.index) for p in self.available_pieces)}")
            return False
        
        # turning position into single coordinates
        x, y = position

        # check if position is available
        logger.debug(f"Willing to play at ({x},{y})")
        if self.board[x, y] != -1:
            # cell is not empty
            logger.warn(f"Not on a free spot {self.board[x, y]} at {x}, {y}")
            return False
        
        # play piece in position. On the board, we are not placing the QuartoPiece, but the corresponding index.
        self.board[x, y] = piece.index
        # remove chosen piece from available ones
        self.available_pieces.remove(piece)
        
        # check if the game is still going on AND next_piece is not available.
        # This means the action you are playing is not valid.
        if not (self.game_over or self.draw) and next_piece not in self.available_pieces:
            # either pieces is empty (no more pieces from which to chose) or game is over
            logger.warn(f"Next piece invalid, chosen {next_piece.index} but valid pieces where {'/'.join(str(p.index) for p in self.available_pieces)}")
            return False
        
        # you played a valid move. The game might also be finished.
        return True  

    @property
    def game_over(self):
        """Check if someone won.

        Returns:
            bool: True if someone won (i.e., 4 pieces sharing a property are aligned), False otherwise.
        """
        # checking over possible rows
        # There are 4 rows and 4 columns
        # Iteratively check if 4 pieces along a line share a common property.
        # Start by checking horizontal and vertical lines.
        for i in range(4):
            if QuartoGame.common(*self.board[i, :]):
                logger.info(f" {i}-th row: common trait")
                return True
            if QuartoGame.common(*self.board[:, i]):
                logger.info(f" {i}-th columns: common trait")
                return True
        
        # Check the 2 diagonals.
        main_diagonal = [self.board[idx, idx] for idx in range(4)]
        inverse_main_diagonal = [self.board[idx, 3-idx] for idx in range(4)]

        if QuartoGame.common(*main_diagonal):
            logger.info("main diagonal: common trait")
            return True

        if QuartoGame.common(*inverse_main_diagonal):
            logger.info("inverse main diagonal: common trait")
            return True
        
        # board is not exhibiting any common trait, nobody won.
        return False

    @property
    def draw(self):
        """ Check if the game is a tie, i.e., all pieces are places but nobody won.

        Returns:
            bool: True if the game is a tie, False otherwise.
        """
        # nobody has won yet AND the board is full.
        return (not self.game_over) and (-1 not in self.board)

    @property
    def free_spots(self):
        """Return free spots on the board.
        
        Returns:
            np.array: array containing tuples (x, y) with the coordinates of the available positions on the board
        """
        return np.argwhere(self.board == -1)  # -1 identifies an empty spot
    
    def common(a, b, c, d):
        """ Given four pieces along a line, check if they share a property.

        Returns:
            bool: True if the 4 pieces share a property, False otherwise.
        """
        # NECESSARY (but not sufficient) CONDITION: there must be 4 pieces along the considered line.
        if -1 in (a, b, c, d):  # one of the considered position is empty
            return False

        # turn the pieces (which are integers atm) into the corresponding Quarto pieces
        piece_a, piece_b, piece_c, piece_d = [QUARTO_DICT[idx] for idx in [a,b,c,d]]
        # check if the 4 Quarto pieces share any of the attributes.
        for attribute in ["tall", "hollow", "dark", "round"]: 
            common_trait = getattr(piece_a, attribute) == getattr(piece_b, attribute) == getattr(piece_c, attribute) == getattr(piece_d, attribute)
            if common_trait: 
                return True        
        # no common trait has been found among the given pieces
        return False

    def get_valid_actions(self) -> np.array:
        """In any situation of the game, returns the available actions as ((x, y), QuartoPiece)
        
        Returns:
            np.array: array of tuples of the type (available position, available piece)
        """
        # retrieve the available positions.
        # These positions are returned as [x, y] coordinates.
        available_pos = self.free_spots
        # return the available pieces.
        # If there is only one available position, then there is no available piece, i.e., self.available_pieces
        # in an empty list. In this case, we want to return None instead of [].
        available_pieces = self.available_pieces if len(available_pos) > 1 else [None]

        # return the array of available actions.
        return np.fromiter(product(available_pos, available_pieces), dtype = tuple)
