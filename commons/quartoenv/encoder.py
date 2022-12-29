from typing import Tuple
from .game import QuartoPiece, QUARTO_DICT

class MoveEncoder:
    """Move Encoding wrapping
    Action is [pos, next]
    """

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
        # convert the Quarto piece into an integer.
        # piece is None during the last move, where there is no pieces left to choose.
        piece_index = piece.index if piece is not None else None
        
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
