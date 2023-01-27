import gym
import logging
from typing import Tuple
from .game import QuartoGame, QuartoPiece

logger = logging.getLogger(__name__)

class QuartoBase(gym.Env):
    def __init__(self): 
        
        self.game = QuartoGame()
        self.turns = 0
        self.piece = None
        self.broken = False
        self.EMPTY = 0
        self.metadata = {'render.modes':['human', 'terminal']}

    @property
    def _observation(self):
        """ State of the game after the move.
        """
        return (self.game.board, self.piece)

    @property
    def done(self):
        return self.broken or self.game.game_over or self.game.draw

    def reset_state(self):
        """This function takes care of resetting the environment to initial state, 
        i.e. empty board with no pieces on."""
        self.game = QuartoGame()
        self.turns = 0
        self.piece = None
        self.broken = False

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
        
        if self.piece is not None:
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
                reward = +1
                info['win'] = True
            
            # check draw
            elif self.game.draw:
                reward = 0.2
                info['draw'] = True
            else:
                # a valid move was played
                reward = 0
        
        # Process the next piece
        self.piece = next
        
        return self._observation, reward, self.done, info

    def render(self, mode:str="human", **kwargs):
        "Renders board printing to standard output pieces in their encoding"
        for row in self.game.board:
            s = ""
            for piece in row:
                if piece is None:
                    s += ". "
                else:
                    s += str(piece) + " "
            print(s)

        print(f"Next: {str(self.piece.index)}, Free: {'/'.join(str(p.index) for p in self.available_pieces())}")

    def __del__(self):
        self.close()
