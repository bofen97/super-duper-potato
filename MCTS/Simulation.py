import numpy as np
from utils import check_sequence

class Move(object):
    def __init__(self,x,y,player):
        self.x = x
        self.y = y
        self.player = player
    def __repr__(self):
        return "x:{0} y:{1} v:{2}".format(self.x,self.y,self.player)
    
players= {
            'black':1,
            'white':-1
            
        }

class SimulationState(object):
    def __init__(self,state,player):
        assert state.shape[0] == state.shape[1] and len(state.shape)== 2
        """
        state " np array "

        """
        
        self.board = state
        self.board_sz = state.shape[0]
        
        
        
        self.current_player = player
    
    def return_winner(self):
        return check_sequence(self.board)


    def is_game_over(self):
        return self.return_winner() is not None
    
    def is_move_legal(self,move):
        if self.current_player != move.player:
            return False
        if not (0<= move.x < self.board_sz):
            return False
        
        if not (0<= move.y < self.board_sz):
            return False
        
        
        
        
        return self.board[move.x,move.y] == 0
    
    def move(self,move):
        if not self.is_move_legal(move):
            raise ValueError("move {0} on board {1} is not legal".format(move,self.board))
            
        new_board = np.copy(self.board)
        new_board[move.x,move.y] = players[move.player]
        
        if self.current_player == "black":
            next_player = "white"
        else:
            next_player = "black"
            
        return SimulationState(new_board,next_player)
    def get_legal_actions(self):
        index = np.where(self.board == 0)
        return [Move(coords[0],coords[1],self.current_player) for coords in list(zip(index[0],index[1]))]
        
        
        