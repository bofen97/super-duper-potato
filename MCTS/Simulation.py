import numpy as np
from utils import WhiteWins,BlackWins,GameOver
class Action(object):
    def __init__(self,x,y,player):
        self.player = player
        self.x = x
        self.y = y
    def __repr__(self):
        return "position_x {}  position_y {}  player {} ".format(self.x,self.y,self.player)


class Board(object):
    def __init__(self,state_np,player):
        self.__board = state_np
        self.board_sz = state_np.shape[0]
        self.__player = player
        
        
        self.player_value = {
            "black":1,
            "white":-1
        }
    def SetPosition(self):
        pass
        
        
        

    def GetPosition(self):
        assert self.__board is not None
        return self.__board
    
    def GameOver(self):
        return GameOver(self.__board)
    def __is_legal(self,a):
        assert isinstance(a,Action)
        assert self.__player is not None
        assert self.board_sz is not None
        assert self.__player == a.player
        assert 0<= a.x < self.board_sz
        assert 0<= a.y < self.board_sz
        assert self.__board[a.x,a.y] == 0
        return True
        
    
    def Play(self,a):
        assert self.__is_legal(a)
        assert self.__board is not None
        board_cp = np.copy(self.__board)
        
        board_cp[a.x,a.y] = self.player_value[a.player]
        if self.__player == "black":
            player = "white"
        else:
            player = "black"
        
        return Board(board_cp,player)
            
        
        
    def BlackWins(self):
        return BlackWins(self.__board)
    def __WhiteWins(self):
        return WhiteWins(self.__board)
        
        
    def Legal(self):
        indexs = np.where(self.__board == 0)
        return [Action(coords[0],coords[1],self.__player) for coords in list(zip(indexs[0],indexs[1]))]
        
        
        
        
        
    def BlackToPlay(self):
        if self.__player == "black":
            return True
        else:
            return False
        
    