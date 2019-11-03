from utils import show,BlackWins,WhiteWins,GameOver
from Simulation import Action,Board
import numpy as np
from TreeNode import Node
from MCTS_Search import MCTS_Search

def play():
    board = np.zeros((3,3))
    while True:
        
        state = Board(board,"black")
        node = Node(state,parent=None)
        mcts = MCTS_Search(node)
        acts = mcts.Search()
        board[acts.x,acts.y] =1
        show(board)
        if GameOver(board):
            if BlackWins(board):
                return "Black Win !!!"
            elif WhiteWins(board):
                return " White Win !!!"
            else:
                return "Tide !!!"
        print("=====================")
        


        
        x= int(input("enter  x  :"))
        y =int(input("enter y  :"))
        board[x,y] =-1
        show(board)
        if GameOver(board):
            if BlackWins(board):
                return "Black Win !!!"
            elif WhiteWins(board):
                return " White Win !!!"
            else:
                return "Tide !!!"

if __name__ == "__main__":
    print(play())


