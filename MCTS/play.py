from utils import show,check,show
from Simulation import Move,SimulationState,players
import numpy as np
from TreeNode import TreeNode
from MCTS_Search import MCTS_Search

def play():
    result = None
    board = np.zeros((3,3))
    while True:

        state = SimulationState(board,"black")
        node = TreeNode(state)
        mcts = MCTS_Search(node)
        acts = mcts.get_action()
        board[acts.x,acts.y] =players["black"]
        show(board)
        print("=====================")
        result = check(board)
        if result is not None:
            if result =="black":
                print("==========Black Win !!! ==========")
                break
            if result =="white":
                print("==========White Win !!! ==========")
                break
            if result =="tied":
                print("==========Tied !!! ==========")
                break


        
        x= int(input("enter  x  :"))
        y =int(input("enter y  :"))
        board[x,y] = players["white"]
        result = check(board)
        show(board)
        if result is not None:

            if result =="black":
                print("==========Black Win !!! ==========")
                break
            if result =="white":
                print("==========White Win !!! ==========")
                break
            if result =="tied":
                print("==========Tied !!! ==========")
                break

if __name__ == "__main__":
    play()


