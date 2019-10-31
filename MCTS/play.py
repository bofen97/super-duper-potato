from utils import show,check_sequence,show
from Simulation import Move,SimulationState,players
import numpy as np
from TreeNode import TreeNode
from MCTS_Search import MCTS_Search

def play():
    result = None
    board = np.zeros((15,15))
    while True:

        state = SimulationState(board,"black")
        node = TreeNode(state)
        mcts = MCTS_Search(node)
        acts = mcts.get_action()
        board[acts.x,acts.y] =players["black"]
        show(board)
        print("=====================")
        result = check_sequence(board)
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


        
        x,y = input("enter your x and y  :")
        board[int(x),int(y)] = players["white"]
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


