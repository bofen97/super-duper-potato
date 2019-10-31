import numpy as np


def show(board):
    dicts={-1.:'@',1.:'#',0.:' '}


    def show_board(board,i):
        print(" {} | {} | {} | {} |".format(i,
            dicts[board[i,0]],dicts[board[i,1]],dicts[board[i,2]] ) )
        print("----------------")
    print("     0   1   2   ")
    for i in range(board.shape[0]):
        show_board(board,i)


def check(board):
        row_sum = np.sum(board,0)
        
        col_sum = np.sum(board,1)
        
        diag_sum_r = board.trace()
        
        diag_sum_l = board[::-1].trace()
        
        black_win = any(row_sum ==board.shape[0])
        black_win += any(col_sum == board.shape[0])
        black_win += (diag_sum_l == board.shape[0])
        black_win += (diag_sum_r == board.shape[0])
        
        if black_win:
            return "black"

        white_win = any(row_sum == -board.shape[0])
        white_win += any(col_sum == -board.shape[0])
        white_win += (diag_sum_l == - board.shape[0])
        white_win += (diag_sum_r == - board.shape[0])
        
        if white_win:
            return "white"
        
        if np.all(board !=0):
            return "tied"
        
        return None