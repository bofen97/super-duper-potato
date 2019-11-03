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



def BlackWins(board):
        row_sum = np.sum(board,axis=0)
        col_sum = np.sum(board,axis=1)
        diag_sum_r = board.trace()
        diag_sum_l = board[::-1].trace()
        
        wins = any(row_sum == board.shape[0])
        wins += any(col_sum == board.shape[0])
        wins += (diag_sum_l == board.shape[0])
        wins += (diag_sum_r == board.shape[0])
        if wins:
            return True
        else:return None
def WhiteWins(board):
    row_sum = np.sum(board,axis=0)
    col_sum = np.sum(board,axis=1)
    diag_sum_r = board.trace()
    diag_sum_l = board[::-1].trace()
        
    wins = any(row_sum == -board.shape[0])
    wins += any(col_sum == -board.shape[0])
    wins += (diag_sum_l == -board.shape[0])
    wins += (diag_sum_r == -board.shape[0])
    if wins:
        return True
    else:return None
def GameOver(board):
    if BlackWins(board) is not None:
        return True
    elif WhiteWins(board) is not None:
        return True
    else:
        return np.all(board !=0 )
    
    