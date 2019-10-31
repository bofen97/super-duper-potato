import numpy as np


def show(board):
    dicts={-1.:'@',1.:'#',0.:' '}


    def show_board(board,i):
        print(" {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(i,
            dicts[board[i,0]],dicts[board[i,1]],dicts[board[i,2]],dicts[board[i,3]],dicts[board[i,4]],\
             dicts[board[i,5]], dicts[board[i,6]], dicts[board[i,7]], dicts[board[i,8]], dicts[board[i,9]],\
                  dicts[board[i,10]], dicts[board[i,11]], dicts[board[i,12]], dicts[board[i,13]], dicts[board[i,14]]
        ) )
        print("----------------------------------------------------------------")
    print("     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  ")
    for i in range(board.shape[0]):
        show_board(board,i)



def check_single_col(col):
    check_sum = 0.
    last_sum = 0.
    for i in range(len(col)):
        check_sum+= col[i]
        if check_sum <=last_sum:
            check_sum = 0.
        last_sum = check_sum
        if check_sum >=5:
            return "black"
    
    check_sum = 0.
    last_sum = 0.
    for i in range(len(col)):
        check_sum += col[i]
        if check_sum>= last_sum:
            check_sum = 0.
        last_sum = check_sum
        if check_sum <=-5:
            return "white"
    
    return None
    
    
    
def get_sequences1(board):
    seqs = []
    for index in range(board.shape[0]):
        index_cp = np.copy(index)
        seq = []
        for index_ in range(index+1):
            seq.append(board[index_cp,index_])
            index_cp -= 1
        
        seqs.append(seq)
    return seqs
def get_sequences2(board):
    seqs = []
    for index in reversed(range(board.shape[0])):
        seq = []
        index_cp = np.copy(index)
        
        for index_ in range(board.shape[0]-index):
            seq.append(board[index_cp,index_])
            index_cp += 1
        seqs.append(seq)
    return seqs
            
            
            
def check_sequence(board):
    seq1 = get_sequences1(board)
    seq2 = get_sequences1(np.mat(board).T)
    seq3 = [board[:,i] for i in range(len(board[0])) ]
    seq4 = [board[i,:] for i in range(len(board[0])) ]
    seq5 = get_sequences2(board)
    seq6 = get_sequences2(np.mat(board).T)
    
    for seq in [seq1,seq2,seq3,seq4,seq5,seq6]:
        for col in seq:
            result = check_single_col(col)
            if result is not None:
                return result
            
    if np.all(board != 0):
        return "tied"
    
        
    
    
    return None