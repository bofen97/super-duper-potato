import numpy as np


def show(board):
    dicts={-1.:'@',1.:'#',0.:' '}


    def show_board(board,i):
        print(" {} | {} | {} | {} | {} | {} |".format(i,
            dicts[board[i,0]],dicts[board[i,1]],dicts[board[i,2]],dicts[board[i,3]],dicts[board[i,4]] ) )
        print("------------------------")
    print("     0   1   2   3   4")
    for i in range(board.shape[0]):
        show_board(board,i)



def BlackWins(board):
    def col_check(board_1):
        last_sum= 0
        sum_ = 0
        board_sz = len(board_1)
        for i in range(board_sz):
            last_sum = sum_
            
            sum_ += board_1[i]

            if sum_ <=last_sum:
                sum_ = 0
            if sum_ >=4:
                return True
        return None
    for i in range(board.shape[0]):
        if col_check( board[:,i]) is not None:
            return True
        if col_check(board[i,:]) is not None:
            return True
    r1 = np.copy( board[::-1])
    r2 =np.copy((np.mat(board)[::-1].T)[::-1])
    r3 =np.copy( (np.mat(board).T)[::-1])
    r4 = np.copy(board)
    
    rs = [r1,r2,r3,r4]
    for r in rs:    
        arrys = get_check_array(r)
        for arry in arrys:
            if col_check(arry) is not None:
                return True
    return None 

def get_check_array(board):
    arrys = []
    for c in range(board.shape[0]):
        arry=[]
        c_cp = np.copy(c)
        for index in range(c+1):
            arry.append(board[c_cp,index])
            c_cp -= 1 
        arrys.append(arry)
    return arrys


def WhiteWins(board):

    def col_check(board_1):
        
        last_sum= 0
        sum_ = 0
        board_sz = len(board_1)
        for i in range(board_sz):
            last_sum = sum_
            
            sum_ += board_1[i]

            if sum_ >=last_sum:
                sum_ = 0
            if sum_ <=-4:
                return True
        return None
            
    for i in range(board.shape[0]):
        if col_check( board[:,i]) is not None:
            return True
        if col_check(board[i,:]) is not None:
            return True
    r1 = np.copy( board[::-1])
    r2 =np.copy((np.mat(board)[::-1].T)[::-1])
    r3 =np.copy( (np.mat(board).T)[::-1])
    r4 = np.copy(board)
    rs = [r1,r2,r3,r4]
    for r in rs:    
        arrys = get_check_array(r)
        for arry in arrys:
            if col_check(arry) is not None:
                return True
    return None


            
            


    for i in range(board.shape[0]):
        if col_check( board[:,i]) is not None:
            return True
        if col_check(board[i,:]) is not None:
            return True
    return None 

def GameOver(board):
    if BlackWins(board) is not None:
        return True
    elif WhiteWins(board) is not None:
        return True
    else:
        return np.all(board !=0 )
    
    