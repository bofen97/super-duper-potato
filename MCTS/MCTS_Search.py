from TreeNode import Node

class MCTS_Search(object):
    def __init__(self,node):
        assert isinstance(node,Node)
        self.root = node
    
    

    def Search(self):
        current_node = self.root
        while  not current_node.is_terminal_node():

            if not current_node.is_full_expanded():
                node =  current_node.expand_node()
                for _ in range(100):
                    node.rollout()
                    
            else:
                current_node,_= current_node.best_children(1.4)
        return self.root.best_children(0.)[-1]
                    
