from TreeNode import TreeNode
class MCTS_Search(object):
    def __init__(self,node):
        assert isinstance(node,TreeNode)
        self.root = node
    
    def get_action(self):
        self.Search()
        _,act = self.root.children_best(0.)
        return act
            

    def Search(self):
        search_steps = 1000
        while (search_steps >0):
            current_node = self.root
            while  not current_node.is_terminal_node():
                if not current_node.is_full_expanded():
                    node =  current_node.expand_node()
                    node.rollout()
                    search_steps -= 1

                else:
                    current_node ,_= current_node.children_best(0.75)

