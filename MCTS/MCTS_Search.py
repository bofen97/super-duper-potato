from TreeNode import TreeNode
class MCTS_Search(object):
    def __init__(self,node):
        assert isinstance(node,TreeNode)
        self.root = node
    
    def get_action(self):
        for _ in range(1):
            self.Search()
        _,act = self.root.children_best(0.)
        return act
            

    def Search(self):
        current_node = self.root

        while  not current_node.is_terminal_node():
            if not current_node.is_full_expanded():
                node =  current_node.expand_node()
                node.rollout()

            else:
                current_node ,_= current_node.children_best(0.75)

