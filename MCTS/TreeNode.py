import numpy as np
from collections import defaultdict
from Simulation import Board
class Node(object):
    def __init__(self,Board_State,parent):
        assert isinstance(Board_State,Board)
        if parent is not None:
            assert isinstance(parent,Node)
        self._not_visited_actions = None
        self.Board_State = Board_State
        self.parent = parent
        
        
        
        self.children = defaultdict(dict)
        self.current_stat = defaultdict(int)
        self.current_stat["N"] = 0
        self.current_stat["BlackWin"] = 0
    
    def best_children(self,c):
        assert c>=0
        
        def calc_(c):
            states = []
            values = []
            for child,value in self.children.items():
                states.append(child)
                values.append(
                value["Q"] + c * np.sqrt(np.log(self.current_stat["N"]/value["N"] ))
                )
            return states,values
        
        
        if self.Board_State.BlackToPlay():
            states,values = calc_(c)
            max_node = states[np.argmax(values)]
            max_action = self.children[max_node]["action"]
            return max_node,max_action
        else:
            states,values = calc_(-1.0* c)
            max_node = states[np.argmax(values)]
            max_action = self.children[max_node]["action"]
            return max_node,max_action
            
            
            
        
        
        
    
    @property
    def not_visited_actions(self):
        if self._not_visited_actions is None:
            self._not_visited_actions = self.Board_State.Legal()
        return self._not_visited_actions
    def expand_node(self):
        action = self.not_visited_actions.pop()
        child_state = self.Board_State.Play(action)
        child_node = Node(child_state,self)
        self.children[child_node]["N"] = 0
        self.children[child_node]["BlackWin"] = 0
        self.children[child_node]["Q"] = 0
        self.children[child_node]["action"] = action
        return child_node
    def BackUp(self,BlackWin):
        if BlackWin:
            self.current_stat["BlackWin"] += 1
        self.current_stat["N"] += 1
        
        self.current_stat["Q"] = self.current_stat["BlackWin"]/self.current_stat["N"]
        
        if self.parent:
            self.parent.BackUp(BlackWin)
            self.parent.children[self]["N"] =self.current_stat["N"]
            self.parent.children[self]["BlackWin"] = self.current_stat["BlackWin"]
            self.parent.children[self]["Q"] = self.current_stat["Q"]
    
    
    def is_terminal_node(self):
        return self.Board_State.GameOver()
    def is_full_expanded(self):
        return len(self.not_visited_actions) == 0
    
    
    def rollout_policy(self,possible_moves):
        
        random_index = np.random.randint(low=0,high = len(possible_moves))
        return possible_moves[random_index]
        
    def rollout(self):
        current_state = self.Board_State
        while not current_state.GameOver():
            all_actions = current_state.Legal()
            action = self.rollout_policy(all_actions)
            current_state = current_state.Play(action)
        if current_state.BlackWins():
            self.BackUp(True)
        else:
            self.BackUp(False)
    