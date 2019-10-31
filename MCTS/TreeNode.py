from collections import defaultdict
import numpy as np
from Simulation import Move,SimulationState

class TreeNode(object):
    def __init__ (self,simul_state,parent = None):
        assert isinstance(simul_state,SimulationState),"Just SimulationState."
        if parent is not None:
            assert isinstance(parent,TreeNode)
        
        self.parent = parent
        self.simulation_state = simul_state
        self._not_visited_actions = None

        
        self.children_score_stat = defaultdict(dict)
        self.current_score_stat = defaultdict(int)
        self.current_score_stat['n'] = 0.
        self.current_score_stat['black'] = 0.
        self.current_score_stat['white'] = 0.
        self.current_score_stat['tied'] = 0.
        

        
    def children_best(self,c_param):
        nodes__ = []
        values__ =[]

        for node,stat in self.children_score_stat.items():
            win_times = stat["black"]
            loses_times = stat['n'] - win_times  - stat['tied']
            q = (win_times - loses_times) / stat['n']

            q += c_param * np.sqrt((2.*np.log(self.current_score_stat['n'])/stat['n']))
            nodes__.append(node)
            values__.append(q)
        maxinodes = nodes__[np.argmax(values__)]
        maxiaction = self.children_score_stat[maxinodes]['action']
        return  maxinodes,maxiaction

    @property
    def not_visited_actions(self):
        
        if self._not_visited_actions is None:
            self._not_visited_actions = self.simulation_state.get_legal_actions()
        return self._not_visited_actions
    def expand_node(self):
        action = self.not_visited_actions.pop()
        child_state = self.simulation_state.move(action)
        child_node = TreeNode(simul_state=child_state,parent=self)        
        self.children_score_stat[child_node]['black'] = 0.
        self.children_score_stat[child_node]['tied'] = 0.
        self.children_score_stat[child_node]['white'] = 0.
        self.children_score_stat[child_node]['n'] = 0.
        self.children_score_stat[child_node]['action'] = action
        return child_node
    def is_terminal_node(self):
        return self.simulation_state.is_game_over()
    
    def rollout(self):
        current_state = self.simulation_state
        while not current_state.is_game_over():
            possible_moves = current_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_state = current_state.move(action)
        simulation_result = current_state.return_winner()
        self.BackUp(simulation_result)
        
        

    def BackUp(self,simulation_result):

        self.current_score_stat[simulation_result] += 1.
        self.current_score_stat['n'] += 1.
        if self.parent is not None:
            self.parent.BackUp(simulation_result)
            self.parent.children_score_stat[self]['n'] = self.current_score_stat['n']
            self.parent.children_score_stat[self][simulation_result] = self.current_score_stat[simulation_result]




        
            
    def rollout_policy(self,possible_moves):
        random_index = np.random.randint(low=0,high = len(possible_moves))
        return possible_moves[random_index]
    
            
            
    def is_full_expanded(self):
        return len(self.not_visited_actions) == 0