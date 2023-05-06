import numpy as np
from abc import abstractmethod
from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import heapdict

class Node():
    def __init__(self) -> None:
        self.state = -1
        self.actions = []
        self.cost = 0
        
    def __eq__(self, other):
        if(isinstance(other, Node)):
            return self.state == other.state
        return False
    
    def __lt__(self, other):
        if(isinstance(other, Node)):
            if(self.cost == other.cost):
                return self.state < other.state
            return self.cost < other.cost
        return False
    
    # @staticmethod
    # def contains_state(node_list, node):
    #     for curr_node in node_list:
    #         if curr_node.state == node.state:
    #             return True
    #     return False
    
    
    @staticmethod   
    def make_node(state, actions=[], cost=0):
        n = Node()
        n.state = state
        n.actions = actions
        n.cost = cost
        return n

class Agent():
    def __init__(self) -> None:
        self.close = set()
        self.cost = 0
        self.expanded = 0
        self.env = None
    
    def expand(self, current):
        self.expanded += 1
        for action, (state, cost, _) in self.env.succ(current.state).items():
            if(state != None):
                yield action, (state, cost)
    
    def solution(self, final_node):
        actions = final_node.actions
        total_cost = 0
        
        for action in actions:
            state, cost, _ = self.env.step(action)
            self.env.set_state(state)
            total_cost += cost
            
        return actions, total_cost, self.expanded
    
    @abstractmethod
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        pass
    
    
class BFSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.open = []
    
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        
        node = Node.make_node(env.get_initial_state())
        if env.is_final_state(node.state):
            return self.solution(node)
        
        self.open.append(node)
        self.close = set()
        
        while len(self.open) > 0:
            node = self.open.pop(0)
            self.close.add(node.state)
            for a, (s, c) in self.expand(node):
                child = Node.make_node(s, node.actions + [a],0)
                if child.state not in self.close and child not in self.open:
                    if env.is_final_state(s):
                        return self.solution(child)
                    
                    self.open.append(child)
        return None
    

class DFSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.open = []

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        
        node = Node.make_node(env.get_initial_state())
        if env.is_final_state(node.state):
            return self.solution(node)
        
        self.close = set()
        return self.search_aux(node)

    def search_aux(self, node):
        if self.env.is_final_state(node.state):
            return self.solution(node)
        self.close.add(node.state)
        for a, (s, _) in self.expand(node):
            child =  Node.make_node(s, node.actions + [a])
            if child.state not in self.close:
                res = self.search_aux(child)
                if res is not None:
                    return res
        return None

class UCSAgent(Agent):
  
    def __init__(self) -> None:
        super().__init__()
        self.open = heapdict.heapdict()

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        
        node = Node.make_node(env.get_initial_state())
        if env.is_final_state(node.state):
            return self.solution(node)
        
        self.open[(0,node.state)] = node
        
        while len(self.open) > 0:
            node = self.open.popitem()[1]
            if env.is_final_state(node.state):
                return self.solution(node)
            
            for a, (s, c) in self.expand(node):
                child = Node.make_node(s, node.actions + [a], node.cost + c)
                self.open[(child.cost, child.state)] = child
                    
        return None
        
    
    @abstractmethod
    def insert_new(self, child):
        self.open[(child.cost, child.state)] = child


class GreedyAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.open = heapdict.heapdict()
        
    def h(self, state):
        r_current, c_current = self.env.to_row_col(state)
        goal_states = self.env.get_goal_states()
        goal_coords = [self.env.to_row_col(state) for state in goal_states]
        manhatans = [abs(r_current - r_goal) + abs(c_current - c_goal) for r_goal, c_goal in goal_coords]
        return min(manhatans+[100])
    
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        
        init_state = self.env.get_initial_state()
        node = Node.make_node(init_state, [], 0)
        self.open[(self.h(init_state), node.state)] = node
        
        while len(self.open) > 0:
            node = self.open.popitem()[1]
            if env.is_final_state(node.state):
                return self.solution(node)
            
            for a, (s, c) in self.expand(node):
                if s not in self.close and (self.h(s),s) not in self.open.keys():
                    child = Node.make_node(s, node.actions + [a], self.h(s))
                    self.open[(self.h(child.state), child.state)] = child
                    
        return None
        

class WeightedAStarAgent(Agent):
    
    def __init__(self) -> None:
        super().__init__()
        self.open = heapdict.heapdict()
        self.h_weight = 0.5
        
    def h(self, state):
        r_current, c_current = self.env.to_row_col(state)
        goal_states = self.env.get_goal_states()
        goal_coords = [self.env.to_row_col(state) for state in goal_states]
        manhatans = [abs(r_current - r_goal) + abs(c_current - c_goal) for r_goal, c_goal in goal_coords]
        return min(manhatans+[100])
    
    def f(self, node):
        return self.h_weight*self.h(node.state) + (1-self.h_weight)*node.cost
    
    def search(self, env: FrozenLakeEnv, h_weight) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        self.h_weight = h_weight
        self.close = {}
        
        init_state = self.env.get_initial_state()
        node = Node.make_node(init_state, [], 0)
        self.open[(node.cost, node.state)] = node
        
        while len(self.open) > 0:
            node = self.open.popitem()[1]
            self.close[node.state] = node
            if env.is_final_state(node.state):
                return self.solution(node)
            
            for a, (s, c) in self.expand(node):
                new_g = node.cost + c
                ss = [n for n in self.open.values() if n.state == s]
                
                if s in self.close.keys():
                    n_curr = self.close[s]
                    if(new_g < n_curr.cost):
                        n_curr = Node.make_node(s, node.actions + [a], new_g)
                        self.open[(n_curr.cost,n_curr.state)] = n_curr.state
                        self.close.pop(n_curr.state)
                        
                elif len(ss) > 0:
                    n_curr = ss[0]
                    if(new_g < n_curr.cost):
                        n_curr = Node.make_node(s, node.actions + [a], new_g)
                        self.open[n_curr] = n_curr.state
                    
                else:
                    n_curr = Node.make_node(s, node.actions + [a], new_g)
                    self.open[n_curr] = n_curr.state
                    
        return None


class IDAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError