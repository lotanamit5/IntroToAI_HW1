import numpy as np
from collections import deque
from abc import abstractmethod
from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import heapdict

VERBOSE = False

class Node:
    def __init__(self, state, parent=None,action=0, cost=0, terminated=False):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.terminated = terminated
        
    def __repr__(self) -> str:
        return f"{self.state}"
    
class Agent:
    def __init__(self):
        self.env: FrozenLakeEnv = None
        self.open = None
        self.close: set = None
        self.expanded: int= 0
        
    def expand(self, node: Node) -> List[Node]:
        if VERBOSE: print(f"expanding node {node.state}")
        
        self.expanded += 1
        
        for action, (state, cost, terminated) in self.env.succ(node.state).items():
            if state != None:
                child = Node(state, parent=node, action=action, cost=cost, terminated=terminated)
                yield child
            
    
    def solution(self, node: Node) -> Tuple[List[int], int, float]:
        total_cost = 0
        actions = []
        
        # reverse the actions to get the path and accumulate the cost
        while node.parent != None:
            total_cost += node.cost
            actions.append(node.action)
            node = node.parent
            
        return reversed(actions), total_cost, self.expanded
    
    def init_search(self, env: FrozenLakeEnv):
        self.env = env
        self.env.reset()
        self.expanded = 0
        self.close = set()
    
    @abstractmethod
    def insert_to_open(self, node):
        pass

    @abstractmethod
    def states_in_open(self):
        pass

    @abstractmethod
    def get_next(self):
        pass

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.init_search(env)
        
        node: Node = Node(env.get_initial_state())
        
        if self.env.is_final_state(node.state):
            return self.solution(node)
        self.insert_to_open(node)
        
        while len(self.open) > 0:
            node = self.get_next()
            self.close.add(node.state)
            if VERBOSE: print(f"close: {[s for s in self.close]}")
            
            for child in self.expand(node):
                if child.state not in self.close and child.state not in self.states_in_open():
                    if self.env.is_final_state(child.state):
                        return self.solution(child)
                    self.insert_to_open(child)
                    if VERBOSE: print(f"open: {[s for s in self.open]}")
    
class BFSAgent(Agent):
    def __init__(self):
        super().__init__()
        self.open: deque = deque()
        
    def insert_to_open(self, node):
        self.open.append(node)
    
    def get_next(self):
        return self.open.popleft()

    def states_in_open(self):
        return [n.state for n in self.open]
             
class DFSAgent(Agent):
    def __init__(self):
        super().__init__()
        self.open: deque = deque()
        
    def insert_to_open(self, node):
        self.open.append(node)
    
    def get_next(self):
        return self.open.pop()

    def states_in_open(self):
        return [n.state for n in self.open]
    
    #The method is not required for valid DFS, 
    # but tests expect specific children ordering
    def expand(self, node):
        res = [n for n in super().expand(node)]
        res.reverse()
        return res
    

class UCSAgent(Agent):
    def __init__(self):
        super().__init__()
        self.open: heapdict = heapdict.heapdict()
        
    def insert_to_open(self, node):
        self.open[node] = (node.cost, node.state)
    
    def get_next(self):
        return self.open.popitem()[0]

    def states_in_open(self):
        return [n[1] for n in self.open.values()]
    
class InformedAgent(Agent):
    def __init__(self):
        super().__init__()
        
    def h(self, state):
        r_current, c_current = self.env.to_row_col(state)
        goal_states = self.env.get_goal_states()
        goal_coords = [self.env.to_row_col(state) for state in goal_states]
        manhatans = [
            abs(r_current - r_goal) + abs(c_current - c_goal)
            for r_goal, c_goal in goal_coords
        ]
        return min(manhatans + [100])
    
#TODO: no insert to close?
class GreedyAgent(InformedAgent):
    def __init__(self):
        super().__init__()
        self.open: heapdict = heapdict.heapdict()

    def insert_to_open(self, node):
        self.open[node] = (self.h(node.state), node.state)
    
    def get_next(self):
        return self.open.popitem()[0]

    def states_in_open(self):
        return [n[1] for n in self.open.values()]
    
    
class WeightedAStarAgent(InformedAgent):
    def __init__(self):
        super().__init__()
        self.h_weight = 0.5
        self.open: heapdict = heapdict.heapdict()
    
    def f(self, node: Node) -> float:
        return (1-self.h_weight)*node.cost + self.h_weight*self.h(node.state)

    def insert_to_open(self, node):
        self.open[node] = (self.f(node), node.state)
    
    def get_next(self):
        return self.open.popitem()[0]

    def states_in_open(self):
        return [n[1] for n in self.open.values()]
    
    def search(self, env: FrozenLakeEnv, h_weight=0.5) -> Tuple[List[int], int, float]:
        self.h_weight = h_weight
        return super().search(env)