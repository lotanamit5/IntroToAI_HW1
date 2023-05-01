import numpy as np
from abc import abstractmethod
from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import heapdict

class Node():
    def __init__(self) -> None:
        self.state = -1
        self.actions = None
        
    @staticmethod
    def contains_state(node_list, node):
        for curr_node in node_list:
            if curr_node.state == node.state:
                return True
        return False
    
    @staticmethod   
    def make_node(state, actions):
        n = Node()
        n.state = state
        n.actions = actions
        return n

class Agent():
    def __init__(self) -> None:
        self.open = []
        self.close = {}
        self.actions = []
        self.cost = 0
        self.expended = 0
        self.env = None
    
    def reverse_action(self, action):
        return (action + 2) % 4
    

    @abstractmethod
    def next(self):
        pass
    
    @abstractmethod
    def expand(self, current):
        pass
    
    def solution(self, final_node):
        actions = final_node.actions
        total_cost = 0
        
        for action in actions:
            state, cost, _ = self.env.step(action)
            self.env.set_state(state)
            total_cost += cost
            
        return actions, total_cost, len(self.close)
    
    @abstractmethod
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        pass
    
    
class BFSAgent():
    def __init__(self) -> None:
        super().__init__()

    def next(self):
        return self.open.popleft()
    
    def expand(self, current):
        for action, (state, cost, terminated) in self.env.succ(current.value):
            if not terminated:
                yield action, (state, cost)
    
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        
        node = Node.make_node(env.get_initial_state(), None)
        if env.is_final_state(node.state):
            return self.solution(node)
        
        self.open.append(node)
        self.close = {}
        
        while not self.open.isEmpty():
            node = self.next() # TODO: maybe refactor
            self.close.add(node.state)
            for a, (s, _) in self.expand(node.state):
                child = Node.make_node(s, node.actions + a)
                if child.state not in self.close and not Node.contains_state(self.open, child):
                    if env.is_final_state(s):
                        return self.solution(child)
                    self.open.append(child)
                    
        return None

class DFSAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError
        


class UCSAgent():
  
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError



class GreedyAgent():
  
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError

class WeightedAStarAgent():
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: FrozenLakeEnv, h_weight) -> Tuple[List[int], int, float]:
        raise NotImplementedError   


class IDAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        raise NotImplementedError