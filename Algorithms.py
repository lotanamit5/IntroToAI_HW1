import numpy as np
from abc import abstractmethod
from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import heapdict

class Node():
    def __init__(self) -> None:
        self.state = -1
        self.actions = []
        
    @staticmethod
    def contains_state(node_list, node):
        for curr_node in node_list:
            if curr_node.state == node.state:
                return True
        return False
    
    @staticmethod   
    def make_node(state, actions=[]):
        n = Node()
        n.state = state
        n.actions = actions
        return n

class Agent():
    def __init__(self) -> None:
        self.open = []
        self.close = set()
        self.actionsMapping = ['down','right','up','left']
        self.cost = 0
        self.expended = 0
        self.env = None
        self.expanded_counter = 0
    
    # def reverse_action(self, action):
    #     return (action + 2) % 4

    @abstractmethod
    def next(self):
        pass
    
    @abstractmethod
    def insert_new(self, child):
        pass

    def expand(self, current):    
        self.expanded_counter += 1
        for action, (state, cost, terminated) in self.env.succ(current.state).items():
            if(state != None):
                yield action, (state, cost)
    
    def solution(self, final_node):
        actions = final_node.actions
        total_cost = 0
        
        for action in actions:
            state, cost, _ = self.env.step(action)
            self.env.set_state(state)
            total_cost += cost
            
        return actions, total_cost, self.expanded_counter
    
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        
        node = Node.make_node(env.get_initial_state())
        if env.is_final_state(node.state):
            return self.solution(node)
        
        self.open.append(node)
        self.close = set()
        
        while len(self.open) > 0:
            node = self.next() # TODO: maybe refactor
            self.close.add(node.state)
            for a, (s, _) in self.expand(node):
                child = Node.make_node(s, node.actions + [a])
                if child.state not in self.close and not Node.contains_state(self.open, child):
                    if env.is_final_state(s):
                        # print(f"{self.close}")
                        return self.solution(child)
                    
                    self.insert_new(child)
        # print(f"{self.close}")        
        return None
    
    
class BFSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def next(self):
        return self.open.pop(0)
    
    def insert_new(self, child):
        self.open.append(child)
    


class DFSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.states_proccessed = 0

    def next(self):
        return self.open.pop(-1)
    
    def insert_new(self, child):
        self.open.append(child)

    def see_state(self, state):
        self.close.add(state)
        self.states_proccessed += 1

    def solution(self, node):
        res = super().solution(node)
        return res[0], res[1], self.states_proccessed

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        
        node = Node.make_node(env.get_initial_state())
        # if env.is_final_state(node.state):
            # return self.solution(node)
        # open = [node]
        self.close = set()
        return self.search_aux(node)

    def search_aux(self, node):
        if self.env.is_final_state(node.state):
            return self.solution(node)
        self.see_state(node.state)
        for a, (s, _) in self.expand(node):
            child =  Node.make_node(s, node.actions + [a])
            if child.state not in self.close:
                res = self.search_aux(child)
                if res is not None:
                    return res
        self.close.remove(node.state)
        return None

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