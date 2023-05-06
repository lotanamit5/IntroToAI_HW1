import numpy as np
from collections import deque
from abc import abstractmethod
from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import heapdict

VERBOSE = True

class Node2:
    def __init__(self, state, action=None, parent=None, cost=0, depth=0,terminated=False):
        self.state = state
        self.action = action
        self.parent = parent
        self.cost = cost
        self.depth = depth
        self.is_terminated = terminated
    
    def __eq__(self, other):
        if isinstance(other, Node2):
            return self.state == other.state
        return False

    def __lt__(self, other):
        return self.cost < other.cost
       
class Agent2:
    def __init__(self):
        self.env = None
        self.open = None
        self.close = set()
        self.expanded = 0

    def pop(self) -> Node2:
        pass
    def push(self, node: Node2):
        pass
    def empty(self) -> bool:
        pass
    
    def expand(self, node:Node2):
        if VERBOSE:
            print(f"Expanding node {node.state}")
            
        if node.is_terminated:
            return
        
        self.expanded += 1
        
        for action, (state, cost, terminated) in self.env.succ(node.state).items():
            if state is not None and state not in self.close:
                child = Node2(state, action, node, node.cost + cost, node.depth + 1, terminated)
                yield child

    def solution(self, node:Node2):
        actions = []
        total_cost = 0
        
        while node.parent:
            actions.append(node.action)
            node = node.parent
            total_cost += node.cost

        return list(reversed(actions)), total_cost, self.expanded
    
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        node = Node2(env.get_initial_state())
        
        if env.is_final_state(node.state):
            return self.solution(node)
        self.push(node)
                    
        while not self.empty():
            node = self.pop()          
            self.close.add(node.state)
            
            for child in self.expand(node):
                if (child.state not in self.close) and (child not in self.open):
                    if env.is_final_state(child.state):
                        return self.solution(child)
                    self.push(child)
                
        return None


class BFSAgent2(Agent2):
    def __init__(self):
        super().__init__()
        self.open = deque()
        
    def pop(self):
        return self.open.pop()
    
    def push(self, node):
        self.open.append(node)
        
    def empty(self):
        return len(self.open) == 0

#######################################################################
# class Node():
#     def __init__(self) -> None:
#         self.state = -1
#         self.actions = []
        
#     @staticmethod
#     def contains_state(node_list, node):
#         for curr_node in node_list:
#             if curr_node.state == node.state:
#                 return True
#         return False

#     @staticmethod   
#     def make_node(state, actions=[]):
#         n = Node()
#         n.state = state
#         n.actions = actions
#         return n
    
# class Agent():
#     def __init__(self) -> None:
#         self.open = []
#         self.close = set()
#         self.actionsMapping = ['down','right','up','left']
#         self.cost = 0
#         self.expended = 0
#         self.env = None
#         self.expanded_counter = 0
    
#     # def reverse_action(self, action):
#     #     return (action + 2) % 4

#     @abstractmethod
#     def next(self):
#         pass
    
#     @abstractmethod
#     def insert_new(self, child):
#         pass

#     def expand(self, current):    
#         self.expanded_counter += 1
#         for action, (state, cost, terminated) in self.env.succ(current.state).items():
#             if(state != None):
#                 yield action, (state, cost)
    
#     def solution(self, final_node):
#         actions = final_node.actions
#         total_cost = 0
        
#         for action in actions:
#             state, cost, _ = self.env.step(action)
#             self.env.set_state(state)
#             total_cost += cost
            
#         return actions, total_cost, self.expanded_counter
    
#     def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
#         self.env = env
#         self.env.reset()
        
#         node = Node.make_node(env.get_initial_state())
#         if env.is_final_state(node.state):
#             return self.solution(node)
        
#         self.open.append(node)
#         self.close = set()
        
#         while len(self.open) > 0:
#             node = self.next() # TODO: maybe refactor
#             self.close.add(node.state)
#             for a, (s, _) in self.expand(node):
#                 child = Node.make_node(s, node.actions + [a])
#                 if child.state not in self.close and not Node.contains_state(self.open, child):
#                     if env.is_final_state(s):
#                         # print(f"{self.close}")
#                         return self.solution(child)
                    
#                     self.insert_new(child)
#         # print(f"{self.close}")        
#         return None
    
    
# class BFSAgent(Agent):
#     def __init__(self) -> None:
#         super().__init__()

#     def next(self):
#         return self.open.pop(0)
    
#     def insert_new(self, child):
#         self.open.append(child)
    


# class DFSAgent(Agent):
#     def __init__(self) -> None:
#         super().__init__()
#         self.states_proccessed = 0

#     def next(self):
#         return self.open.pop(-1)
    
#     def insert_new(self, child):
#         self.open.append(child)

#     def see_state(self, state):
#         self.close.add(state)
#         self.states_proccessed += 1

#     def solution(self, node):
#         res = super().solution(node)
#         return res[0], res[1], self.states_proccessed

#     def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
#         self.env = env
#         self.env.reset()
        
#         node = Node.make_node(env.get_initial_state())
#         # if env.is_final_state(node.state):
#             # return self.solution(node)
#         # open = [node]
#         self.close = set()
#         return self.search_aux(node)

#     def search_aux(self, node):
#         if self.env.is_final_state(node.state):
#             return self.solution(node)
#         self.see_state(node.state)
#         for a, (s, _) in self.expand(node):
#             child =  Node.make_node(s, node.actions + [a])
#             if child.state not in self.close:
#                 res = self.search_aux(child)
#                 if res is not None:
#                     return res
#         self.close.remove(node.state)
#         return None

# class UCSAgent():
  
#     def __init__(self) -> None:
#         raise NotImplementedError

#     def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
#         raise NotImplementedError



# class GreedyAgent():
  
#     def __init__(self) -> None:
#         raise NotImplementedError

#     def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
#         raise NotImplementedError

# class WeightedAStarAgent():
    
#     def __init__(self):
#         raise NotImplementedError

#     def search(self, env: FrozenLakeEnv, h_weight) -> Tuple[List[int], int, float]:
#         raise NotImplementedError   


# class IDAStarAgent():
#     def __init__(self) -> None:
#         raise NotImplementedError
        
#     def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
#         raise NotImplementedError