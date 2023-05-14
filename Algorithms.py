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
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        pass
    
class BFSAgent(Agent):
    def __init__(self):
        super().__init__()
        
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.init_search(env)
        
        self.open: deque = deque()
        node: Node = Node(env.get_initial_state())
        
        if self.env.is_final_state(node.state):
            return self.solution(node)
        self.open.append(node)
        
        while len(self.open) > 0:
            node = self.open.popleft()
            self.close.add(node.state)
            if VERBOSE: print(f"close: {[s for s in self.close]}")
            
            for child in self.expand(node):
                if child.state not in self.close and child.state not in [n.state for n in self.open]:
                    if self.env.is_final_state(child.state):
                        return self.solution(child)
                    self.open.append(child)
                    if VERBOSE: print(f"open: {[s for s in self.open]}")
             
class DFSAgent(Agent):
    def __init__(self):
        super().__init__()
    
    def recursive_dfs(self) -> Tuple[List[int], int, float]:
        node = self.open.pop()
        self.close.add(node.state)
        
        if(self.env.is_final_state(node.state)):
            return self.solution(node)
        
        for child in self.expand(node):
            if child.state not in self.close and child.state not in [n.state for n in self.open]:
                self.open.append(child)
                
                result = self.recursive_dfs()
                if(result != None):
                    return result
                
        return None
     
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.init_search(env)
        
        self.open: deque = deque()
        node: Node = Node(env.get_initial_state())
        self.open.append(node)
        
        return self.recursive_dfs()

class UCSAgent(Agent):
    def __init__(self):
        super().__init__()
    
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.init_search(env)
        self.open: heapdict = heapdict.heapdict()
        
        node: Node = Node(env.get_initial_state())
        
        if self.env.is_final_state(node.state):
            return self.solution(node)
        self.open[node] = (node.cost, node.state)
        
        while len(self.open) > 0:
            node, (cost, _) = self.open.popitem()
            self.close.add(node.state)
            if VERBOSE: print(f"close: {[s for s in self.close]}")
            
            for child in self.expand(node):
                if child.state not in self.close and child.state not in [n[1] for n in self.open.values()]:
                    if self.env.is_final_state(child.state):
                        return self.solution(child)
                    self.open[child] = (cost + child.cost, child.state)
                    if VERBOSE: print(f"open: {[s for s in self.open]}")
    
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
    
class GreedyAgent(InformedAgent):
    def __init__(self):
        super().__init__()
    
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.init_search(env)
        self.open: heapdict = heapdict.heapdict()

        node = Node(self.env.get_initial_state())
        self.open[node] = (self.h(node.state), node.state)

        while len(self.open) > 0:
            node, _ = self.open.popitem()
            if env.is_final_state(node.state):
                return self.solution(node)

            for child in self.expand(node):
                if child.state not in self.close and child.state not in [n[1] for n in self.open.values()]:
                    self.open[child] = (self.h(child.state), child.state)

        return None
    
class WeightedAStarAgent(InformedAgent):
    def __init__(self):
        super().__init__()
    
    def f(self, node: Node) -> float:
        return (1-self.h_weight)*node.g + self.h_weight*self.h(node.state)
    
    def update_node(self, state: int, parent: Node, g: float, f: float) -> Node:
        pass
    
    def search(self, env: FrozenLakeEnv, h_weight=0.5) -> Tuple[List[int], int, float]:
        self.init_search(env)
        self.h_weight = h_weight
        self.open: heapdict = heapdict.heapdict()
        
        node = Node(self.env.get_initial_state())
        self.open[node] = (self.f(node), node.state)
        
        while len(self.open) > 0:
            node, _ = self.open.popitem()
            
            self.close.add(node.state)
            
            if env.is_final_state(node.state):
                return self.solution(node)
            
            for child in self.expand(node):
                new_g = node.g + child.cost
                open = [n[1] for n in self.open.values()]
                
                if child.state not in self.close and child.state not in open:
                    self.open[child] = (self.f(child), child.state)
                    
                elif child.state in open:
                    n_curr = next(filter(lambda n, _: n.state == child.state, self.open.keys()), None)[0]
                    if new_g < n_curr.g:
                        n_curr = self.update_node(child.state, node, new_g, new_g + self.h(child.state))
                        self.open.update_key(n_curr)
                        
                else:  # child.state in close
                    n_curr = next(filter(lambda n, _: n.state == child.state, self.close))
                    if new_g < n_curr.g:
                        n_curr = self.update_node(child.state, node, new_g, new_g + self.h(child.state))
                        self.open.update_key(n_curr)
                        open[n_curr] = (self.f(n_curr), n_curr.state)
                        self.close.remove(n_curr)