import numpy as np
from collections import deque
from abc import abstractmethod
from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import math
import heapdict

class Node:
    def __init__(self, state, parent=None,action=0, cost=0, terminated=False):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.terminated = terminated
        self.g = parent.g + cost if parent is not None else cost
        
    def __repr__(self) -> str:
        return f"{self.state}"

  
class Agent:
    def __init__(self):
        self.env: FrozenLakeEnv = None
        self.open = None
        self.close: set = None
        self.expanded: int= 0
        
    def expand(self, node: Node) -> List[Node]:
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
            
        return list(reversed(actions)), total_cost, self.expanded
    
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
            
            for child in self.expand(node):
                if child.state not in self.close and child.state not in [n.state for n in self.open]:
                    if self.env.is_final_state(child.state):
                        return self.solution(child)
                    self.open.append(child)

            
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
        self.open[node] = (node.g, node.state)
        
        while len(self.open) > 0:
            node, _ = self.open.popitem()
            self.close.add(node.state)
            
            for child in self.expand(node):
                OPEN = [n.state for n in self.open.keys()]
                if child.state not in self.close and child.state not in OPEN:
                    if self.env.is_final_state(child.state):
                        return self.solution(child)
                    self.open[child] = (child.g, child.state)


class InformedNode(Node):
    def __init__(self, state, parent=None, action=0, cost=0, terminated=False,
                 h=0, f=0):
        super().__init__(state, parent=parent, action=action, cost=cost, terminated=terminated)
        self.h = h
        self.f = f


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
    
    def f(self, node: InformedNode) -> float:
        return node.g + self.h(node.state)
    
    def expand(self, node: InformedNode) -> List[InformedNode]:
        self.expanded += 1
        
        for action, (state, cost, terminated) in self.env.succ(node.state).items():
            if state != None:
                child = InformedNode(state, parent=node, action=action, cost=cost, terminated=terminated, h=self.h(state))
                child.f = self.f(child)
                yield child
    
   
class GreedyAgent(InformedAgent):
    def __init__(self):
        super().__init__()
    
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.init_search(env)
        self.open: heapdict = heapdict.heapdict()

        node = InformedNode(self.env.get_initial_state(), h=self.h(self.env.get_initial_state()))
        self.open[node] = (node.h, node.state)

        while len(self.open) > 0:
            node, _ = self.open.popitem()
            self.close.add(node.state)
            if env.is_final_state(node.state):
                return self.solution(node)

            for child in self.expand(node):
                OPEN = [n[1] for n in self.open.values()]
                if child.state not in self.close and child.state not in OPEN:
                    self.open[child] = (child.h, child.state)

        return None


class WeightedAStarAgent(InformedAgent):
    def __init__(self):
        super().__init__()
    
    def f(self, node: InformedNode) -> float:
        w = self.h_weight
        if w == 1: # avoid g==inf problem
            return node.h
        f = (1 - w) * node.g + w * node.h
        return f
    
    def update_node(self, node: InformedNode, parent: InformedNode, g: float) -> InformedNode:
        return InformedNode(node.state, parent=parent, action=node.action, cost=node.cost,
                            terminated=node.terminated, g=g, h=self.h(node.state), f=self.f(node))
    
    def search(self, env: FrozenLakeEnv, h_weight=0.5) -> Tuple[List[int], int, float]:
        assert 0 <= h_weight <= 1, "h_weight must be between 0 and 1"
        self.h_weight = h_weight
        
        self.init_search(env)
        self.open: heapdict = heapdict.heapdict()
        
        init_state = self.env.get_initial_state()
        node = InformedNode(init_state, h=self.h(init_state))
        node.f = self.f(node)
        self.open[node] = (node.f, node.state)
        
        while len(self.open) > 0:
            node, _ = self.open.popitem()
            
            self.close.add(node)
            
            if env.is_final_state(node.state):
                return self.solution(node)
            
            for child in self.expand(node):
                new_g = node.g + child.cost
                OPEN = [n for n in self.open.keys() if n.state == child.state]
                CLOSE = [n for n in self.close if n.state == child.state]
                
                if len(CLOSE) == 0 and len(OPEN) == 0:
                    self.open[child] = (child.f, child.state)
                    
                elif len(OPEN) > 0:
                    n_curr = OPEN[0]
                    if new_g < n_curr.g:
                        n_curr = self.update_node(child, parent=node, g=new_g)
                        self.open[n_curr] = (n_curr.f, n_curr.state)
                        for n in OPEN:
                            del self.open[n]
                        
                else:  # child in close
                    n_curr = CLOSE[0]
                    if new_g < n_curr.g:
                        n_curr = self.update_node(child, parent=node, g=new_g)
                        OPEN[n_curr] = (n_curr.f, n_curr.state)
                        self.close.remove(n_curr)


class IDAStarAgent(InformedAgent):
    def __init__(self):
        super().__init__()
        self.FOUND = -1
        self.FAILURE = -2
        self.new_limit = math.inf
    
    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.init_search(env)
        init_state = self.env.get_initial_state()
        root = InformedNode(init_state, h=self.h(init_state))
        self.new_limit = root.h

        while True:
            f_limit = self.new_limit
            self.new_limit = math.inf
            result = self.dfs_f(root, [], f_limit)
            if result != None:
                return self.solution(result[-1])
        
        return None

    def dfs_f(self, node:InformedNode, path: list, f_limit: int):
        new_f = node.g + node.h
        
        if new_f > f_limit:
            self.new_limit = min(self.new_limit, new_f)
            return None
        
        if self.env.is_final_state(node.state):
            return path
        
        for child in self.expand(node):
            path_states = [n.state for n in path]
            if child.state not in path_states:
                result = self.dfs_f(child, path + [child], f_limit)
                if result != None:
                    return result
        
        return None
        