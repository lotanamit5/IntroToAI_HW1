import numpy as np
from collections import deque
from abc import abstractmethod
#from Algo_lior import Node
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
        self.g = parent.g + cost if parent is not None else cost
        
    def __repr__(self) -> str:
        return f"{self.state}"
    
class Agent:
    def __init__(self):
        self.env: FrozenLakeEnv = None
        self.open = None
        self.close = set()
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
            
        return list(reversed(actions)), total_cost, self.expanded
    
    def init_search(self, env: FrozenLakeEnv):
        self.env = env
        self.env.reset()
        self.expanded = 0
        self.close.clear()
    
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
        self.open[node] = (node.g, node.state)
    
    def get_next(self):
        return self.open.popitem()[0]

    def states_in_open(self):
        return [n[1] for n in self.open.values()]
    
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
    
    def f(self, node: InformedNode) -> float:
        w = self.h_weight
        if w == 1: # avoid g==inf problem
            return node.h
        f = (1 - w) * node.g + w * node.h
        return f
    
    def update_node(self, node: InformedNode, parent: InformedNode, g: float) -> InformedNode:
        return InformedNode(node.state, parent=parent, action=node.action, cost=node.cost,
                            terminated=node.terminated, h=self.h(node.state), f=self.f(node))
    
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
                        self.open[n_curr] = (n_curr.f, n_curr.state)


class IDAStarAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.env = None

    def h(self, state):
        r_current, c_current = self.env.to_row_col(state)
        goal_states = self.env.get_goal_states()
        goal_coords = [self.env.to_row_col(state) for state in goal_states]
        manhatans = [
            abs(r_current - r_goal) + abs(c_current - c_goal)
            for r_goal, c_goal in goal_coords
        ]
        return min(manhatans + [100])

    def dfs_f(self, node, bound):
        f = node.cost + self.h(node.state)

        if f > bound:
            # print(f"Beyond bound {bound} for {node.state}")
            return None, f

        if self.env.is_final_state(node.state):
            # print(f"Found sol for node {node.state}")
            return node, f

        new_limit = math.inf

        # print(f"Expanding node {node.state}, with cost {node.cost}")
        for a, (s, c) in self.expand(node):
            child = Node.make_node(s, node.actions + [a], node.cost + c)
            final_node, child_bound = self.dfs_f(child, bound)

            if final_node is not None:
                return final_node, child_bound

            new_limit = min(new_limit, child_bound)

        return None, new_limit

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        init_state = self.env.get_initial_state()
        bound = self.h(init_state)
        init_node = Node.make_node(init_state, [], 0)

        while 1:
            final_node, bound = self.dfs_f(init_node, bound)
            # print(f"New bound is {bound}")
            if final_node is not None:
                return self.solution(final_node)

            if bound == math.inf:
                return None