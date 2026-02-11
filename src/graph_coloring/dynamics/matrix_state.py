import logging

from copy import copy
from typing import Any, List
from itertools import product

import networkx as nx
from scipy.sparse import csr_matrix, diags, lil_matrix

from mdp.stochopti.discrete_world import space
from gym import Space, Env
from gym.spaces import Discrete
import numpy as np

from graph_coloring.visualizer import Visualizer


class MatrixState:
    @property
    def shape(self):
        return self.matrix.shape
    
    def __init__(self, graph: nx.Graph = None, matrix: np.ndarray = None):
        if matrix is None:
            if graph is None:
                raise AttributeError('Either the graph or the matrix need to be provided.')
            matrix = nx.linalg.laplacian_matrix(graph)
        
        self.matrix = matrix
        # Stores the rows in which there are colors
        self.color_idxs = [0]
        # Stores for each node which row represents it.
        self.nodes_to_row = {i: i for i in range(matrix.shape[0])}
    
    def __len__(self):
        return len(self.color_idxs)
    
    def color_node(self, node_to_color, color_idx, strict=True):
        node_i = self.nodes_to_row[node_to_color]
        node_j = node_i if color_idx >= len(self.color_idxs) else self.color_idxs[color_idx]
        if node_i in self.color_idxs:
            logging.warning(f'Node has already been colored...')
            return 0
        
        if strict:
            if self.matrix[node_i, node_j] < 0:
                logging.warning(f'Infeasible to color node {node_to_color} with color {color_idx}.\n'
                                f'There exists adjacent vertexes among nodes that have already been colored with '
                                f'that color.')
                return 0
        
        if color_idx >= len(self.color_idxs):
            node_j = self.nodes_to_row[node_to_color]
            for node_i in self.color_idxs:
                if self.matrix[node_i, node_j] == 0:
                    self.matrix[node_i, node_j] = -1
                    self.matrix[node_j, node_i] = -1
                    self.matrix[node_j, node_j] += 1
                    self.matrix[node_i, node_i] += 1
            
            self.color_idxs.append(node_j)
        else:
            if node_i < node_j:
                # In this case we "move" all colored nodes to the small idx
                self.color_idxs[color_idx] = node_i
                for k, v in self.nodes_to_row.items():
                    if v == node_j:
                        self.nodes_to_row[k] = node_i
                    elif v > node_j:
                        self.nodes_to_row[k] -= 1
                self.matrix = self._color_node(node_j, node_i)
            else:
                for k in self.nodes_to_row.keys():
                    if self.nodes_to_row[k] > node_i:
                        self.nodes_to_row[k] -= 1
                    elif self.nodes_to_row[k] == node_i:
                        self.nodes_to_row[k] = node_j
                
                self.matrix = self._color_node(node_i, node_j)
    
    def _color_node(self, node_to_color, color_idx):
        n = self.matrix.shape[0] - 1
        
        new_mat = lil_matrix((n, n), dtype=np.int32)
        
        for i, j in zip(*self.matrix.nonzero()):
            new_i = i if i < node_to_color else i - 1
            new_j = j if j < node_to_color else j - 1
            
            if i == node_to_color:
                if j in [node_to_color, color_idx]:
                    new_mat[color_idx, color_idx] += self.matrix[i, j]
                else:
                    new_mat[color_idx, new_j] += self.matrix[i, j]
                    if new_mat[color_idx, new_j] < -1:
                        new_mat[color_idx, new_j] += 1
                        new_mat[new_j, color_idx] += 1
                        new_mat[color_idx, color_idx] -= 1
            
            elif j == node_to_color:
                if i in [node_to_color, color_idx]:
                    new_mat[color_idx, color_idx] += self.matrix[i, j]
                else:
                    new_mat[new_i, color_idx] += self.matrix[i, j]
                    if new_mat[new_i, color_idx] < -1:
                        new_mat[new_i, color_idx] += 1
                        new_mat[color_idx, new_i] += 1
                        new_mat[color_idx, color_idx] -= 1
            else:
                new_mat[new_i, new_j] = self.matrix[i, j]
        
        return new_mat.tocsr()

    def create_graph(self):
        return nx.Graph(diags(self.matrix.diagonal()).toarray() - self.matrix)
    
    def feasible_colors_for_node(self, node):
        idx = self.nodes_to_row[node]
        if idx not in self.color_idxs:
            yield len(self.color_idxs)
            for c in self.color_idxs:
                if self.matrix[idx, c] == 0:
                    yield self.color_idxs.index(c)
    
    def feasible_node_color_pairs(self):
        for node in self.nodes_to_row.keys():
            for_node = self.feasible_colors_for_node(node)
            yield node, next(for_node)
    

class ActionSpace(Discrete):
    """
    GYM AI representation of the action space for the graph coloring problem.
    
    In this version, the action space represents a pair, (n, c) that indicates represents to tag the node n with
    the color c.
    """
    
    def __init__(self, n, env: 'GraphColoringEnv'):
        super().__init__(n)
        self.env: GraphColoringEnv = env
    
    def sample(self):
        cur_state = self.env.observation_space.current_state
        feasible_actions = self.env.dynamics.admisible_actions(cur_state)
        
        return self.np_random.choice(feasible_actions)
    
    def feasible_actions(self, state: MatrixState=None):
        if state is None:
            state = self.env.observation_space.current_state
        for node, idx in state.nodes_to_row.items():
            for c in state.color_idxs:
                if state[idx, c] == 0:
                    yield node, state.color_idxs.index(c)


class StateSpace(Space):
    """GYM AI representation of the state space for the graph coloring problem."""
    
    def __init__(self, graph: nx.Graph):
        super().__init__()
        self.graph = graph
        self.current_state = MatrixState(graph)
    
    def reset_observation_space(self):
        c = MatrixState(self.graph)
        initial_node = list(self.graph.nodes)[0]
        c.color_node(initial_node, 0)
        self.current_state = c
    
    def sample(self):
        epoch = np.random.randint(0, len(self.graph))
        coloring = MatrixState(self.graph)
        
        for i in range(epoch):
            colored = False
            node = list(self.graph.nodes)[i]
            while not colored:
                color = np.random.randint(0, len(coloring))
                colored = coloring.color_node(node, color, strict=True)
    
    def contains(self, x):
        pass


class GraphColoringEnv(Env):
    """GYM AI graph coloring class."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.observation_space: StateSpace = StateSpace(graph)
        self.action_space: ActionSpace = ActionSpace(len(self.graph), self)
        
        self.visualizer = None
        self._done = False
    
    def simulate_transition_state(self, action, state=None):
        """
        Simulates a state transition without changing the current state.

        Parameters
        ----------
        action: int
            The action that wants to be simulated.
        state: Coloring
            The state in which the action wants to be simulated.

        Returns
        -------
        new_state
            Returns the state that is achieved by taking action in state.

        """
        if state is None:
            cur_state = self.observation_space.current_state
        else:
            cur_state = state
        distribution = self.dynamics.transition_kernel(cur_state, action)
        values = list(distribution.keys())
        probabilities = list(distribution.values())
        index = np.random.choice(a=range(len(values)), p=probabilities)
        
        return values[index]
    
    def step(self, action):
        """
		Given an action takes a step in the specified environment

		Parameters
		----------
		action: int
			An action to be takin in the current state.

		Returns
		-------
		Tuple
			next_state:
			The resulting state of taking the given action in the previous step
			reward:
			The resulting reward of taking the given action in the previous step
			done:
			Boolean flag that indicates if the arrived state is a terminal state
			info:
			Additional information.
		"""
        cur_state = self.observation_space.current_state
        next_state: 'Coloring' = self.simulate_transition_state(action)
        reward = self.dynamics.reward_(cur_state, next_state, action)
        
        info = {'state': next_state, 'colored_nodes': next_state.colored_nodes}
        
        if next_state.is_coloring(soft=True):
            done = True
        else:
            done = False
        info['found_coloring'] = done
        
        self.observation_space.current_state = next_state
        self._done = done
        return next_state, reward, done, info
    
    def reset(self, **kwargs):
        """Resets the environment to its initial state.

        Parameters
        ----------
        **kwargs
        """
        self.observation_space.reset_observation_space()
        self._done = False
        return self.observation_space.current_state
    
    def render(self, mode='human'):
        """Renders the current state."""
        # just raise an exception
        if mode == 'ansi':
            print(self.observation_space.current_state)
        elif mode == 'human':
            if self.visualizer is None:
                self.visualizer = Visualizer(graph=self.graph)
            
            self.visualizer.render(self.observation_space.current_state, final=self._done)
        else:
            super(GraphColoringEnv, self).render(mode=mode)
