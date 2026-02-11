import numpy as np
from math import ceil

from graph_coloring.dynamics.base_class import GraphColoringEnv
from graph_coloring.heuristics.base import BaseHeuristic, DualBound, FunctionApproximation
import logging

import matplotlib.pyplot as plt

from mdp.utilities.counters import TallyCounter, Timer

logging.basicConfig(level=logging.DEBUG)

Action = int


class RolloutColoring:
    """
	An Implementation of the rollout policy for the Graoph coloring problem.

	Attributes
	----------
	graph: nx.Graph
		The graph to be colored
	heuristic: BaseHeuristic
		The heuristic to use.
	max_depth: int
		The maximum depth to use in the rollout online exploration.
	total_reward: int
		The estimated chromatic number for the given graph
	"""
    DEFAULT_DEPTH = 2
    
    def __init__(self, graph, heuristic: BaseHeuristic, depth=None, **kwargs):
        self.graph = graph
        
        self.env = GraphColoringEnv(graph)
        self.heuristic = heuristic
        
        self.max_depth = RolloutColoring.DEFAULT_DEPTH if depth is None else depth
        
        self._temp_path = []
        # Local bound
        self._temp_reward = self.heuristic.reward_to_go()
        
        # Global bound
        self.total_reward = self._temp_reward
        self.path = []
        
        self.logger = logging.getLogger(__name__)
        
        self.bound_counter = TallyCounter(name='Bound pruning')
        self.heuristics_call = TallyCounter(name='heuristic call')
        self.timer = Timer(name=self.__class__.__name__)
    
    def update_temporary_search(self, temp_path, temp_reward):
        """
		Updates the online exploration incumbent path and reward.

		Parameters
		----------
		temp_path:
			The current temporary path. A sequence of state and action paris that indicates the path used.
		temp_reward:
			Temporary number of colors used.
		"""
        cur_state = temp_path[-1]
        heuristic_reward = self.heuristic.reward_to_go(cur_state)
        self.heuristics_call.count()
        if temp_reward + heuristic_reward < self._temp_reward:
            self._temp_reward = temp_reward + heuristic_reward
            self._temp_path = temp_path
    
    def update_global(self, candidate):
        if candidate < self.total_reward:
            self.total_reward = candidate
    
    def check_pruning_strategies(self, temp_path, temp_reward):
        """
		Checks the pruning strategies for a given partial path.

		Parameters
		----------
		temp_path:
			The current temporary path. A sequence of state and action paris that indicates the path used.
		temp_reward:
			Temporary number of colors used.

		Returns
		-------
		bool
			True if the path should be further explored,
			False otherwise, i.e, there exists arguments to prune that path.
		"""
        flag = True
        if temp_reward > self.total_reward:
            flag = False
            self.bound_counter.count()
        return flag
    
    def roll_out_search(self, rollout_path, rollout_reward, depth):
        """
		Performs the rollout search.

		If the depth is greater or equal than the max depth or the current state is terminal
		(i.e., is a coloring) the function tries to update the current incumbent temporary path, i.e., calls
		update_temporary_search. Otherwise, it recursively explores all the next possible paths by calling the
		roll_out_search for all the following states.

		Note that the recursion eventually ends because the depth argument is increased by 1 every time the
		function is called.

		Parameters
		----------
		rollout_path:
			The current temporary path. A sequence of state and action paris that indicates the path used.
		rollout_reward:
			Temporary number of colors used.
		depth:
			The current exploration depth.
		"""
        if depth >= self.max_depth or rollout_path[-1].is_coloring(soft=True):
            self.update_temporary_search(rollout_path, rollout_reward)
        else:
            for action in self.env.action_space.feasible_actions(rollout_path[-1]):
                next_state, instant_reward = self.env.simulate_transition_state(action, rollout_path[-1])
                
                new_path = rollout_path + [action, next_state]
                new_reward = rollout_reward + instant_reward
                
                if self.check_pruning_strategies(new_path, new_reward):
                    self.roll_out_search(rollout_path=new_path, rollout_reward=new_reward, depth=depth + 1)
    
    def roll_out(self, state) -> Action:
        """
		Initializes the local search.

		This is done by calling the roll_out_search method at _temp_path [state], with
		a _temp_reward of -inf and a depth of 0.

		Parameters
		----------
		state:
			State at which the local search will be initialized.

		Returns
		-------
			The action to be taken after performing the local rollout search,
			i.e., the action that maximizes the self.max_depth rollout minimization problem.
		"""
        self._temp_path = [state]
        self._temp_reward = float('inf')
        
        self.roll_out_search(self._temp_path, len(state), 0)
        self.update_global(self._temp_reward)
        
        return self._temp_path[1]
    
    def solve(self):
        """
		Finds a Coloring for the given graph using the rollout policy.

		In every step it calls the roll_out function to build the rollout policy.
		Returns
		-------
		Coloring:
			The coloring that results of running the rollout policy with the given heuristic.
		"""
        self.timer.start()
        coloring = self.env.reset()
        self.path.append(coloring)
        done = False
        
        while not done:
            action = self.roll_out(coloring)
            coloring, cost, done, info = self.env.step(action)
            self.path.append(coloring)

        self.timer.stop()
        return coloring


class RolloutLB(RolloutColoring):
    def __init__(
            self,
            graph,
            heuristic: BaseHeuristic,
            lower_bound: DualBound,
            function_approximation: FunctionApproximation = None,
            depth=None,
            fortified=False,
            **kwargs
    ):
        super().__init__(graph, heuristic, depth, **kwargs)
        self.lower_bound: DualBound = lower_bound
        self.function_approximation = lower_bound if function_approximation is None else function_approximation
        self.fortified = fortified
        
    def update_temporary_search(self, temp_path, temp_reward):
        """
        Updates the online exploration incumbent path and reward.

        Parameters
        ----------
        temp_path:
            The current temporary path. A sequence of state and action paris that indicates the path used.
        temp_reward:
            Temporary number of colors used.
        """
        cur_state = temp_path[-1]
        function_approx = self.function_approximation.reward_to_go(cur_state)
        self.heuristics_call.count()
        if temp_reward + function_approx < self._temp_reward:
            self._temp_reward = temp_reward + function_approx
            self._temp_path = temp_path
    
    def check_pruning_strategies(self, temp_path, temp_reward):
        """
        Checks the pruning strategies for a given partial path.

        Parameters
        ----------
        temp_path:
            The current temporary path. A sequence of state and action paris that indicates the path used.
        temp_reward:
            Temporary number of colors used.

        Returns
        -------
        bool
            True if the path should be further explored,
            False otherwise, i.e, there exists arguments to prune that path.
        """
        flag = True
        if temp_reward > self.total_reward:
            flag = False
            self.bound_counter.count()
        return flag
    
    def roll_out(self, state) -> Action:
        """
        Initializes the local search.

        This is done by calling the roll_out_search method at _temp_path [state], with
        a _temp_reward of -inf and a depth of 0.

        Parameters
        ----------
        state:
            State at which the local search will be initialized.

        Returns
        -------
            The action to be taken after performing the local rollout search,
            i.e., the action that maximizes the self.max_depth rollout minimization problem.
        """
        self._temp_path = [state]
        self._temp_reward = float('inf')
        
        self.roll_out_search(self._temp_path, len(state), 0)

        # Now we run the heuristic from
        new_ub = len(self._temp_path[-1]) + self.heuristic(self._temp_path[-1])
        self.update_global(new_ub)
        if self.fortified:
            if new_ub <= self.total_reward:
                return self._temp_path[1]
            else:
                next_state = self.heuristic.run_heuristic_n_steps(state, 1)
                node = next_state.colored_nodes.difference(state.colored_nodes).pop()
                return next_state(node)
        else:
            if len(self._temp_path) > 1:
                return self._temp_path[1]
            else:
                next_state = self.heuristic.run_heuristic_n_steps(state, 1)
                node = next_state.colored_nodes.difference(state.colored_nodes).pop()
                return next_state(node)

    def solve(self):
        """
        Finds a Coloring for the given graph using the rollout policy.

        In every step it calls the roll_out function to build the rollout policy.
        Returns
        -------
        Coloring:
            The coloring that results of running the rollout policy with the given heuristic.
        """
        coloring = self.env.reset()
        self.path.append(coloring)
        done = False
        ub = float('inf')
        lb = float('-inf')

        while not done:
            action = self.roll_out(coloring)
            coloring, cost, done, info = self.env.step(action)
            self.path.append(coloring)
            
            lb = max(lb, len(coloring) + self.lower_bound(coloring))
            ub = min(ub, len(coloring) + self.heuristic(coloring))
            if ub - lb < 1:
                coloring = self.heuristic.run_heuristic(partial_coloring=coloring)
                break
                
        return coloring
    
    def bounds_plot(self, optimal_value=None):
        lb = []
        ub = []
        
        for p in self.path:
            if len(lb) > 0:
                lb.append(max(lb[-1], len(p) + self.lower_bound(p)))
            else:
                lb.append(len(p) + self.lower_bound(p))
            
            if len(ub) > 0:
                ub.append(min(ub[-1], len(p) + self.heuristic(p)))
            else:
                ub.append(len(p) + self.heuristic(p))
        
        x = range(len(lb))
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(x, lb, label='Lower Bound')
        ax.plot(x, ub, label='Upper Bound')
        
        if optimal_value is not None:
            plt.hlines(optimal_value, 0, len(lb) - 1, 'green', '--', label='Optimal Value')
        
        plt.xlabel('Number of iterations')
        plt.ylabel(r'Numbers of colors used ($\hat{\chi}(G)$)')
        ax.legend()
        plt.show()


class RandomRolloutLB(RolloutLB):
    
    def __init__(self,
                 graph,
                 heuristic: BaseHeuristic,
                 lower_bound: DualBound,
                 function_approximation: FunctionApproximation = None,
                 depth=None,
                 **kwargs):
        super().__init__(graph, heuristic, lower_bound, function_approximation, depth, **kwargs)
        self.temp_actions_weight = {}

    def update_temporary_search(self, temp_path, temp_reward):
        """
        Updates the online exploration incumbent path and reward.

        Parameters
        ----------
        temp_path:
            The current temporary path. A sequence of state and action paris that indicates the path used.
        temp_reward:
            Temporary number of colors used.
        """
        cur_state = temp_path[-1]
        heuristic_reward = self.function_approximation.reward_to_go(cur_state)
        self.heuristics_call.count()
        self.temp_actions_weight[(temp_path[1], temp_path[2])] = heuristic_reward
    
    def roll_out(self, state) -> Action:
        """
        Initializes the local search.

        This is done by calling the roll_out_search method at _temp_path [state], with
        a _temp_reward of -inf and a depth of 0.

        Parameters
        ----------
        state:
            State at which the local search will be initialized.

        Returns
        -------
            The action to be taken after performing the local rollout search,
            i.e., the action that maximizes the self.max_depth rollout minimization problem.
        """
        self._temp_path = [state]
        self._temp_reward = float('inf')
        self.temp_actions_weight = {}
        self.roll_out_search(self._temp_path, len(state), 0)

        w = np.array(list(map(abs, self.temp_actions_weight.values()))) + 1
        if len(w) > 0:
            idx = np.random.choice(range(len(w)), p=w/w.sum())
            action, next_state = list(self.temp_actions_weight.keys())[idx]
            new_ub = len(next_state) + self.heuristic(next_state)
        else:
            new_ub = np.inf
            
        if new_ub < self.total_reward:
            self.update_global(new_ub)
            return action
        else:
            next_state = self.heuristic.run_heuristic_n_steps(state, 1)
            node = next_state.colored_nodes.difference(state.colored_nodes).pop()
            return next_state(node)
    
    def solve(self):
        """
        Finds a Coloring for the given graph using the rollout policy.

        In every step it calls the roll_out function to build the rollout policy.
        Returns
        -------
        Coloring:
            The coloring that results of running the rollout policy with the given heuristic.
        """
        coloring = self.env.reset()
        self.path.append(coloring)
        done = False
        
        while not done:
            action = self.roll_out(coloring)
            coloring, cost, done, info = self.env.step(action)
            self.path.append(coloring)
            
            lb = self.lower_bound(coloring)
            ub = self.heuristic(coloring)
            if ub - lb < 1:
                coloring = self.heuristic.run_heuristic(partial_coloring=coloring)
                break
        
        return coloring

