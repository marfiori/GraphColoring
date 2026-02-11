import numpy as np

from graph_coloring.dynamics.base_class import Coloring
from abc import ABC, abstractmethod
from typing import Optional

from mdp.utilities.counters import Timer


class FunctionApproximation(ABC):
    def __init__(self, graph):
        self.graph = graph
        self._cost_to_go_cache = dict()
        self.timer = Timer(name=self.__class__.__name__, verbose=False)
    
    def __call__(self, partial_coloring: Optional = None, **kwargs) -> int:
        return self.reward_to_go(partial_coloring, **kwargs)
        
    def _compute_reward_to_go(self, partial_coloring: Optional = None) -> int:
        """Method to be implemented"""
        ...
    
    def reward_to_go(self, partial_coloring: Optional = None, truncate=True) -> int:
        """
        Returns the colors the more used by running the heuristic with the given partial coloring.

        Parameters
        ----------
        partial_coloring: Coloring
            A partial coloring

        Returns
        -------
        int:
            The number of colors the more needed to color the whole graph based in the implemented heuristic.
        """
        if partial_coloring is None:
            partial_coloring = Coloring(self.graph)
        _key = hash(partial_coloring)
        
        cost_to_go = self._cost_to_go_cache.get(_key, None)
        if cost_to_go is None:
            initial_colors = len(partial_coloring)
            # self.timer.start()
            full = self._compute_reward_to_go(partial_coloring)
            # self.timer.stop()
            if truncate:
                cost_to_go = max(full - initial_colors, 0)
                self._cost_to_go_cache[_key] = cost_to_go
            else:
                cost_to_go = full - initial_colors
        return cost_to_go


class BaseHeuristic(FunctionApproximation, ABC):
    """Base class for the heuristics."""
    
    def __init__(self, graph, **kwargs):
        super().__init__(graph)
        self._result_cache = dict()
    
    @abstractmethod
    def run_heuristic(self, partial_coloring: Optional = None) -> Coloring:
        """Implement this method with the heuristic."""
        ...
    
    def reward_to_go(self, partial_coloring: Optional = None, **kwargs) -> int:
        """
        Returns the colors the more used by running the heuristic with the given partial coloring.

        Parameters
        ----------
        partial_coloring: Coloring
            A partial coloring

        Returns
        -------
        int:
            The number of colors the more needed to color the whole graph based in the implemented heuristic.
        """
        if partial_coloring is None:
            partial_coloring = Coloring(self.graph)
        _key = hash(partial_coloring)
        
        cost_to_go = self._cost_to_go_cache.get(_key, None)
        if cost_to_go is None:
            initial_colors = len(partial_coloring)
            # self.timer.start()
            coloring = self.run_heuristic(partial_coloring)
            # self.timer.stop()
            cost_to_go = len(coloring) - initial_colors
            self._cost_to_go_cache[_key] = cost_to_go
        
        return cost_to_go

    @abstractmethod
    def run_heuristic_n_steps(self, partial_coloring: Optional = None, n: int = np.inf) -> Coloring:
        ...


class DualBound(FunctionApproximation, ABC):
    def __init__(self, graph):
        super().__init__(graph)
        
