from graph_coloring.rollout import *
from graph_coloring.heuristics import *
from graph_coloring import Coloring
from graph_coloring.graph_generator import *

import networkx as nx

from mdp.utilities.counters import Timer
from time import sleep
from graph_coloring.data_handler import *
import pandas as pd
import matplotlib.pyplot as plt
logging.getLogger('matplotlib.font_manager').disabled = True


def time_summary(algorith: RolloutColoring):
    d = {''}


def rollout():
    all_instances = pd.read_csv('data/DimacsInstances/index.csv')
    #
    instance = 'mulsol.i.5.col'
    # graph = graph_from_dimacs('data/DimacsInstances/instances', instance)
    graph = nx.generators.random_graphs.erdos_renyi_graph(10, 0.2, seed=754)

    # print(f'Graph {instance} with {len(g)} nodes, and {len(g.edges)} edges')
    t = Timer('Heuristic', verbose=True)
    t.start()
    heuristic = GreedyColoring(graph)
    h_coloring = heuristic.run_heuristic()
    t.stop()
    print(len(h_coloring))
    # draw_coloring(g, h_coloring)
    # plt.show()
    sleep(0.1)

    t = Timer('Rollout', verbose=True)
    heuristic = GreedyColoring(graph)
    ro = RolloutColoring(graph, heuristic, depth=2)
    t.start()
    ro_coloring = ro.solve()
    t.stop()
    print(len(ro_coloring))
    print(f'Bound pruning: {ro.bound_counter}')
    print(f'Heuristic calls: {ro.heuristics_call}')


def rollout_lb():
    # all_instances = pd.read_csv('data/DimacsInstances/index.csv')
    #
    # instance = 'mulsol.i.5.col'
    # graph = graph_from_dimacs('data/DimacsInstances/instances', instance)
    # graph = nx.generators.random_graphs.erdos_renyi_graph(10, 0.2, seed=754)
    np.random.seed(754)
    size = 100
    num_colors = 7
    density = 0.2
    color_distribution = None
    graph = generate_graph(
        size,
        num_colors,
        color_distribution=color_distribution,
        arc_connection_kwargs=dict(density=density)
    )

    # print(f'Graph {instance} with {len(g)} nodes, and {len(g.edges)} edges')
    heuristic = GreedyColoring(graph)
    
    lower_bound = SpectralBound(graph)
    
    ro = RolloutLB(
        graph=graph,
        heuristic=heuristic,
        lower_bound=lower_bound,
        function_approximation=heuristic,
        depth=2,
        fortified=False
    )
    ro_coloring = ro.solve()
    
    print(len(ro_coloring))
    ro.bounds_plot(optimal_value=num_colors)
    print(f'Bound pruning: {ro.bound_counter}')
    print(f'Heuristic calls: {ro.heuristics_call}')


def spectral_bounds():
    np.random.seed(757)
    g = generate_graph(70, 69)
    
    suggested = create_coloring(g)
    
    c = Coloring(g)
    if suggested.is_coloring(False):
        sb = SpectralBound(graph=g)
        heuristic = GreedyColoring(g)
    
        print(sb.reward_to_go(c), heuristic.reward_to_go(c))
    else:
        suggested.is_coloring(False)


def graph_gen():
    g = generate_graph(1000, 100)
    

def main():
    rollout_lb()


if __name__ == '__main__':
    main()




