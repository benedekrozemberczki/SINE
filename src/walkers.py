import random
import networkx as nx
from tqdm import tqdm
from tqdm import trange

class RandomWalker:
    """
    Class to generate vertex sequences.
    """
    def __init__(self, graph, repetitions, length):
        """
        :param graph: Networkx graph.
        :param repetitions: Number of truncated random walks per source node.
        :param length: Length of truncated random walks.
        """
        print("Model initialization started.")
        self.graph = graph
        self.repetitions = repetitions 
        self.length = length
        
    def small_walk(self, start_node):
        """
        Doing a truncated random walk.
        :param start_node: Start node for random walk.
        :return walk: Truncated random walk with fixed maximal length.
        """
        walk = [start_node]
        while len(walk) < self.args.walk_length:
            if len(nx.neighbors(self.graph,walk[-1])) == 0:
                break
            walk = walk + [random.sample(nx.neighbors(self.graph,walk[-1]),1)[0]]
        return walk
       
    def do_walks(self):
        """
        Do a series of random walks.
        """
        print("\nRandom walks started.\n")
        self.walks = []
        for rep in trange(self.repetitions, desc = "Series: "):
            for node in tqdm(self.graph.nodes(), desc="Nodes: "):
                walk = self.small_walk(node)
                self.walks.append(walk)
