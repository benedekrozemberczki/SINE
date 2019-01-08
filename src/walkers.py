import random
import networkx as nx
from tqdm import tqdm
from tqdm import trange
class RandomWalker:
    """
    Class to generate vertex sequences.
    """
    def __init__(self, graph, repetitions, length):
        print("Model initialization started.")
        self.graph = graph
        self.repetitions = repetitions 
        self.length = length

    def small_walk(self, start_node):
        """
        Generate a node sequence from a start node.
        """
        walk = [start_node]
        while len(walk) != self.length:
            end_point = walk[-1]
            neighbors = nx.neighbors(self.graph, end_point)
            if len(neighbors) > 0:
                walk = walk + random.sample(neighbors, 1)
            else:
                break
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
