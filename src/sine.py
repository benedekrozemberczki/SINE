"""SINE model."""

import random
import torch
import numpy as np
import pandas as pd
from tqdm import trange
from walkers import RandomWalker
from utils import read_graph, read_features

class SINELayer(torch.nn.Module):
    """
    Scalable Incomplete Network Embedding Layer.
    For details see: https://arxiv.org/abs/1810.06768
    """
    def __init__(self, args, shapes, device):
        """
        Initializing layer.
        :param args: Arguments object.
        :param shapes: Node number and feature number.
        :param device: CPU or CUDA placement.
        """
        super(SINELayer, self).__init__()
        self.args = args
        self.shapes = shapes
        self.device = device
        self.define_weights()
        self.initialize_weights()

    def define_weights(self):
        """
        Defining the embeddings.
        """
        self.node_embedding = torch.nn.Embedding(self.shapes[0],
                                                 self.args.dimensions,
                                                 padding_idx=0)

        self.node_noise_factors = torch.nn.Embedding(self.shapes[0],
                                                     self.args.dimensions,
                                                     padding_idx=0)

        self.feature_noise_factors = torch.nn.Embedding(self.shapes[1],
                                                        self.args.dimensions,
                                                        padding_idx=0)

    def initialize_weights(self):
        """
        Initializing the weights.
        """
        torch.nn.init.xavier_normal_(self.node_embedding.weight.data)
        torch.nn.init.xavier_normal_(self.node_noise_factors.weight.data)
        torch.nn.init.xavier_normal_(self.feature_noise_factors.weight.data)

    def forward(self, source, target, score):
        """
        Forward propagation pass.
        :param source: Source node.
        :param target: Context nodes.
        :param score: Random score to make decision whther feature or node is picked.
        """
        source_node_vector = self.node_embedding(source)
        if score > 0.5:
            target_matrix = self.node_noise_factors(target)
        else:
            target_matrix = self.feature_noise_factors(target)
        scores = target_matrix*source_node_vector
        scores = torch.sum(scores, dim=1)
        scores = torch.clamp(scores, -20, 20)
        scores = torch.sigmoid(scores)
        targets = [1.0]+[0.0 for i in range(self.args.noise_samples)]
        targets = torch.FloatTensor(targets).to(self.device)
        main_loss = targets*torch.log(scores)+(1.0-targets)*torch.log(1.0-scores)
        main_loss = -torch.mean(main_loss)
        return main_loss

class SINETrainer(object):
    '''
    Class to train the Scalable Incomplete Network Embedding model.
    '''
    def __init__(self, args):
        """
        Initializing the training object.
        :param args: Arguments parsed from command line.
        """
        self.args = args
        self.graph = read_graph(self.args.edge_path)
        self.features = read_features(self.args.feature_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_model()
        self.simulate_walks()

    def initialize_model(self):
        """
        Initializing the SINE model.
        """
        self.node_count = len(self.graph.nodes())
        self.feature_index = len(self.features.keys())
        self.feature_count = [max([val for val in v]) for k, v in self.features.items()]
        self.feature_count = max(self.feature_count) + 1
        shape = (self.node_count, self.feature_count)
        self.model = SINELayer(self.args, shape, self.device).to(self.device)

    def simulate_walks(self):
        """
        Simulating truncated random walks.
        """
        self.walker = RandomWalker(self.graph, self.args.number_of_walks, self.args.walk_length)
        self.walker.do_walks()

    def pick_a_node_pair(self):
        """
        Choosing a random node pair skip-gram style.
        """
        walk_index = random.choice(range(self.node_count*self.args.number_of_walks))
        walk = self.walker.walks[walk_index]
        if random.uniform(0, 1) > 0.5:
            node_index = random.choice(range(self.args.walk_length-self.args.window_size))
            modifier = random.choice(range(1, self.args.window_size+1))
        else:
            node_index = random.choice(range(self.args.window_size, self.args.walk_length))
            modifier = random.choice(range(-self.args.window_size, 0))
        source = walk[node_index]
        target = walk[node_index+modifier]
        return source, target

    def pick_a_node_feature_pair(self):
        """
        Choosing a random node-feature pair.
        """
        index = random.choice(range(self.feature_index))
        source, target = self.features[index]
        return source, target

    def pick_noise_nodes(self):
        """
        Picking noise nodes based on node frequency distribution in walks.
        """
        noise_nodes = []
        for _ in range(self.args.noise_samples):
            walk_index = random.choice(range(self.node_count*self.args.number_of_walks))
            walk = self.walker.walks[walk_index]
            node_index = random.choice(range(self.args.walk_length))
            noise_nodes.append(walk[node_index])
        return noise_nodes

    def pick_noise_features(self):
        """
        Picking noise feature based on feature frequency.
        """
        noise_features = []
        for _ in range(self.args.noise_samples):
            index = random.choice(range(self.feature_index))
            source, target = self.features[index]
            noise_features.append(target)
        return noise_features

    def process_a_node(self, source_node, target, noise):
        """
        Given a node, target and noise samples create indexing tensors.
        :param source_node: Source node.
        :param target: Target node/feature index.
        :param noise: Noise samples.
        :return source: Source node tensor.
        :return targets: Real and noise target indices.
        """
        source = torch.LongTensor([source_node]).to(self.device)
        targets = torch.LongTensor([target] + noise).to(self.device)
        return source, targets

    def update_accuracy(self, loss, step):
        """
        Updating the cummulative predictive accuracy.
        :param hit: Boolean describing correct prediction.
        :param step: Number of sampled processed.
        """
        self.cummulative_accuracy = self.cummulative_accuracy + loss.item()
        self.budget.set_description("SINE (Loss=%g)" % round(self.cummulative_accuracy/(step+1), 4))

    def fit(self):
        """
        Model training.
        """
        print("\n\nTraining the model.\n")
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.optimizer.zero_grad()
        self.cummulative_accuracy = 0
        self.budget = trange(self.args.budget, desc="Samples: ")
        losses = 0
        for step in self.budget:
            score = random.uniform(0, 1)
            if score > 0.5:
                source_node, target = self.pick_a_node_pair()
                noise = self.pick_noise_nodes()
            else:
                source_node, target = self.pick_a_node_feature_pair()
                noise = self.pick_noise_features()
            source, targets = self.process_a_node(source_node, target, noise)
            loss = self.model(source, targets, score)
            losses = losses + loss
            self.update_accuracy(loss, step)
            if (step+1) % self.args.batch_size == 0:
                losses.backward(retain_graph=True)
                self.optimizer.step()
                losses = 0
                self.optimizer.zero_grad()

    def save_embedding(self):
        """
        Saving the node embedding.
        """
        print("\n\nSaving the model.\n")
        nodes = [node for node in range(self.model.shapes[0])]
        nodes = torch.LongTensor(nodes).to(self.device)
        embedding = self.model.node_embedding(nodes).cpu().detach().numpy()
        cols = ["id"] + ["x_" + str(x) for x in range(self.args.dimensions)]
        embedding = [np.array(range(embedding.shape[0])).reshape(-1, 1), embedding]
        embedding = np.concatenate(embedding, axis=1)
        embedding = pd.DataFrame(embedding, columns=cols)
        embedding.to_csv(self.args.output_path, index=None)
