import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
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
        self.node_embedding = torch.nn.Embedding(self.shapes[0], self.args.dimensions, padding_idx=0)
        self.node_noise_factors = torch.nn.Embedding(self.shapes[0], self.args.dimensions, padding_idx=0)
        self.feature_noise_factors = torch.nn.Embedding(self.shapes[1], self.args.dimensions, padding_idx=0)

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
        source_norm = torch.norm(source_node_vector, p = 2, dim = 1).view(-1,1)
        source_node_vector = source_node_vector/source_norm
        if score > 0.5:
            target_matrix = self.node_noise_factors(target)
        else:
            target_matrix = self.feature_noise_factors(target)
        target_norm = torch.norm(target_matrix, p = 2, dim = 1).view(-1,1)
        target_matrix = target_matrix/target_norm
        scores = torch.t(torch.nn.functional.softmax(torch.mm(target_matrix,torch.t(source_node_vector)), dim = 0))
        target = torch.tensor([0]).to(self.device)
        prediction_loss = torch.nn.functional.nll_loss(scores, target)
        hit = (torch.argmax(scores).item() == target.item())
        return prediction_loss, hit
        
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
        self.feature_count = max([max([val for val in v]) for k, v in self.features.items()])+1
        self.model = SINELayer(self.args, (self.node_count, self.feature_count), self.device).to(self.device)

    def simulate_walks(self):
        """
        Simulating truncated random walks.
        """
        self.walker = RandomWalker(self.graph, self.args.number_of_walks, self.args.walk_length)
        self.walker.do_walks()
        

    def pick_a_node_pair(self):
        walk_index = random.choice(range(self.node_count*self.args.number_of_walks))
        walk = self.walker.walks[walk_index]
        if random.uniform(0,1) >0.5:
            node_index = random.choice(range(self.args.walk_length-self.args.window_size))
            modifier = random.choice(range(1,self.args.window_size+1))
        else:
            node_index = random.choice(range(self.args.window_size,self.args.walk_length))
            modifier = random.choice(range(-self.args.window_size,0))
        source = walk[node_index]
        target = walk[node_index+modifier]
        return source, target

    def pick_a_node_feature_pair(self):
        index = random.choice(range(self.feature_index))
        source, target = self.features[index]
        return source, target

    def pick_noise_nodes(self):
        noise_nodes = []
        for i in range(self.args.node_noise_samples):
            walk_index = random.choice(range(self.node_count*self.args.number_of_walks))
            walk = self.walker.walks[walk_index]
            node_index = random.choice(range(self.args.walk_length))
            noise_nodes.append(walk[node_index])
        return noise_nodes          

    def pick_noise_features(self):
        noise_features = []
        for i in range(self.args.feature_noise_samples):
            index = random.choice(range(self.feature_index))
            source, target = self.features[index]
            noise_features.append(target)
        return noise_features

    def process_a_node(self, source_node, target, noise):
        source = torch.LongTensor([source_node]).to(self.device)
        targets = torch.LongTensor([target] + noise).to(self.device)
        return source, targets
        

    def update_loss(self, loss, step):
        self.cummulative_accuracy = self.cummulative_accuracy + loss
        self.budget.set_description("SINE (Accuracy=%g)" % round(self.cummulative_accuracy/(step+1),4))

    def fit(self):
        print("\n\nTraining the model.\n")
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.optimizer.zero_grad()
        self.cummulative_accuracy = 0
        self.budget = trange(self.args.budget, desc="Samples: ")
        losses = 0
        for step in self.budget:
            score = random.uniform(0,1)
            if score > 0.5:
                source_node, target = self.pick_a_node_pair()
                noise = self.pick_noise_nodes()
            else:
                source_node, target = self.pick_a_node_feature_pair()
                noise = self.pick_noise_features()
            source, targets = self.process_a_node(source_node, target, noise)
            loss, hit = self.model(source, targets, score)
            self.update_loss(hit, step)
            losses = losses + loss
            if (step + 1) %self.args.batch_size ==0:
                losses.backward(retain_graph = True)
                self.optimizer.step()
                losses = 0
                self.optimizer.zero_grad()
    def save_embedding(self):
        print("\n\nSaving the model.\n")
        nodes = torch.LongTensor([node for node in self.graph.nodes()]).to(self.device)
        self.embedding = self.model.node_embedding(nodes).cpu().detach().numpy()
        embedding_header = ["id"] + ["x_" + str(x) for x in range(self.args.dimensions)]
        self.embedding  = np.concatenate([np.array(range(self.embedding.shape[0])).reshape(-1,1),self.embedding],axis=1)
        self.embedding = pd.DataFrame(self.embedding, columns = embedding_header)
        self.embedding.to_csv(self.args.output_path, index = None)    
