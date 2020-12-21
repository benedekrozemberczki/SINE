Scalable Incomplete Network Embedding 
============================================

![License](https://img.shields.io/github/license/benedekrozemberczki/SINE.svg?color=blue&style=plastic) [![Arxiv](https://img.shields.io/badge/ArXiv-1904.05003-orange.svg?color=blue&style=plastic)](https://arxiv.org/pdf/1904.05003.pdf) [![codebeat badge](https://codebeat.co/badges/c01c7c97-d873-4ba6-ac5c-21147aae5f74)](https://codebeat.co/projects/github-com-benedekrozemberczki-sine-master) [![repo size](https://img.shields.io/github/repo-size/benedekrozemberczki/SINE.svg)](https://github.com/benedekrozemberczki/SINE/archive/master.zip)


A PyTorch  implementation of **Scalable Incomplete Network Embedding (ICDM 2018)**.

<div style="text-align:center"><img src ="sine.jpg" ,width=720/></div>
	
### Abstract
<p align="justify">
Attributed network embedding aims to learn low-dimensional vector representations for nodes in a network, where each node contains rich attributes/features describing node content. Because network topology structure and node attributes often exhibit high correlation, incorporating node attribute proximity into network embedding is beneficial for learning good vector representations. In reality, large-scale networks often have incomplete/missing node content or linkages, yet existing attributed network embedding algorithms all operate under the assumption that networks are complete. Thus, their performance is vulnerable to missing data and suffers from poor scalability. In this paper, we propose a Scalable Incomplete Network Embedding (SINE) algorithm for learning node representations from incomplete graphs. SINE formulates a probabilistic learning framework that separately models pairs of node-context and node-attribute relationships. Different from existing attributed network embedding algorithms, SINE provides greater flexibility to make the best of useful information and mitigate negative effects of missing information on representation learning. A stochastic gradient descent based online algorithm is derived to learn node representations, allowing SINE to scale up to large-scale networks with high learning efficiency. We evaluate the effectiveness and efficiency of SINE through extensive experiments on real-world networks. Experimental results confirm that SINE outperforms state-of-the-art baselines in various tasks, including node classification, node clustering, and link prediction, under settings with missing links and node attributes. SINE is also shown to be scalable and efficient on large-scale networks with millions of nodes/edges and high-dimensional node features.</p>

This repository provides an implementation of SINE as described in the paper:

> SINE: Scalable Incomplete Network Embedding.
> Daokun Zhang, Jie Yin, Xingquan Zhu, Chengqi Zhang.
> ICDM, 2018.
> [[Paper]](https://arxiv.org/pdf/1810.06768.pdf)


The SINE model is available in [[Karate Club]](https://github.com/benedekrozemberczki/karateclub) framework.

The original C implementation is available [[here]](https://github.com/daokunzhang/SINE).

### Requirements
The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx          2.4
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
torch             1.1.0.
torchvision       0.3.0
```
### Datasets
<p align="justify">
The code takes an input graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. Sample graphs for the `Twitch Brasilians` and `Wikipedia Chameleons` are included in the  `input/` directory. </p>
<p align="justify">
The feature matrix can be stored two ways as a **sparse binary** one. For simplicity, it is a JSON. Nodes are keys of the json and features are the values. For each node feature column ids are stored as elements of a list. The feature matrix is structured as:</p>

```javascript
{ 0: [0, 1, 38, 1968, 2000, 52727],
  1: [10000, 20, 3],
  2: [],
  ...
  n: [2018, 10000]}
```

### Options
<p align="justify">
Learning of the embedding is handled by the `src/main.py` script which provides the following command line arguments.</p>

#### Input and output options

```
  --edge-path    STR     Input graph path.           Default is `input/chameleon_edges.csv`.
  --feature-path STR     Input Features path.        Default is `input/chameleon_features.json`.
  --output-path  STR     Embedding path.             Default is `output/chameleon_sine.csv`.
```

#### Model options

```
  --dimensions              INT       Number of embeding dimensions.         Default is 128.
  --budget                  INT       Sampling budget.                       Default is 10^5.
  --noise-samples           INT       Number of noise samples.               Default is 5.
  --batch-size              INT       Number of source nodes per batch.      Default is 32.
  --walk-length             INT       Truncated random walk length.          Default is 80.  
  --number-of-walks         INT       Number of walks per source node.       Default is 10.
  --window-size             INT       Skip-gram window size.                 Default is 5.
  --learning-rate           FLOAT     Learning rate value.                   Default is 0.001.
```

### Examples
<p align="justify">
The following commands learn a graph embedding and write the embedding to disk. The node representations are ordered by the ID.</p>
<p align="justify">
Creating a SINE embedding of the default dataset with the default hyperparameter settings. Saving the embedding at the default path.</p>

```
python src/main.py
```
<p align="center">
<img style="float: center;" src="sine_run_example.jpg">
</p>

Creating a SINE embedding of the default dataset with 256 dimensions.
```
python src/main.py --dimensions 256
```
Creating a SINE embedding of the default dataset with a low sampling budget.
```
python src/main.py --budget 1000
```
Creating an embedding of an other dense structured dataset the `Twitch Brasilians`. Saving the output in a custom folder.
```
python src/main.py --edge-path input/ptbr_edges.csv --feature-path input/ptbr_features.json --output-path output/ptbr_sine.csv
```
