import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it gives an embedding of the Wikipedia Chameleons dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """

    parser = argparse.ArgumentParser(description = "Run SINE.")


    parser.add_argument("--edge-path",
                        nargs = "?",
                        default = "./input/chameleon_edges.csv",
	                help = "Edge list csv.")

    parser.add_argument("--feature-path",
                        nargs = "?",
                        default = "./input/chameleon_features.json",
	                help = "Node features csv.")

    parser.add_argument("--output-path",
                        nargs = "?",
                        default = "./output/chameleon_sine.csv",
	                help = "Target embedding csv.")

    parser.add_argument("--dimensions",
                        type = int,
                        default = 32,
	                help = "Number of dimensions. Default is 128.")

    parser.add_argument("--budget",
                        type = int,
                        default = 10**5,
	                help = "Number of samples generated. Default is 10**5.")

    parser.add_argument("--noise-samples",
                        type = int,
                        default = 5,
	                help = "Number of negative samples. Default is 5.")

    parser.add_argument("--batch-size",
                        type = int,
                        default = 32,
	                help = "Mini-batch sample number. Default is 32.")

    parser.add_argument("--walk-length",
                        type = int,
                        default = 80,
	                help = "Number of nodes in truncated random walk. Default is 80.")

    parser.add_argument("--number-of-walks",
                        type = int,
                        default = 10,
	                help = "Number of random walks per source node. Default is 10.")

    parser.add_argument("--window-size",
                        type = int,
                        default = 5,
	                help = "Skip-gram window size. Default is 5.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.01,
	                help = "Adam learning rate. Default is 0.01.")
    
    return parser.parse_args()
