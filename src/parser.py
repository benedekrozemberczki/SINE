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
                        default = "./input/edges/chameleon.csv",
	                help = "Edge list csv.")

    parser.add_argument("--feature-path",
                        nargs = "?",
                        default = "./input/features/chameleon.json",
	                help = "Node features csv.")

    parser.add_argument("--output-path",
                        nargs = "?",
                        default = "./output/chameleon_sine.csv",
	                help = "Target embedding csv.")

    parser.add_argument("--dimensions",
                        type = int,
                        default = 128,
	                help = "Number of dimensions. Default is 128.")

    parser.add_argument("--budget",
                        type = int,
                        default = 2*10**5,
	                help = "Number of samples generated. Default is 10**5.")

    parser.add_argument("--node-noise-samples",
                        type = int,
                        default = 5,
	                help = "Number of node negative samples. Default is 5.")

    parser.add_argument("--feature-noise-samples",
                        type = int,
                        default = 5,
	                help = "Number of feature negative samples. Default is 5.")

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
	                help = "Adam learning rate. Default is 0.001.")
    
    return parser.parse_args()
