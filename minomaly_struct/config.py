import argparse
from common import utils

def parse_decoder(parser):
    dec_parser = parser.add_argument_group()
    # dec_parser.add_argument('--sample_method', type=str,
    #                     help='"tree" or "radial"')
    # dec_parser.add_argument('--motif_dataset', type=str,
    #                     help='Motif dataset')
    # dec_parser.add_argument('--radius', type=int,
    #                     help='radius of node neighborhoods')
    # dec_parser.add_argument('--radial_subgraph_sample_size', type=int,
    #                     help='number of nodes to take from each neighborhood')
    # dec_parser.add_argument('--out_path', type=str,
    #                     help='path to output candidate motifs')
    # dec_parser.add_argument('--n_clusters', type=int)
    # dec_parser.add_argument('--min_pattern_size', type=int)
    # dec_parser.add_argument('--max_pattern_size', type=int)
    # dec_parser.add_argument('--n_trials', type=int,
    #                     help='number of search trials to run')
    # dec_parser.add_argument('--search_strategy', type=str,
    #                     help='"greedy" or "mcts"')
    # dec_parser.add_argument('--use_whole_graphs', action="store_true",
    #     help="whether to cluster whole graphs or sampled node neighborhoods")
    dec_parser.add_argument('--min_neighborhood_size', type=int)
    dec_parser.add_argument('--max_neighborhood_size', type=int)
    dec_parser.add_argument('--n_neighborhoods', type=int)
    dec_parser.add_argument('--out_batch_size', type=int,
                        help='number of motifs to output per graph size')
    dec_parser.add_argument('--analyze', action="store_true")

    dec_parser.add_argument('--min_steps', type=int, default=1)
    dec_parser.add_argument('--max_steps', type=int, default=7)
    dec_parser.add_argument('--alpha', type=float, default=0.33)
    dec_parser.add_argument('--max_freq', type=float)
    # dec_parser.add_argument('--max_weight', type=float)
    # dec_parser.add_argument('--max_strength', type=float)
    dec_parser.add_argument('--max_unchanged', type=int, default=5)
    dec_parser.add_argument('--nodes_batch_size', type=int, default=16)
    dec_parser.add_argument('--reduction_method', type=str, default=None)
    dec_parser.add_argument('--n_beams', type=int, default=1)
    dec_parser.add_argument('--max_cands', type=int, default=None)
    dec_parser.add_argument('--add_verified_neighs', default=False, action='store_true')
    dec_parser.add_argument('--no_add_verified_neighs', dest='add_verified_neighs', action='store_false')
    # dec_parser.add_argument('--node_anchored', default=True, action='store_true')
    dec_parser.add_argument('--sample_random_cands', type=float, default=None)
    dec_parser.add_argument('--min_strength', type=float, default=0)
    dec_parser.add_argument('--min_neigh_repeat', type=int, default=2)
    
    dec_parser.add_argument('--n_threads', type=int, default=1)

    dec_parser.set_defaults(out_path="results/out-patterns.p",
                        # n_trials=1000,
                        # decode_thresh=0.5,
                        # radius=2,
                        # radial_subgraph_sample_size=0,
                        # sample_method="tree",
                        # skip="learnable",
                        # min_pattern_size=5,
                        # max_pattern_size=20,
                        # search_strategy="greedy",
                        n_neighborhoods=10000,
                        min_neighborhood_size=1,
                        max_neighborhood_size=30,
                        out_batch_size=10,
                        add_verified_neighs=False,
                        # node_anchored=True,
                        analyze=True)

    parser.set_defaults(dataset="inj_cora",
                        batch_size=1000,
                        analyze=True)
