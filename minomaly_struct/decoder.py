import os
import argparse
from itertools import islice, chain
import pickle
import time
from datetime import timedelta, datetime

# from matplotlib import path

import minomaly_struct.utils as dec_utils
import minomaly_struct.sampling as sampl

import numpy as np
import torch

from torch_geometric.datasets import TUDataset, PPI
from torch_geometric.datasets import Planetoid
import torch_geometric.utils as pyg_utils

from common import models
from common import utils
from common import combined_syn
from minomaly_struct.config import parse_decoder
from subgraph_matching.config import parse_encoder
from minomaly_struct.search_agents import StrengthSearchAgent, Beam

import random
from collections import defaultdict
import networkx as nx

import matplotlib.pyplot as plt

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

torch.cuda.empty_cache()

import gc
import torch
import numpy as np
import os

def clear_all_cache():
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
  
    gc.collect()
    
    # Reset numpy memory
    np.random.seed(42)  # Reset random state
    
    # Free memory pools from PyTorch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
    
    # Clear IPython notebook memory (if in notebook)
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.magic('reset -f')
    except:
        pass

# Call before starting any major computation
clear_all_cache()

def set_deterministic():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Optional: Deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# Call at start of training
set_deterministic()

def batch_nodes(nodes, batch_size):
    total_nodes = len(nodes)
    print(f'Total starting nodes: {total_nodes}')
    processed_nodes = 0
    random.shuffle(nodes)
    it = iter(nodes)
    batch_number = 0
    start_total_time = time.time()
    while True:
        start_time = time.time()
        batch = list(islice(it, batch_size))
        if not batch:
            break
        batch_number += 1
        print(f'--- Batch {batch_number} ---')
        get_time = lambda: timedelta(seconds=time.time() - start_time)
        yield batch, batch_number, get_time
        processed_nodes += len(batch)
        remaining_nodes = total_nodes - processed_nodes
        batch_time = timedelta(seconds=time.time() - start_time)
        total_time = timedelta(seconds=time.time() - start_total_time)
        print(f'Batch {batch_number}: Processed nodes: {processed_nodes}/{total_nodes} (Remaining: {remaining_nodes}), Batch time: {str(batch_time)}, Total time: {str(total_time)}')
    print(f'Total time: {str(total_time)}')

def sample_neighs(args, graphs, anomalies, model, max_steps, results_out_path):
    file_path = os.path.join(results_out_path, f"{args.dataset}_{args.n_neighborhoods}_{args.min_neighborhood_size}_{args.max_neighborhood_size}_{args.node_anchored}_{args.batch_size}.p")
    # if os.path.exists(file_path):
    #     print("Loading sampled neighborhoods and embeddings from", file_path)
    #     with open(file_path, "rb") as f:
    #         return pickle.load(f)

    sample_method = sampl.TreeSampling(graphs, args.n_neighborhoods,
        args.min_neighborhood_size, args.max_neighborhood_size)
    neighs, anchors, real_anchors = sample_method.sample(args.node_anchored)

    anomalous_nodes = [[idx for idx, is_anomalous in enumerate(graph_anomalies) if is_anomalous] for graph_anomalies in anomalies]
    sample_method = sampl.RadialSampling(graphs, 2, subgraph_sample_size=max_steps, nodes=anomalous_nodes)
    anomalous_neighs, anomalous_anchors, anomalous_real_anchors = sample_method.sample(args.node_anchored)

    embs, anomalous_embs = [], []

    print("Embedding node neighborhoods")
    embs = dec_utils.embed_neighs(model.emb_model, neighs, anchors, args.batch_size, args.node_anchored)
    print("Embedding anomalous node neighborhoods")
    anomalous_embs = dec_utils.embed_neighs(model.emb_model, anomalous_neighs, anomalous_anchors, len(anomalous_anchors), args.node_anchored)

    results = ((neighs, anchors, real_anchors, embs), (anomalous_neighs, anomalous_anchors, anomalous_real_anchors, anomalous_embs), anomalous_nodes)

    print("Saving sampled neighborhoods and embeddings to", file_path)
    if not os.path.exists(os.path.dirname(results_out_path)):
        os.makedirs(os.path.dirname(results_out_path))
    with open(file_path, "wb") as f:
        pickle.dump(results, f)
    
    return results

def get_starting_nodes(args, embs, model, max_freq, real_anchors, neighs, max_steps, embs_np, results_out_path):
    file_path = os.path.join(results_out_path, f"{args.dataset}_{args.n_neighborhoods}_{args.min_neighborhood_size}_{args.max_neighborhood_size}_{args.node_anchored}_{args.batch_size}_starting_nodes.p")
    # if os.path.exists(file_path):
    #     print("Loading starting nodes from", file_path)
    #     with open(file_path, "rb") as f:
    #         return pickle.load(f)

    starting_nodes, outlier_embs_np = dec_utils.find_outliers(embs, model, max_freq, real_anchors, neighs, None)
    _starting_nodes, _outlier_embs_np = dec_utils.get_outlier_neighs(embs_np, real_anchors, neighs, max_steps, contamination='auto')
    starting_nodes = starting_nodes.union(_starting_nodes)
    outlier_embs_np = np.concatenate((outlier_embs_np, _outlier_embs_np))

    results = (starting_nodes, outlier_embs_np)

    print("Saving starting nodes to", file_path)
    if not os.path.exists(os.path.dirname(results_out_path)):
        os.makedirs(os.path.dirname(results_out_path))
    with open(file_path, "wb") as f:
        pickle.dump(results, f)

    return results

def pattern_growth(dataset, task, args):
    torch.cuda.reset_peak_memory_stats()
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    if not os.path.exists(f"plots/{current_time}/cluster"):
        os.makedirs(f"plots/{current_time}/cluster/tracked_nodes")

    model = models.OrderEmbedder(1, args.hidden_dim, args)
    model.to(utils.get_device())
    model.eval()
    print("Loading model from", args.model_path)
    model.load_state_dict(torch.load(args.model_path,
        map_location=utils.get_device()))

    graphs, anomalies = [], []
    for i, graph in enumerate(dataset):
        if task == "struct-anomaly":
            anomalies.append(graph.y >> 1 & 1)
        if not type(graph) == nx.Graph:
            graph = pyg_utils.to_networkx(graph).to_undirected()
        graphs.append(graph)
    
    all_nodes = [node for graph in graphs for node in graph.nodes]
    num_nodes = len(all_nodes)
    min_steps = args.min_steps
    # each step means adding one node to the neighborhood (max structural neighborhood size)
    max_steps = args.max_steps
    alpha = args.alpha
    max_freq, max_weight = (args.max_freq / num_nodes), (max_steps / num_nodes)
    # max_strength = (alpha * max_freq) + ((1-alpha) * (max_weight))
    max_strength = Beam.calculate_score(max_freq, max_weight, alpha)

    max_unchanged = args.max_unchanged
    unchange_direction = False
    nodes_batch_size = args.nodes_batch_size
    reduction_method = args.reduction_method
    n_beams = args.n_beams
    max_cands = args.max_cands
    add_verified_neighs = args.add_verified_neighs
    sample_random_cands = args.sample_random_cands
    min_strength = args.min_strength
    min_neigh_repeat = args.min_neigh_repeat
    
    n_threads = args.n_threads

    # print the parameters
    print(f"min_steps: {min_steps}")
    print(f"max_steps: {max_steps}")
    print(f"alpha: {alpha}")
    print(f"max_freq: {max_freq}")
    print(f"max_weight: {max_weight}")
    print(f"max_strength: {max_strength}")
    print(f"max_unchanged: {max_unchanged}")
    print(f"unchange_direction: {unchange_direction}")
    print(f"nodes_batch_size: {nodes_batch_size}")
    print(f"reduction_method: {reduction_method}")
    print(f"n_beams: {n_beams}")
    print(f"max_cands: {max_cands}")
    print(f"add_verified_neighs: {add_verified_neighs}")
    print(f"sample_random_cands: {sample_random_cands}")
    print(f"min_strength: {min_strength}")
    print(f"min_neigh_repeat: {min_neigh_repeat}")
    print(f"n_threads: {n_threads}")

    (neighs, anchors, real_anchors, embs), (anomalous_neighs, anomalous_anchors, anomalous_real_anchors, anomalous_embs), anomalous_nodes = sample_neighs(args, graphs, anomalies, model, max_steps, "savings/embeddings/")
    embs_np = torch.stack(embs).view(-1, embs[0].shape[-1]).cpu().numpy()
    starting_nodes, outlier_embs_np = get_starting_nodes(args, embs, model, max_freq, real_anchors, neighs, max_steps, embs_np, "savings/starting_nodes/")

    # starting_nodes = set([4290, 12355, 4674, 5350, 1147, 9005, 11096, 6389, 4725, 4056, 2392, 187, 12734, 13087])
    # starting_nodes = set([
    #   94, 95, 104, 159, 187, 220, 272, 280, 339, 354, 372, 379, 388, 405, 456,
    #   526, 538, 664, 680, 689, 705, 755, 771, 821, 876, 901, 972, 1000, 1066,
    #   1121, 1147, 1207, 1218, 1235, 1276, 1322, 1363, 1591, 1601, 1604, 1691,
    #   1698, 1708, 1796, 1846, 1852, 1855, 1978, 2003, 2110, 2204, 2232, 2245,
    #   2264, 2274, 2307, 2323, 2329, 2392, 2425, 2436, 2495, 2545, 2586, 2617,
    #   2671, 2684, 2687, 2710, 2716, 2753, 2766, 2779, 2781, 2817, 2863, 2896,
    #   2921, 2997, 3013, 3048, 3106, 3188, 3203, 3297, 3322, 3334, 3348, 3468,
    #   3478, 3553, 3565, 3586, 3632, 3645, 3719, 3771, 3784, 3800, 3945, 3953,
    #   3968, 3982, 4056, 4073, 4082, 4090, 4101, 4128, 4133, 4151, 4217, 4263,
    #   4283, 4290, 4326, 4464, 4482, 4660, 4674, 4722, 4725, 4890, 4923, 4927,
    #   4976, 5008, 5009, 5087, 5091, 5095, 5153, 5191, 5215, 5236, 5250, 5307,
    #   5350, 5397, 5431, 5433, 5451, 5458, 5463, 5483, 5534, 5575, 5633, 5678,
    #   5692, 5700, 5701, 5752, 5793, 5825, 5881, 5985, 5989, 5998, 6039, 6114,
    #   6127, 6151, 6152, 6158, 6160, 6191, 6205, 6257, 6292, 6294, 6330, 6344,
    #   6389, 6638, 6650, 6715, 6720, 6775, 6784, 6937, 6973, 7011, 7017, 7030,
    #   7048, 7076, 7224, 7280, 7283, 7325, 7445, 7503, 7552, 7607, 7626, 7656,
    #   7690, 7752, 7793, 7899, 7913, 8014, 8036, 8046, 8074, 8087, 8156, 8174,
    #   8177, 8207, 8223, 8224, 8249, 8275, 8293, 8308, 8316, 8340, 8352, 8563,
    #   8564, 8586, 8644, 8676, 8688, 8817, 8883, 8888, 8895, 8972, 8973, 9005,
    #   9037, 9142, 9169, 9190, 9213, 9220, 9310, 9338, 9362, 9400, 9541, 9632,
    #   9660, 9666, 9680, 9691, 9825, 9834, 9929, 9997, 10009, 10021, 10112,
    #   10181, 10199, 10246, 10249, 10301, 10381, 10414, 10454, 10511, 10518,
    #   10527, 10622, 10670, 10685, 10694, 10764, 10944, 10956, 11022, 11048,
    #   11096, 11152, 11176, 11207, 11304, 11306, 11431, 11461, 11500, 11545,
    #   11586, 11600, 11606, 11610, 11664, 11858, 11874, 11946, 12001, 12015,
    #   12055, 12082, 12102, 12171, 12233, 12237, 12299, 12312, 12355, 12364,
    #   12455, 12508, 12585, 12613, 12700, 12710, 12714, 12725, 12734, 12736,
    #   12744, 12759, 12764, 12815, 12818, 12892, 12947, 12948, 13005, 13006,
    #   13018, 13019, 13029, 13055, 13079, 13087, 13106, 13121, 13125, 13209,
    #   13287, 13291, 13351, 13364, 13464, 13480, 13486, 13538, 13640, 13659,
    #   13663, 13729, 13741, 13748
    # ])
    
    if args.analyze:
        analyze_out_path = f"plots/{current_time}/analyze.png"
        starting_analyze_out_path = f"plots/{current_time}/analyze_start.png"
        print(f"Analyzing dataset samples neighborhoods (saving to {analyze_out_path})")
        anomalous_embs_np = torch.stack(anomalous_embs).cpu().numpy().squeeze()
        labels = np.array([0] * embs_np.shape[0] + [1] * anomalous_embs_np.shape[0])

        all_embs = np.concatenate((embs_np, anomalous_embs_np), axis=0)
        dec_utils.scatter_embs(all_embs, labels, legend=['Node neighborhood', 'Anomalous neighborhood'], reduce_dim=reduction_method, out_path=starting_analyze_out_path)

        dec_utils.scatter_embs(all_embs, labels, legend=['Node neighborhood', 'Anomalous neighborhood'], reduce_dim=reduction_method, out_path=None)
        dec_utils.scatter_embs(outlier_embs_np, None, legend='Starting neighborhood', reduce_dim=reduction_method, out_path=analyze_out_path)

    print('Start anomaly search agent')

    verified, analyze_embs, counts = set([]), [], defaultdict(lambda: defaultdict(list))

    anomalies_out_path = f"plots/{current_time}/anomalies.json"
    anomalous_nodes = list(chain(*anomalous_nodes))

    true_starting_nodes = list(set(starting_nodes).intersection(anomalous_nodes))

    dec_utils.write_verified(anomalies_out_path, [], {
        'starting_nodes': list(starting_nodes),
        'starting_nodes_len': len(starting_nodes),
        'true_anomalies': anomalous_nodes,
        'true_anomalies_len': len(anomalous_nodes),
        'true_starting_nodes': true_starting_nodes,
        'true_starting_nodes_len': len(true_starting_nodes),
    })

    # return
    
    tracked_anchors = []
    track_dict = dict()
    track_out_graphs = []

    def track_node(beam_set, label, stats={}):
        if len(beam_set) == 0:
            return
        # We don't take the other beams into account, we take one strength score as representative
        beam = beam_set[0]
        if beam.anchor() in tracked_anchors:
            track_dict[beam.anchor()] = {
                'scores': track_dict.get(beam.anchor(), {'scores': []})['scores'] + [beam.score],
                'is_true': beam.anchor() in anomalous_nodes,
            }
            track_out_graphs.append((beam.get_neigh(), beam.anchor()))

    count_copied_beams = dict()
    total_time = timedelta(seconds=0)
    
    for graph_idx in range(len(graphs)):
        # filter starting nodes to only those in the current graph
        starting_nodes_graph = list(set(starting_nodes).intersection(graphs[graph_idx].nodes))
        # for nodes, batch_number in batch_nodes(list(graphs[graph_idx].nodes), nodes_batch_size):
        for nodes, batch_number, get_time in batch_nodes(starting_nodes_graph, nodes_batch_size):

            nodes = list(set(nodes).difference([beam.anchor() for beam in verified]))
            agent = StrengthSearchAgent(max_unchanged, max_strength,
                model, graphs, embs, node_anchored=args.node_anchored,
                analyze=args.analyze, n_beams=n_beams,
                min_strength=min_strength, alpha=alpha, sample_random_cands=sample_random_cands, max_cands=max_cands,
                unchange_direction=unchange_direction, min_steps=min_steps, max_steps=max_steps, add_verified_neighs=add_verified_neighs, track_node=track_node, n_threads=n_threads)
            agent.run_search(nodes, graph_idx)

            verified.update(agent.verified)
            
            if add_verified_neighs:
                for beam in agent.copied_verified:
                    count_copied_beams[beam.anchor()] = beam, count_copied_beams.get(beam.anchor(), (None, 0))[1] + 1    
                beams_to_remove = []
                for beam, count in count_copied_beams.values():
                    if count > min_neigh_repeat:
                        verified.add(beam)
                        beams_to_remove.append(beam.anchor())

                for beam in beams_to_remove:
                    count_copied_beams.pop(beam)


            batch_time = get_time()
            total_time = batch_time + total_time
            stat_results = dec_utils.get_stat_results(anomalous_nodes, verified, all_nodes)

            dec_utils.write_verified(f"plots/{current_time}/batch_{batch_number}_anomalies.json", agent.verified, {
                'stat_results': stat_results,
                'true_anomalies': anomalous_nodes,
                'predicted_anomalies_len': len(agent.verified),
                'batch_number': batch_number,
                'tracked_nodes': track_dict,
                'batch_time': str(batch_time),
                'total_time': str(total_time),
                'gpu_memory': torch.cuda.max_memory_allocated() / (1024**3),
            })
            dec_utils.export_patterns(track_out_graphs, graphs_out_path=f"plots/{current_time}/cluster/tracked_nodes", results_out_path=f"results/{current_time}/out-tracked_patterns_batch{batch_number}.p", node_anchored=args.node_anchored, count_by_anchor_=True)
            
            dec_utils.concat_counts(counts, agent.counts)
            analyze_embs.extend(agent.analyze_embs)
            
            track_out_graphs = []

    if args.analyze:
        analyze_out_path = f"plots/{current_time}/analyze_search.png"
        print(f"Analyzing search results (saving to {analyze_out_path})")
        analyze_embs_np = np.array(analyze_embs)
        dec_utils.scatter_embs(embs_np, None, legend='Node neighborhood', reduce_dim=None, out_path=None)
        dec_utils.scatter_embs(analyze_embs_np, None, legend='Analyzed search', reduce_dim=None, out_path=analyze_out_path)
    
    out_graphs = dec_utils.get_uniq_patterns(counts, args.out_batch_size)

    stat_results = dec_utils.get_stat_results(anomalous_nodes, verified, all_nodes)

    dec_utils.write_verified(anomalies_out_path, verified, {
        'dataset': args.dataset,
        'stat_results': stat_results,
        'n_neighborhoods': args.n_neighborhoods,
        'true_anomalies': anomalous_nodes,
        'predicted_anomalies_len': len(verified),
        'unique_anomalous_patterns_len': len(out_graphs),
        'node_anchored': args.node_anchored,
        'search_params': {
            'alpha': alpha,
            'max_steps': max_steps,
            'max_strength': max_strength,
            'min_strength': min_strength,
            'max_unchanged': max_unchanged,
            'unchange_direction': unchange_direction,
            'nodes_batch_size': nodes_batch_size,
            'max_freq': max_freq,
            'max_weight': max_weight,
            'n_beams': n_beams,
            'max_cands': max_cands,
            'add_verified_neighs': add_verified_neighs,
            'sample_random_cands': sample_random_cands,
            'min_neigh_repeat': min_neigh_repeat,
        },
        'starting_nodes': list(starting_nodes),
        'starting_nodes_len': len(starting_nodes),
        'true_anomalies': anomalous_nodes,
        'true_anomalies_len': len(anomalous_nodes),
        'true_starting_nodes': true_starting_nodes,
        'true_starting_nodes_len': len(true_starting_nodes),
        'tracked_nodes': track_dict,
        'total_time': str(total_time),
        'gpu_memory': torch.cuda.max_memory_allocated() / (1024**3),
    })

    dec_utils.export_patterns(out_graphs, graphs_out_path=f"plots/{current_time}/cluster", results_out_path=f"results/{current_time}/out-patterns.p", node_anchored=args.node_anchored)
    dec_utils.export_patterns(track_out_graphs, graphs_out_path=f"plots/{current_time}/cluster/tracked_nodes", results_out_path=f"results/{current_time}/out-tracked_patterns.p", node_anchored=args.node_anchored, count_by_anchor_=True)

import requests
import shutil

def make_plant_dataset(size):
    generator = combined_syn.get_generator([size])
    random.seed(3001)
    np.random.seed(14853)
    # PATTERN 1
    pattern = generator.generate(size=10)
    nx.draw(pattern, with_labels=True)
    plt.savefig("plots/cluster/plant-pattern.png")
    plt.close()
    graphs = []
    for i in range(1000):
        graph = generator.generate()
        n_old = len(graph)
        graph = nx.disjoint_union(graph, pattern)
        for j in range(1, 3):
            u = random.randint(0, n_old - 1)
            v = random.randint(n_old, len(graph) - 1)
            graph.add_edge(u, v)
        graphs.append(graph)
    return graphs

def load_data(name, cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.pygod/data')
    file_path = os.path.join(cache_dir, name+'.pt')
    zip_path = os.path.join(cache_dir, name+'.pt.zip')

    if os.path.exists(file_path):
        data = torch.load(file_path)
    else:
        url = "https://github.com/pygod-team/data/raw/main/" + name + ".pt.zip"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        shutil.unpack_archive(zip_path, cache_dir)
        data = torch.load(file_path)
    return data

def main():
    parser = argparse.ArgumentParser(description='Decoder arguments')
    parse_encoder(parser)
    parse_decoder(parser)
    args = parser.parse_args()
    # args.dataset = "enzymes"

    print("Using dataset {}".format(args.dataset))
    if args.dataset == 'inj_cora':
        dataset = [torch.load('pygod_inj_cora.pth.tar', map_location='cpu')]
        task = 'struct-anomaly'
    elif args.dataset == 'inj_amazon':
        dataset = [torch.load('pygod_inj_amazon.pth.tar', map_location='cpu')]
        task = 'struct-anomaly'
    elif args.dataset == 'inj_flickr':
        dataset = [torch.load('pygod_inj_flickr.pth.tar', map_location='cpu')]

    pattern_growth(dataset, task, args) 

if __name__ == '__main__':
    main()