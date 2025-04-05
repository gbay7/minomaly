import os
import argparse
from itertools import islice, chain
import time
from datetime import timedelta, datetime

from matplotlib import path

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
from minomaly_struct.search_agents import StrengthSearchAgent

import random
from collections import defaultdict
import networkx as nx

import matplotlib.pyplot as plt

nodes = [
      94, 95, 104, 159, 187, 220, 272, 280, 339, 354, 372, 379, 388, 405, 456,
      526, 538, 664, 680, 689, 705, 755, 771, 821, 876, 901, 972, 1000, 1066,
      1121, 1147, 1207, 1218, 1235, 1276, 1322, 1363, 1591, 1601, 1604, 1691,
      1698, 1708, 1796, 1846, 1852, 1855, 1978, 2003, 2110, 2204, 2232, 2245,
      2264, 2274, 2307, 2323, 2329, 2392, 2425, 2436, 2495, 2545, 2586, 2617,
      2671, 2684, 2687, 2710, 2716, 2753, 2766, 2779, 2781, 2817, 2863, 2896,
      2921, 2997, 3013, 3048, 3106, 3188, 3203, 3297, 3322, 3334, 3348, 3468,
      3478, 3553, 3565, 3586, 3632, 3645, 3719, 3771, 3784, 3800, 3945, 3953,
      3968, 3982, 4056, 4073, 4082, 4090, 4101, 4128, 4133, 4151, 4217, 4263,
      4283, 4290, 4326, 4464, 4482, 4660, 4674, 4722, 4725, 4890, 4923, 4927,
      4976, 5008, 5009, 5087, 5091, 5095, 5153, 5191, 5215, 5236, 5250, 5307,
      5350, 5397, 5431, 5433, 5451, 5458, 5463, 5483, 5534, 5575, 5633, 5678,
      5692, 5700, 5701, 5752, 5793, 5825, 5881, 5985, 5989, 5998, 6039, 6114,
      6127, 6151, 6152, 6158, 6160, 6191, 6205, 6257, 6292, 6294, 6330, 6344,
      6389, 6638, 6650, 6715, 6720, 6775, 6784, 6937, 6973, 7011, 7017, 7030,
      7048, 7076, 7224, 7280, 7283, 7325, 7445, 7503, 7552, 7607, 7626, 7656,
      7690, 7752, 7793, 7899, 7913, 8014, 8036, 8046, 8074, 8087, 8156, 8174,
      8177, 8207, 8223, 8224, 8249, 8275, 8293, 8308, 8316, 8340, 8352, 8563,
      8564, 8586, 8644, 8676, 8688, 8817, 8883, 8888, 8895, 8972, 8973, 9005,
      9037, 9142, 9169, 9190, 9213, 9220, 9310, 9338, 9362, 9400, 9541, 9632,
      9660, 9666, 9680, 9691, 9825, 9834, 9929, 9997, 10009, 10021, 10112,
      10181, 10199, 10246, 10249, 10301, 10381, 10414, 10454, 10511, 10518,
      10527, 10622, 10670, 10685, 10694, 10764, 10944, 10956, 11022, 11048,
      11096, 11152, 11176, 11207, 11304, 11306, 11431, 11461, 11500, 11545,
      11586, 11600, 11606, 11610, 11664, 11858, 11874, 11946, 12001, 12015,
      12055, 12082, 12102, 12171, 12233, 12237, 12299, 12312, 12355, 12364,
      12455, 12508, 12585, 12613, 12700, 12710, 12714, 12725, 12734, 12736,
      12744, 12759, 12764, 12815, 12818, 12892, 12947, 12948, 13005, 13006,
      13018, 13019, 13029, 13055, 13079, 13087, 13106, 13121, 13125, 13209,
      13287, 13291, 13351, 13364, 13464, 13480, 13486, 13538, 13640, 13659,
      13663, 13729, 13741, 13748
    ]

import requests
import shutil

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

dataset = [torch.load('pygod_inj_amazon.pth.tar', map_location='cpu')]

graphs = []
for i, graph in enumerate(dataset):
    graph = pyg_utils.to_networkx(graph).to_undirected()
    graphs.append(graph)

    subgraph = nx.subgraph(graph, nodes)

    # Calculate positions for the nodes to reduce overlap
    pos = nx.spring_layout(subgraph)

    nx.draw(subgraph, pos, with_labels=True)
    plt.savefig(f'11_{i}.pdf')
    plt.close()

