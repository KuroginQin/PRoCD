# Baseline of running LPA from scratch

import networkx as nx
import pickle
import time
from utils import *

# ====================
data_name = 'arxiv' # protein, arxiv, dblp, amazon, youtube, roadca
file_path = 'data/%s_edges.pickle' % (data_name)

# ====================
pkl_file = open(file_path, 'rb')
edges_ = pickle.load(pkl_file)
pkl_file.close()
edges = list(edges_)
# ==========
num_edges = len(edges)
if np.min(np.min(edges))==1:
    edges = [(src-1, dst-1) for (src, dst) in edges]
    num_nodes = np.max(np.max(edges))
else:
    edges = edges.copy()
    num_nodes = np.max(np.max(edges))+1

# ====================
G = nx.Graph()
G.add_edges_from(edges)

# ====================
time_start = time.time()
clus_mem = nx.community.asyn_lpa_communities(G, seed=0)
clus_mem = list(clus_mem)
time_end = time.time()
run_time = time_end - time_start

# ====================
time_start = time.time()
num_clus_est = len(clus_mem)
clus_res = [-1 for _ in range(num_nodes)]
for lbl_idx in range(num_clus_est):
    comm = clus_mem[lbl_idx]
    for node_idx in comm:
        clus_res[node_idx] = lbl_idx
# ==========
mod_mtc = get_mod_mtc(edges, clus_res, num_clus_est)
print('LPA Est K %d Mod %f Time %f' % (num_clus_est, mod_mtc, run_time))
