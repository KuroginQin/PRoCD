# Baseline of running InfoMap from scratch

from infomap import Infomap
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
# ==========
num_edges = len(edges_)
if np.min(np.min(edges_))==1:
    edges = [(src-1, dst-1) for (src, dst) in edges_]
    num_nodes = np.max(np.max(edges))
else:
    edges = edges_.copy()
    num_nodes = np.max(np.max(edges))+1

# ====================
time_start = time.time()
im = Infomap(silent=True)
for (src, dst) in edges:
    im.add_link(src, dst)
# ==========
im.run()
lbl_set = set()
clus_res = [-1 for _ in range(num_nodes)]
for node in im.tree:
    if node.is_leaf:
        clus_res[node.node_id] = node.module_id - 1
        if node.module_id not in lbl_set:
            lbl_set.add(node.module_id)
num_clus_est = len(lbl_set)
time_end = time.time()
run_time = time_end - time_start

# ====================
mod_mtc = get_mod_mtc(edges, clus_res, num_clus_est)
print('InfoMap Est K %d Mod %f Time %f' % (num_clus_est, mod_mtc, run_time))
