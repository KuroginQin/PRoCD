# Baseline of running Locale from scratch

from sdp_clustering import leiden_locale, init_random_seed
from scipy.io import mmread
import pickle
import time
from utils import *

init_random_seed(0)

# ====================
data_name = 'arxiv' # protein, arxiv, dblp, amazon, youtube, roadca
file_path = 'data/%s_edges.pickle' % (data_name)
# ==========
EPS = 1e-6 # Stopping criterion for optimization problem
max_outer = 10 # Maximum number of outer iterations
max_lv = 10 # 'Maximum number of levels in an outer iteration
max_inner = 2 # Maximum number of inner iters for optimization
verbose = 0 # Verbosity
k = 8

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
    node_idx_0base = False
else:
    edges = edges.copy()
    num_nodes = np.max(np.max(edges))+1
    node_idx_0base = True

# ====================
f_output = open('buf/Locale_buf.mtx', 'w')
f_output.write('%%MatrixMarket matrix coordinate pattern symmetric\n')
f_output.write('%d %d %d\n' % (num_nodes, num_nodes, num_edges))
for edge in edges:
    if node_idx_0base:
        f_output.write('%d %d\n' % (edge[0]+1, edge[1]+1))
    else:
        f_output.write('%d %d\n' % (edge[0], edge[1]))
f_output.close()
graph = mmread('buf/Locale_buf.mtx')

# ==================
time_start = time.time()
clus_res_ = leiden_locale(graph, k, EPS, max_outer, max_lv, max_inner, verbose)
time_end = time.time()
run_time = time_end - time_start

# ====================
lbl_map = {}
lbl_cnt = 0
clus_res = []
for lbl in clus_res_:
    if lbl not in lbl_map:
        lbl_map[lbl] = lbl_cnt
        clus_res.append(lbl_cnt)
        lbl_cnt += 1
    else:
        clus_res.append(lbl_map[lbl])
num_clus_est = lbl_cnt
# ==========
mod_mtc = get_mod_mtc(edges, clus_res, num_clus_est)
print('Locale Est K %d Mod %f Time %f' % (num_clus_est, mod_mtc, run_time))
