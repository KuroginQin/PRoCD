from modules.X0 import *
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sdp_clustering import leiden_locale, init_random_seed
from infomap import Infomap

import pickle
import random
import time
from utils import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
rand_seed_gbl = 0
setup_seed(rand_seed_gbl)

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_init_res(ind_est, edges):
    # ====================
    num_nodes = np.max(np.max(edges)) + 1
    num_edges = len(edges)
    psv_src_idxs = []
    psv_dst_idxs = []
    psv_vals = []
    for t in range(num_edges):
        if ind_est[t] >= 0.5:
            (src, dst) = edges[t]
            # ==========
            psv_src_idxs.append(src)
            psv_dst_idxs.append(dst)
            psv_vals.append(1.0)
            # ==========
            psv_src_idxs.append(dst)
            psv_dst_idxs.append(src)
            psv_vals.append(1.0)
    adj_sp = sp.csr_matrix((psv_vals, (psv_src_idxs, psv_dst_idxs)))
    # ==========
    # Extract clusters/communities w.r.t. connected components
    num_clus_est, clus_res_ = connected_components(csgraph=adj_sp, directed=False, return_labels=True)
    clus_res_ = list(clus_res_)
    comp_mem_cnt = [0 for _ in range(num_clus_est)]
    for i in range(len(clus_res_)):
        comp_mem_cnt[clus_res_[i]] += 1
    clus_res = [-(i+1) for i in range(num_nodes)]
    for i in range(len(clus_res_)):
        if comp_mem_cnt[clus_res_[i]] > 1:
            clus_res[i] = clus_res_[i]
    print('#NODE %d #SUP-NODE %d' % (num_nodes, num_clus_est))
    if num_clus_est == 1:
        return clus_res
    # ==========
    init_node_idx = 0
    init_node_map = {}
    init_edge_map = {}
    for (src, dst) in edges:
        src_lbl = clus_res[src]
        dst_lbl = clus_res[dst]
        #if src_lbl == dst_lbl: continue
        # ==========
        if src_lbl not in init_node_map:
            src_node_idx = init_node_idx
            init_node_map[src_lbl] = init_node_idx
            init_node_idx += 1
        else:
            src_node_idx = init_node_map[src_lbl]
        # ==========
        if dst_lbl not in init_node_map:
            dst_node_idx = init_node_idx
            init_node_map[dst_lbl] = init_node_idx
            init_node_idx += 1
        else:
            dst_node_idx = init_node_map[dst_lbl]
        # ==========
        if src_node_idx > dst_node_idx:
            tmp = src_node_idx
            src_node_idx = dst_node_idx
            dst_node_idx = tmp
        if (src_node_idx, dst_node_idx) not in init_edge_map:
            init_edge_map[(src_node_idx, dst_node_idx)] = 1.0
        else:
            init_edge_map[(src_node_idx, dst_node_idx)] += 1.0
    # ==========
    init_edges = [(src, dst, init_edge_map[(src, dst)]) for (src, dst) in init_edge_map]
    #init_edges = sorted(init_edges)
    init_src_idxs = []
    init_dst_idxs = []
    init_vals = []
    for (src, dst, val) in init_edges:
        init_src_idxs.append(src)
        init_dst_idxs.append(dst)
        init_vals.append(val)
        if src != dst:
            init_src_idxs.append(dst)
            init_dst_idxs.append(src)
            init_vals.append(val)
    graph = sp.coo_matrix((init_vals, (init_src_idxs, init_dst_idxs)))

    return clus_res, graph, init_edges, init_node_map, init_node_idx

def clus_reorder(num_nodes, clus_res):
    clus_res_ = []
    lbl_cnt = 0
    lbl_map = {}
    for i in range(num_nodes):
        lbl = clus_res[i]
        if lbl not in lbl_map:
            lbl_map[lbl] = lbl_cnt
            clus_res_.append(lbl_cnt)
            lbl_cnt += 1
        else:
            clus_res_.append(lbl_map[lbl])

    return clus_res_, lbl_cnt

def LPA_rfn(clus_res, adj_list):
    # ====================
    num_nodes = len(adj_list)
    stop_flag = False
    iter_cnt = 0
    # ==========
    clus_res_ = clus_res.copy()
    while not stop_flag:
        stop_flag = True
        nxt_clus_res = [-(i + 1) for i in range(num_nodes)]
        for node_idx in range(num_nodes):
            # ==========
            neigh_lbl_cnt = {}
            for neigh in adj_list[node_idx]:
                #if clus_res_[neigh] < 0: continue
                if clus_res_[neigh] not in neigh_lbl_cnt:
                    neigh_lbl_cnt[clus_res_[neigh]] = 1
                else:
                    neigh_lbl_cnt[clus_res_[neigh]] += 1
            # ==========
            lbl = -(node_idx+1)
            lbl_cnt = 0
            for l in neigh_lbl_cnt:
                if neigh_lbl_cnt[l] > lbl_cnt:
                    lbl = l
                    lbl_cnt = neigh_lbl_cnt[l]
            nxt_clus_res[node_idx] = lbl
            # ==========
            if nxt_clus_res[node_idx] != clus_res_[node_idx]: stop_flag = False
        # ==========
        clus_res_ = nxt_clus_res
        iter_cnt += 1
        if iter_cnt > 20: break

    return clus_res_

def InfoMap_rfn(init_edges, init_node_map, init_num_nodes, clus_res, num_nodes):
    # ====================
    im = Infomap(silent=True)
    for (src, dst, wei) in init_edges:
        im.add_link(src, dst, wei)
    im.run()
    # ==========
    rfn_res_ = [-1 for _ in range(init_num_nodes)]
    for node in im.tree:
        if node.is_leaf:
            rfn_res_[node.node_id] = node.module_id - 1
    rfn_res = [rfn_res_[init_node_map[clus_res[i]]] for i in range(num_nodes)]

    return rfn_res

def locale_rfn(graph, init_node_map, clus_res, num_nodes, rand_seed):
    init_random_seed(rand_seed)
    rfn_res_ = leiden_locale(graph, k=8, eps=1e-6, max_outer=10, max_lv=10, max_inner=2, verbose=0)
    rfn_res = [rfn_res_[init_node_map[clus_res[i]]] for i in range(num_nodes)]

    return rfn_res

# ====================
data_name = 'youtube'

feat_dims = [64, 64, 64] # Layer conf of feat red unit
feat_dim = feat_dims[0]
emb_dim = feat_dims[-1]
num_GNN_lyr = 4 # L_GNN
num_MLP_lyr_tmp = 2 # Number of MLP layers for temp param L_BC

BCE_param = 100 # alpha
mod_rsl = 100 # lambda
num_add_pairs = 10000 # n_S
drop_rate = 0.2

eph_idx = 89 # n_P

# ====================
pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
tst_edges = pickle.load(pkl_file)
pkl_file.close()

# ====================
if np.min(np.min(tst_edges)) == 1:
    tst_edges = [(src-1, dst-1) for (src, dst) in tst_edges]
tst_num_nodes = np.max(np.max(tst_edges)) + 1
tst_num_edges = len(tst_edges)
# ==========
tst_degs = [0 for _ in range(tst_num_nodes)]
tst_adj_list = [set() for _ in range(tst_num_nodes)]
tst_src_idxs = []
tst_dst_idxs = []
for (src, dst) in tst_edges:
    # ==========
    tst_degs[src] += 1
    tst_degs[dst] += 1
    # ==========
    tst_adj_list[src].add(dst)
    tst_adj_list[dst].add(src)
    # ==========
    tst_src_idxs.append(src)
    tst_dst_idxs.append(dst)
# ==========
# Sample set of node pairs for inference
src_rand_idxs = [i for i in range(tst_num_nodes)]
dst_rand_idxs = [i for i in reversed(src_rand_idxs)]
random.shuffle(src_rand_idxs)
random.shuffle(dst_rand_idxs)
tst_edges_inf = tst_edges.copy()
for t in range(num_add_pairs):
    src = src_rand_idxs[t]
    dst = dst_rand_idxs[t]
    tst_edges_inf.append((src, dst))
    tst_src_idxs.append(src)
    tst_dst_idxs.append(dst)
# ==========
# Get GNN support
idxs, vals = get_sp_GCN_sup(tst_edges, tst_num_nodes)
idxs_tnr = torch.LongTensor(idxs).to(device)
vals_tnr = torch.FloatTensor(vals).to(device)
sup_tnr = torch.sparse.FloatTensor(idxs_tnr.t(), vals_tnr,
                                   torch.Size([tst_num_nodes, tst_num_nodes])).to(device)

# ====================
# Define the model
mdl = MDL_X0(feat_dims, num_GNN_lyr, num_MLP_lyr_tmp, drop_rate).to(device)
mdl.load_state_dict(torch.load('chpt/%s_mdl_%.1f_%.1f_%d.pt'
                               % (data_name, BCE_param, mod_rsl, eph_idx)))
mdl.eval()

# ====================
# Extract input feature - Gaussian random projection
time_start = time.time()
# ==========
idxs, vals = get_sp_mod_feat(tst_edges, tst_degs)
idxs_tnr = torch.LongTensor(idxs).to(device)
vals_tnr = torch.FloatTensor(vals).to(device)
sp_mod_tnr = torch.sparse.FloatTensor(idxs_tnr.t(), vals_tnr,
                                      torch.Size([tst_num_nodes, tst_num_nodes])).to(device)
rand_mat = get_rand_proj_mat(tst_num_nodes, feat_dim, rand_seed=rand_seed_gbl)
rand_mat_tnr = torch.FloatTensor(rand_mat).to(device)
red_feat_tnr = torch.spmm(sp_mod_tnr, rand_mat_tnr)
time_end = time.time()
# ==========
feat_time = time_end - time_start

# ====================
time_start = time.time()
emb_tnr, lft_tmp_tnr, rgt_tmp_tnr = mdl(red_feat_tnr, sup_tnr)
edge_ind_est = get_edge_ind_est(emb_tnr, lft_tmp_tnr, rgt_tmp_tnr, tst_src_idxs, tst_dst_idxs)
time_end = time.time()
prop_time = time_end - time_start
# ==========
if torch.cuda.is_available():
    edge_ind_est = edge_ind_est.cpu().data.numpy()
else:
    edge_ind_est = edge_ind_est.data.numpy()
#print('edge_ind_est', edge_ind_est)

# ====================
# Extract initialized result
time_start = time.time()
clus_res_init, init_graph, init_edges, init_node_map, init_num_nodes = get_init_res(edge_ind_est, tst_edges_inf)
time_end = time.time()
ext_time = time_end - time_start
# ==========
# Refined by LPA
time_start = time.time()
clus_res_LPA = LPA_rfn(clus_res_init, tst_adj_list)
time_end = time.time()
rfn_time_LPA = time_end - time_start
# ==========
# Refined by InfoMap
time_start = time.time()
clus_res_IM = InfoMap_rfn(init_edges, init_node_map, init_num_nodes, clus_res_init, tst_num_nodes)
time_end = time.time()
rfn_time_IM = time_end - time_start
# ==========
# Refined by Locale
time_start = time.time()
clus_res_Lcl = locale_rfn(init_graph, init_node_map, clus_res_init, tst_num_nodes, rand_seed=0)
time_end = time.time()
rfn_time_Lcl = time_end - time_start

# ====================
clus_res_init_, num_clus_est = clus_reorder(tst_num_nodes, clus_res_init)
mod_init = get_mod_mtc(tst_edges, clus_res_init_, num_clus_est)
print('INIT EST-K %d MOD %.4f' % (num_clus_est, mod_init))
# ==========
clus_res_LPA, num_clus_est = clus_reorder(tst_num_nodes, clus_res_LPA)
mod_LPA = get_mod_mtc(tst_edges, clus_res_LPA, num_clus_est)
time_LPA = feat_time + prop_time + ext_time + rfn_time_LPA
print('LPA EST-K %d MOD %.4f TIME %.4f (%.4f %.4f %.4f %.4f)'
      % (num_clus_est, mod_LPA, time_LPA, feat_time, prop_time, ext_time, rfn_time_LPA))
# ==========
clus_res_IM, num_clus_est = clus_reorder(tst_num_nodes, clus_res_IM)
mod_IM = get_mod_mtc(tst_edges, clus_res_IM, num_clus_est)
time_IM = feat_time + prop_time + ext_time + rfn_time_IM
print('InfoMap EST-K %d MOD %.4f TIME %.4f (%.4f %.4f %.4f %.4f)'
      % (num_clus_est, mod_IM, time_IM, feat_time, prop_time, ext_time, rfn_time_IM))
# ==========
clus_res_Lcl, num_clus_est = clus_reorder(tst_num_nodes, clus_res_Lcl)
mod_Lcl = get_mod_mtc(tst_edges, clus_res_Lcl, num_clus_est)
time_Lcl = feat_time + prop_time + ext_time + rfn_time_Lcl
print('Locale EST-K %d MOD %.4f TIME %.4f (%.4f %.4f %.4f %.4f)'
      % (num_clus_est, mod_Lcl, time_Lcl, feat_time, prop_time, ext_time, rfn_time_Lcl))
print()
