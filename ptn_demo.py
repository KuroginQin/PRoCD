# Demo code of offline pre-training

from modules.PRoCD import *
import torch.optim as optim

import pickle
import random
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

# ====================
data_name = 'dblp'
feat_dims = [64, 64, 64] # Layer conf of feat red unit
feat_dim = feat_dims[0]
emb_dim = feat_dims[-1]
num_GNN_lyr = 4 # Number of GNN layers - L_GNN
num_MLP_lyr_tmp = 2 # Number of MLP layers for temp param - L_BC

BCE_param = 0.1 # alpha
mod_rsl = 10 # lambda
num_eph = 100 # Number of pre-training epochs - n_P
learn_rate = 1e-4 # Learning rate - eta
num_add_pairs = 10000 # Number of sampled node pairs - n_S

trn_num_snap = 1000 # Number of pre-training graphs - T
drop_rate = 0.2

# ====================
# Read synthetic pre-training graphs (w/ ground-truth)
pkl_file = open('data/ptn_edges_list.pickle', 'rb')
trn_edges_list = pickle.load(pkl_file)
pkl_file.close()
# ==========
pkl_file = open('data/ptn_gnd_list.pickle', 'rb')
trn_gnd_list = pickle.load(pkl_file)
pkl_file.close()

# ====================
# Precompute data stat
num_nodes_list = []
gnd_mem_list = []
degs_list = []
src_idxs_list = []
dst_idxs_list = []
for t in range(trn_num_snap):
    # ==========
    edges = trn_edges_list[t]
    gnd = trn_gnd_list[t] # Ground-truth
    # ==========
    num_nodes = len(gnd)
    num_nodes_list.append(num_nodes)
    # ==========
    if np.min(gnd) == 0:
        num_clus = np.max(gnd) + 1
    else:
        num_clus = np.max(gnd)
    gnd_mem = np.zeros((num_nodes, num_clus))
    for i in range(num_nodes):
        r = gnd[i]
        gnd_mem[i, r] = 1.0
    gnd_mem_list.append(gnd_mem)
    # ==========
    degs = [0 for _ in range(num_nodes)]
    src_idxs = []
    dst_idxs = []
    for (src, dst) in edges:
        # ==========
        degs[src] += 1
        degs[dst] += 1
        # ==========
        src_idxs.append(src)
        dst_idxs.append(dst)
    degs_list.append(degs)
    src_idxs_list.append(src_idxs)
    dst_idxs_list.append(dst_idxs)

# ====================
# Define the model
mdl = PRoCD_MDL(feat_dims, num_GNN_lyr, num_MLP_lyr_tmp, drop_rate).to(device)
# ==========
# Define the optimizer
opt = optim.Adam(mdl.parameters(), lr=learn_rate)

# ====================
for eph in range(num_eph):
    # ====================
    mdl.train()
    loss_acc = 0.0
    for t in range(trn_num_snap):
        # ====================
        num_nodes = num_nodes_list[t]
        edges = trn_edges_list[t] # Edge list
        num_edges = len(edges)
        degs = degs_list[t]
        degs_tnr = torch.FloatTensor(degs).to(device)
        gnd_mem = gnd_mem_list[t]
        gnd_mem_tnr = torch.FloatTensor(gnd_mem).to(device)
        pair_ind_gnd = torch.mm(gnd_mem_tnr, gnd_mem_tnr.t()) # Ground-truth pairwise constraint
        src_idxs = src_idxs_list[t]
        dst_idxs = dst_idxs_list[t]
        # ===========
        # Extract input feature - Gaussian random projection
        idxs, vals = get_sp_mod_feat(edges, degs)
        idxs_tnr = torch.LongTensor(idxs).to(device)
        vals_tnr = torch.FloatTensor(vals).to(device)
        sp_mod_tnr = torch.sparse.FloatTensor(idxs_tnr.t(), vals_tnr,
                                              torch.Size([num_nodes, num_nodes])).to(device)
        rand_mat = get_rand_proj_mat(num_nodes, feat_dim, rand_seed=rand_seed_gbl)
        rand_mat_tnr = torch.FloatTensor(rand_mat).to(device)
        red_feat_tnr = torch.spmm(sp_mod_tnr, rand_mat_tnr)
        # ==========
        # Get GNN support
        idxs, vals = get_sp_GCN_sup(edges, num_nodes)
        idxs_tnr = torch.LongTensor(idxs).to(device)
        vals_tnr = torch.FloatTensor(vals).to(device)
        sup_tnr = torch.sparse.FloatTensor(idxs_tnr.t(), vals_tnr,
                                           torch.Size([num_nodes, num_nodes])).to(device)

        # ====================
        emb_tnr, lft_tmp_tnr, rgt_tmp_tnr = mdl(red_feat_tnr, sup_tnr)
        edge_ind_est = get_edge_ind_est(emb_tnr, lft_tmp_tnr, rgt_tmp_tnr, src_idxs, dst_idxs)
        pair_ind_est = get_node_pair_ind_est(emb_tnr, lft_tmp_tnr, rgt_tmp_tnr)
        BCE_loss = get_BCE_loss(pair_ind_gnd, pair_ind_est)
        mod_loss = get_mod_max_loss(edge_ind_est, pair_ind_est, degs_tnr, num_edges, resolution=mod_rsl)
        loss = BCE_param*BCE_loss + mod_loss
        # ==========
        loss_acc += loss.item()
    print('Epoch %d loss %f' % (eph+1, loss_acc))

# ====================
torch.save(mdl, 'chpt/PRoCD_%s_%d.pkl' % (data_name, num_eph))
