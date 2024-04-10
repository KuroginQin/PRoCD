import numpy as np

# ========================================
# Feature Extraction

def get_rand_proj_mat(data_dim, red_dim, rand_seed=None):
    '''
    Function to get random matrix for Gaussian random projection
    :param data_dim: original data dimensionality
    :param red_dim: reduced dimensionality
    :param rand_seed: random seed
    :return: random matrix
    '''
    # ===================
    np.random.seed(rand_seed)
    rand_mat = np.random.normal(0, 1.0/red_dim, (data_dim, red_dim))

    return rand_mat

def get_sp_mod_feat(edges, degs):
    '''
    Function to get the sparse modularity features
    :param edges: edge list
    :param degs: degree w.r.t. each node
    :return:
    '''
    # ====================
    num_edges = len(edges)*2
    # ==========
    idxs = []
    vals = []
    for (src, dst) in edges:
        v = 1 - degs[src]*degs[dst]/num_edges
        idxs.append((src, dst))
        vals.append(v)
        idxs.append((dst, src))
        vals.append(v)

    return idxs, vals

def get_sp_GCN_sup(edges, num_nodes):
    '''
    Function to get the sparse GCN support (i.e., normalized adjacency matrix w/ self-edges)
    :param edges: edge list
    :param num_nodes: number of nodes
    :return:
    '''
    # ====================
    degs = [1 for _ in range(num_nodes)] # Node degrees containing self-edges
    for (src, dst) in edges:
        degs[src] += 1
        degs[dst] += 1
    # ==========
    idxs = []
    vals = []
    for (src, dst) in edges:
        # ==========
        v = 1/(np.sqrt(degs[src]) * np.sqrt(degs[dst]))
        # ==========
        idxs.append((src, dst))
        vals.append(v)
        idxs.append((dst, src))
        vals.append(v)
    for idx in range(num_nodes): # Self-edges
        idxs.append((idx, idx))
        vals.append(1/degs[idx])

    return idxs, vals

# ========================================
# Quality Evaluation

def get_mod_mtc(edges, clus_res, num_clus):
    '''
    Function to get the modularity metric w.r.t. a clustering result
    :param edges: edge list (undirected & 0-base node indices)
    :param clus_res: clustering result
    :param num_clus: number of clusters
    :return:
    '''
    # ====================
    mod_mtc = 0.0
    # ==========
    num_edges = len(edges)*2 # Number of edges
    clus_in_edges = [0 for _ in range(num_clus)] # Number of intra-cluster edges w.r.t. each cluster
    clus_edges = [0 for _ in range(num_clus)] # Number of induced edges w.r.t. each cluster
    # ==========
    for (src, dst) in edges:
        # ==========
        src_lbl = clus_res[src]
        dst_lbl = clus_res[dst]
        # ==========
        if src_lbl==dst_lbl:
            clus_in_edges[src_lbl] += 2
        clus_edges[src_lbl] += 1
        clus_edges[dst_lbl] += 1
    # ==========
    for lbl in range(num_clus):
        L = clus_in_edges[lbl]/num_edges
        R = clus_edges[lbl]/num_edges
        mod_mtc += (L - R*R)

    return mod_mtc