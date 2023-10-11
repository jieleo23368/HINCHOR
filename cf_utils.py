import copy
import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
import networkx as nx
from sknetwork.embedding import Spectral
from sknetwork.utils import membership_matrix
from sknetwork.hierarchy import Ward, cut_straight
from sknetwork.clustering import Louvain, KMeans, PropagationClustering
import pysbm


def load_t_files(args, logger, adj_train,adj_label,all_x,all_y,node2type):
    print('begin load t file....')
    node_embs_raw=np.loadtxt('data/raw-embedding.txt')
    T_f = get_t(adj_train, args.t, args.k, args.selfloopT)
    three_T_f,three_adj_f=get_three_adj_T_f(adj_label,T_f,all_x,all_y)
    three_T_cf, three_adj_cf= get_CF(adj_train,adj_label,three_adj_f, node_embs_raw, T_f,all_x, args.dist, args.gamma, args.n_workers,node2type)
    return three_T_f,three_T_cf, three_adj_cf

def get_three_adj_T_f(adj_label,T_f,all_x,all_y):
    three_T_f=[]
    three_adj_f=dict()
    nodes_pairs=all_x
    for nodes,label in zip(nodes_pairs,all_y):
        a=nodes[0]
        b=nodes[1]
        c=nodes[2]
        try:
            if T_f[a,b]==T_f[b,c]&T_f[a,b]==1:
                three_T_f.append([a,b,c])
        except:
            pass

        three_adj_f[(a, b, c)] = label
    return three_T_f,three_adj_f

def get_t(adj_train, method, k, selfloop=False):
    adj = copy.deepcopy(adj_train)
    if not selfloop:
        adj.setdiag(0)
        adj.eliminate_zeros()
    if method == 'louvain':
        T = louvain(adj)
    elif method == 'spectral_clustering':
        T = spectral_clustering(adj, k)
    elif method == 'kcore':
        T = kcore(adj)
    elif method == 'hierarchy':
        T = ward_hierarchy(adj, k)
    elif method == 'sbm':
        T = SBM(adj, k)
    return T

def SBM(adj, k):
    nx_g = nx.from_scipy_sparse_matrix(adj)
    standard_partition = pysbm.NxPartition(graph=nx_g, number_of_blocks=k)
    rep = standard_partition.get_representation()
    labels = np.asarray([v for k, v in sorted(rep.items(), key=lambda item: item[0])])
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T

def ward_hierarchy(adj, k):
    ward = Ward()
    dendrogram = ward.fit_transform(adj)
    labels = cut_straight(dendrogram, k)
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T


def louvain(adj):
    louvain = Louvain()
    labels = louvain.fit_transform(adj)
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T


def spectral_clustering(adj, k):
    kmeans = KMeans(n_clusters = k, embedding_method=Spectral(256))
    labels = kmeans.fit_transform(adj)
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T

def kcore(adj):
    G = nx.from_scipy_sparse_matrix(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    labels = np.array(list(nx.algorithms.core.core_number(G).values()))-1
    mem_mat = membership_matrix(labels)
    T = (mem_mat @ mem_mat.T).astype(int)
    return T



def get_CF(adj,adj_label,three_adj_f, node_embs, T_f,all_x, dist, thresh, n_workers,node2type,verbose = True):
    if dist == 'cosine':
        # cosine similarity (flipped to use as a distance measure)
        embs = normalize(node_embs, norm='l1', axis=1)
        simi_mat = embs @ embs.T
        simi_mat = 1 - simi_mat
    elif dist == 'euclidean':
        # Euclidean distance
        simi_mat = cdist(node_embs, node_embs, 'euclidean')
    thresh = np.percentile(simi_mat, thresh)
    # give selfloop largest distance
    np.fill_diagonal(simi_mat, np.max(simi_mat)+1)
    # nearest neighbor nodes index for each node
    node_nns = np.argsort(simi_mat, axis=1)
    # find nearest CF for each x
    node_pairs = all_x
    print('This step may be slow...')


    three_T_cf = []
    three_adj_cf = dict()
    c = 0
    for id, node_pair in enumerate(node_pairs):
        node_pair = list(node_pair)
        old_community = 0
        new_community = 0
        c += 1
        if verbose and c % 2000 == 0:
            print(f'{c} / {len(node_pairs)} done')
        temp = {node_pair[0]: node2type[node_pair[0]], node_pair[1]: node2type[node_pair[1]],
                node_pair[2]: node2type[node_pair[2]]}
        sort_temp = sorted(temp.items(), key=lambda temp: temp[1])
        a = sort_temp[0][0]
        src = sort_temp[1][0]
        b = sort_temp[2][0]
        type_a = node2type[a]
        type_b = node2type[b]

        nns_a = node_nns[a]
        nns_b = node_nns[b]
        i, j = 0, 0
        if T_f[src, a] == T_f[a, b] & T_f[src, a] == 1:
            old_community = 1
        else:
            old_community = 0
        same_type_a = []
        same_type_b = []
        for node in nns_a:
            if node2type[node] == type_a:
                same_type_a.append(node)
        for node in nns_b:
            if node2type[node] == type_b:
                same_type_b.append(node)

        while i < len(same_type_a) - 1 and j < len(same_type_b) - 1:
            if simi_mat[a, same_type_a[i]] + simi_mat[b, same_type_b[j]] > 2 * thresh:
                if old_community == 1:
                    three_T_cf.append(list(node_pair))
                three_adj_cf[tuple(node_pair)] = three_adj_f[tuple(node_pair)]
                break
            if T_f[src, same_type_a[i]] == T_f[same_type_a[i], same_type_b[j]] & T_f[src, same_type_a[i]] == 1:
                new_community = 1
            else:
                new_community = 0

            if old_community != new_community:
                if new_community == 1:
                    three_T_cf.append(list(node_pair))
                pattern = judge_pattern([src, same_type_a[i], same_type_b[j]], adj_label)
                three_adj_cf[tuple(node_pair)] = pattern
                break
            if simi_mat[a, same_type_a[i + 1]] < simi_mat[b, same_type_b[j + 1]]:
                i += 1
            else:
                j += 1
            if i == len(same_type_a) - 1 or j == len(same_type_b) - 1:
                if old_community == 1:
                    three_T_cf.append(list(node_pair))
                three_adj_cf[tuple(node_pair)] = three_adj_f[tuple(node_pair)]
    return three_T_cf, three_adj_cf


def judge_pattern(nodes,adj):
    a=nodes[0]
    b=nodes[1]
    c=nodes[2]
    if adj[a,b]==adj[a,c]==adj[b,c]==0:
        pattern=0
    elif adj[a,b]==1 and adj[a,c]==adj[b,c]==0:
        pattern=1
    elif adj[a,c]==1 and adj[a,b]==adj[b,c]==0:
        pattern=2
    elif adj[b,c]==1 and adj[a,b]==adj[a,c]==0:
        pattern=3
    elif adj[a,b]==adj[a,c]==1 and adj[b,c]==0:
        pattern=4
    elif adj[a,b]==adj[b,c]==1 and adj[a,c]==0:
        pattern=5
    elif adj[a,c]==adj[b,c]==1 and adj[a,b]==0:
        pattern=6
    elif adj[a,c]==adj[a,b]==adj[b,c]==1 :
        pattern=7
    return pattern
