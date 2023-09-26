import numpy as np
from numpy import random
import torch
import torch.nn as nn
import torch.nn.functional as F




class HINCHOR(nn.Module):
    def __init__(self,args,dim_feat, dim_h, dim_z,walk_num,walk_length,node2type, dropout, dec='hadamard'):
        super(HINCHOR, self).__init__()
        self.args=args
        self.device=args.device
        self.encoder = Encoder(args,dim_feat, dim_h,walk_num,walk_length,node2type, dropout)
        self.decoder = Decoder(args,dec, dim_in, dim_z)
        self.init_params()
        self.dim_h=dim_h


    def forward(self,subgraph_paths_d, T_f_batch, T_cf_batch):
        z = self.encoder(subgraph_paths_d)
        logits_f = self.decoder(z,T_f_batch)
        logits_cf = self.decoder(z,T_cf_batch)
        return z, logits_f,logits_cf

    def init_params(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()


class Encoder(nn.Module):
    def __init__(self, args,dim_feat, dim_h, walk_num,walk_length,node2type,dropout):
        super(Encoder, self).__init__()
        self.act = nn.ELU()
        self.dropout = dropout
        self.node2type=node2type
        self.dim_feat=dim_feat
        self.dim_h=dim_h
        self.device=args.device
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.lstm = nn.LSTM(3*(self.walk_length)+1,self.dim_feat,1)
        self.all_nodes=node2type.keys()

    def forward(self,subgraphs_paths_d):
        nodes_emb_temp = []
        for subgraph_paths_d in subgraphs_paths_d:
            three_nodes_emb_temp = []
            for paths_d in subgraph_paths_d:
                paths_d = np.array(paths_d).reshape(self.walk_length, self.walk_num, 3*self.walk_length+1)
                paths_d = torch.tensor(paths_d, dtype=torch.float).to(device)
                paths_emb, (h,c) = self.lstm(paths_d)
                paths_emb = paths_emb[-1].squeeze()
                node_emb =torch.mean(paths_emb,dim=0)
                three_nodes_emb_temp.append(node_emb)
            three_nodes_emb=torch.cat(three_nodes_emb_temp,dim=-1)
            nodes_emb_temp.append(three_nodes_emb)
        features=torch.stack(nodes_emb_temp,dim=0)

        print ('features.shape:')
        print (features.shape)
        return features


def get_init_feature(G,subgraph,node2type,dim_feat,walk_num,walk_length,alias_nodes,alias_edges):
    print('begin biased random walk.....')
    subgraph_paths=[]
    for node in subgraph:
        node=node.item()
        node_paths=[]
        for i in range(walk_num):
            path=random_walk(G,walk_length,node,alias_nodes,alias_edges)
            node_paths.append(path)
        subgraph_paths.append(node_paths)
    subgraph_paths_d=get_d(subgraph_paths,node2type,dim_feat,walk_num,walk_length)
    return subgraph_paths_d


def random_walk(G, walk_length, start_node,alias_nodes,alias_edges):
    '''
    Simulate a random walk starting from start node.
    '''
    G = G
    alias_nodes =alias_nodes
    alias_edges =alias_edges

    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = sorted(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
            else:
                prev = walk[-2]
                next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                           alias_edges[(prev, cur)][1])]
                walk.append(next)
        else:
            break
    if len(walk)<walk_length:
        walk=[start_node for i in range(walk_length) ]
    return walk

def get_alias_edge(G,p,q,src, dst):
    unnormalized_probs = []
    for dst_nbr in sorted(G.neighbors(dst)):
        if dst_nbr == src:
            unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
        elif G.has_edge(dst_nbr, src):
            unnormalized_probs.append(G[dst][dst_nbr]['weight'])
        else:
            unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
    return alias_setup(normalized_probs)

def preprocess_transition_probs(G,p,q,is_directed=False):
    '''
    Preprocessing of transition probabilities for guiding the random walks.
    '''
    G = G
    is_directed = is_directed

    alias_nodes = {}
    for node in G.nodes():
        unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        alias_nodes[node] = alias_setup(normalized_probs)
    alias_edges = {}
    triads = {}
    if is_directed:
        for edge in G.edges():
            alias_edges[edge] =get_alias_edge(G,p,q,edge[0], edge[1])
    else:
        for edge in G.edges():
            alias_edges[edge] =get_alias_edge(G,p,q,edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = get_alias_edge(G,p,q,edge[1], edge[0])
    return alias_nodes,alias_edges

def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []

    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def get_d(subgraph_paths,node2type):

    subgraph_paths=np.array(subgraph_paths)
    print(type(subgraph_paths))
    print(subgraph_paths.shape)
    i_paths=subgraph_paths[0]
    src_i=i_paths[0][0]
    j_paths=subgraph_paths[1]
    src_j = j_paths[0][0]
    k_paths=subgraph_paths[2]
    src_k = k_paths[0][0]

    nodes_list = dict()
    nodes=set(np.array(subgraph_paths).reshape(1,1,-1).squeeze())
    for node in nodes:
        if node in nodes_list:
            continue
        d=[]
        i_paths_T=np.array(i_paths).T
        j_paths_T=np.array(j_paths).T
        k_paths_T=np.array(k_paths).T
        i_count=[0]+[list(i).count(node)  for i in i_paths_T[1:]]
        j_count=[0]+[list(i).count(node)  for i in j_paths_T[1:]]
        k_count=[0]+[list(i).count(node)  for i in k_paths_T[1:]]
        d.extend(i_count)
        d.extend(j_count)
        d.extend(k_count)
        d.append(node2type[node])
        nodes_list[node]=d

    subgraph_paths_d=[]
    for paths in subgraph_paths:
        paths_d=[]
        for path in paths:
            path_d=[]
            for node in path:
                path_d.append(nodes_list[node])
            paths_d.append(path_d)
        subgraph_paths_d.append(paths_d)
    return subgraph_paths_d




class Decoder(nn.Module):
    # hadamard,256,256
    def __init__(self,args, dec, dim_in,dim_z=8):
        super(Decoder, self).__init__()
        if dec=='concatenate':
            dim_in=3*dim_in+1
        self.mlp_out = nn.Sequential(
            nn.Linear(dim_in, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, dim_z, bias=False),
        )
        self.device=args.device

    def forward(self, z, T):
        print('begin decoder.....')
        z = z.to(torch.float32).to(self.device)
        if self.dec == 'concatenate':
            T=torch.tensor(T).to(self.device)
            h=torch.cat((z,T.view(-1, 1)), dim=1).to(torch.float32)
        h = self.mlp_out(h).squeeze()
        return h

    def reset_parameters(self):
        for lin in self.mlp_out:
            try:
                #lin.reset_parameters()
                nn.init.kaiming_normal_(lin.weight, mode='fan_in', nonlinearity='relu')
            except:
                continue
