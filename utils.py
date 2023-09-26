import math
import logging
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score,normalized_mutual_info_score,f1_score,accuracy_score,precision_score,recall_score
from sklearn.preprocessing import label_binarize



def load_data():
    #original_graph
    graph=nx.Graph()
    with open('data/original_graph.txt','r',encoding='utf-8')as f:
        origal_edges=f.read()
    origal_edges = origal_edges.split('\n')[:-1]
    for edge in origal_edges:
        edge = edge.split(' ')
        graph.add_edge(int(edge[0]),int(edge[1]))
    nodelist=sorted(graph.nodes)
    adj_label=nx.adjacency_matrix(graph,nodelist=nodelist)

    ##train_graph
    new_graph=nx.Graph()
    with open('data/graph.txt','r',encoding='utf-8')as f:
        edges=f.read()
    edges = edges.split('\n')[:-1]
    for edge in edges:
        edge = edge.split('\t')
        new_graph.add_edge(int(edge[0]),int(edge[1]))
    adj_train=nx.adjacency_matrix(new_graph,nodelist=nodelist)


    all_x=[]
    with open('data/train.txt','r',encoding='utf-8')as f:
        motifs=f.read()
    motifs = motifs.split('\n')[:-1]
    for motif in motifs:
        motif = motif.split('\t')
        all_x.append([int(x) for x in motif])


    with open('data/label.txt','r',encoding='utf-8')as f:
        all_y=f.read()
    all_y = all_y.split('\n')[:-1]
    all_y=[int(x) for x in all_y]


    node2type_dic=dict()
    with open('data/nodes2type.txt', 'r', encoding='utf-8')as f:
        node2type = f.read()
    node2type = node2type.split('\n')[:-1]
    for pair in node2type:
        node,type=pair.split('\t')
        node2type_dic[int(node)]=int(type)

    return adj_label,adj_train,all_x,all_y,node2type_dic


def eval_ep_batched(logits, labels):
    logits = F.softmax(logits)
    logits = logits.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    labels_1d = labels
    labels = label_binarize(labels, classes=[0, 1, 2, 3, 4, 5, 6, 7])
    roc_auc = {}
    auc = 0.0
    for i in range(8):
        try:
            roc_auc[i] = roc_auc_score(labels[:, i], logits[:, i])
        except:
            roc_auc[i] = 0
        auc += roc_auc[i]
    auc = auc / 8
    # NMI:
    logits_1d = np.argmax(logits, 1)
    result_NMI = normalized_mutual_info_score(labels_1d, logits_1d)
    # F1-score
    macro_f1 = f1_score(labels_1d, logits_1d, average='macro')
    micro_f1 = f1_score(labels_1d, logits_1d, average='micro')
    # acc
    acc = accuracy_score(labels_1d, logits_1d)
    # precison
    precison = precision_score(labels_1d, logits_1d, average='macro')
    # recall
    recall = recall_score(labels_1d, logits_1d, average='macro')
    results = {'auc': auc, 'nmi': result_NMI, 'macro_f1': macro_f1, 'micro_f1': micro_f1, 'acc': acc,
               'precision': precison, 'recall': recall}

    return results

class MultipleOptimizer():
    """ a class that wraps multiple optimizers """
    def __init__(self, lr_scheduler, *op):
        self.optimizers = op
        self.steps = 0
        self.reset_count = 0
        self.next_start_step = 10
        self.multi_factor = 2
        self.total_epoch = 0
        if lr_scheduler == 'sgdr':
            self.update_lr = self.update_lr_SGDR
        elif lr_scheduler == 'cos':
            self.update_lr = self.update_lr_cosine
        elif lr_scheduler == 'zigzag':
            self.update_lr = self.update_lr_zigzag
        elif lr_scheduler == 'none':
            self.update_lr = self.no_update

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
    def no_update(self, base_lr):
        return base_lr

    def update_lr_SGDR(self, base_lr):
        end_lr = 1e-3 # 0.001
        total_T = self.total_epoch + 1
        if total_T >= self.next_start_step:
            self.steps = 0
            self.next_start_step *= self.multi_factor
        cur_T = self.steps + 1
        lr = end_lr + 1/2 * (base_lr - end_lr) * (1.0 + math.cos(math.pi*cur_T/total_T))
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.steps += 1
        self.total_epoch += 1
        return lr

    def update_lr_zigzag(self, base_lr):
        warmup_steps = 50
        annealing_steps = 20
        end_lr = 1e-4
        if self.steps < warmup_steps:
            lr = base_lr * (self.steps+1) / warmup_steps
        elif self.steps < warmup_steps+annealing_steps:
            step = self.steps - warmup_steps
            q = (annealing_steps - step) / annealing_steps
            lr = base_lr * q + end_lr * (1 - q)
        else:
            self.steps = self.steps - warmup_steps - annealing_steps
            lr = end_lr
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.steps += 1
        return lr

    def update_lr_cosine(self, base_lr):
        """ update the learning rate of all params according to warmup and cosine annealing """
        # 400, 1e-3
        warmup_steps = 10
        annealing_steps = 500
        end_lr = 1e-3
        if self.steps < warmup_steps:
            lr = base_lr * (self.steps+1) / warmup_steps
        elif self.steps < warmup_steps+annealing_steps:
            step = self.steps - warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / annealing_steps))
            lr = base_lr * q + end_lr * (1 - q)
        else:
            # lr = base_lr * 0.001
            self.steps = self.steps - warmup_steps - annealing_steps
            lr = end_lr
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.steps += 1
        return lr

def get_logger(name):
    """ create a nice logger """
    logger = logging.getLogger(name)
    # clear handlers if they were created in other runs
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # create console handler add add to logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler add add to logger when name is not None
    if name is not None:
        fh = logging.FileHandler(f'{name}.log')
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    return logger



