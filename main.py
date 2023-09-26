# coding=utf-8
import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import networkx as nx
import numpy as np
from collections import Counter

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)


from utils import *
from models import *
from cf_utils import load_t_files


def get_args():
    parser = argparse.ArgumentParser(description='HINCHOR')
    parser.add_argument('--seed', type=int, default=42, help='fix random seed if needed')
    parser.add_argument('--dataset', type=str, default='dblp')
    parser.add_argument('--datapath', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--embraw', type=str, default='metapath2vec')   
    parser.add_argument('--t', type=str, default='sbm', help='choice of the treatment function')
    parser.add_argument('--k', type=int, default=30, help='parameter for the treatment function (if needed)')
    parser.add_argument('--selfloopT', default=False, action='store_true', help='whether to add selfloop when getting T')
    parser.add_argument('--dist', type=str, default='euclidean', help='distant metric used when finding nearest neighbors')
    parser.add_argument('--gamma', type=float, default=50.0, help='maximum distance thresold for finding nearest neighbors')
    parser.add_argument('--dec', type=str, default='concatenate', choices=['innerproduct','hadamard','mlp'], help='choice of decoder')
    parser.add_argument('--verbose', type=int, default=1, help='whether to print per-epoch logs')
    parser.add_argument('--gpu', type=int, default=-1, help='-2 for CPU, -1 for default GPU, >=0 for specific GPU')
    parser.add_argument('--n_workers', type=int, default=1, help='number of CPU processes for finding counterfactual links in the first run')
    parser.add_argument('--dim_h', type=int, default=128)
    parser.add_argument('--dim_z', type=int, default=8)
    parser.add_argument('--dim_feat', type=int, default=128)
    parser.add_argument('--walk_num', type=int, default=300)
    parser.add_argument('--walk_length', type=int, default=10)
    parser.add_argument('--p', type=int, default=1,help='Probability of repeated visits to the vertex just visited')
    parser.add_argument('--q', type=int, default=2,help='Control whether the randwalk is outward or inward')
    parser.add_argument('--patience', type=int, default=20, help='number of patience steps for early stopping')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--l2reg', type=float, default=5e-6)
    parser.add_argument('--lr_scheduler', type=str, default='zigzag', choices=['sgdr', 'cos', 'zigzag', 'none'], help='lr scheduler')
    parser.add_argument('--metric', type=str, default='auc', choices=['Macro_F1', 'Micro_F1', 'NMI','auc','Precision','Rceall'], help='main evaluation metric')
    parser.add_argument('--name', type=str, default='debug', help='name for this run for logging')
    args = parser.parse_args()
    args.argv = sys.argv

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    args.device = torch.device('cuda:0' if args.gpu >= -1 else 'cpu')

    return args




class MyDataset(Dataset):
    def __init__(self, data, labels1,labels2,labels3,labels4,labels5):
        self.data = data
        self.labels1 = labels1
        self.labels2= labels2
        self.labels3= labels3
        self.labels4= labels4
        self.labels5= labels5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y1 = self.labels1[index]
        y2 = self.labels2[index]
        y3 = self.labels3[index]
        y4 = self.labels4[index]
        y5 = self.labels5[index]
        return x, y1,y2,y3,y4,y5

class testDataset(Dataset):
    def __init__(self, data1, data2,data3):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[index]
        x3 = self.data3[index]
        return x1, x2,x3

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False



def precess_data(args,logger):
    print('begin training...')
    #load raw data
    adj_label, adj_train, all_x, all_y, node2type = load_data(args, logger)
    if not os.path.exists('data/three_T_f.txt'):
        three_T_f_index, three_T_cf_index, three_adj_cf_index = load_t_files(args, logger, adj_train, adj_label, all_x,
                                                                             all_y, node2type)
        # three_T_cf_index = three_T_cf_index
        three_T_f = []
        three_T_cf = []
        three_adj_cf = []
        for x in all_x:
            if x in three_T_cf_index:
                three_T_cf.append(1)
            else:
                three_T_cf.append(0)
            if x in three_T_f_index:
                three_T_f.append(1)
            else:
                three_T_f.append(0)
            three_adj_cf.append(three_adj_cf_index[tuple(x)])

        np.savetxt('data/three_T_f.txt', three_T_f, fmt='%d', delimiter='\t')
        np.savetxt('data/three_T_cf.txt', three_T_cf, fmt='%d', delimiter='\t')
        np.savetxt('data/three_adj_cf.txt', three_adj_cf, fmt='%d', delimiter='\t')
    else:
        three_T_f = list(np.loadtxt('data/three_T_f.txt', dtype=int))
        three_T_cf = list(np.loadtxt('data/three_T_cf.txt', dtype=int))
        three_adj_cf = list(np.loadtxt('data/three_adj_cf.txt', dtype=int))

     
    #split data into training set, validation set, test set
    if os.path.exists('train_val_test/train_x.npy'):
        train_x = np.load('train_val_test/train_x.npy')
        train_y = np.load('train_val_test/train_y.npy')
        train_three_T_f = np.load('train_val_test/train_three_T_f.npy')
        train_three_T_cf = np.load('train_val_test/train_three_T_cf.npy')
        train_three_adj_cf = np.load('train_val_test/train_three_adj_cf.npy')

        val_x = np.load('train_val_test/val_x.npy')
        val_y = np.load('train_val_test/val_y.npy')
        val_three_T_f = np.load('train_val_test/val_three_T_f.npy')

        test_x = np.load('train_val_test/test_x.npy')
        test_y = np.load('train_val_test/test_y.npy')
        test_three_T_f = np.load('train_val_test/test_three_T_f.npy')

    else:
        train_rate = 0.6
        val_rate = 0.2
        counts = dict(Counter(all_y))
        train_data_index = [[] for i in range(8)]
        val_data_index = [[] for i in range(8)]
        test_data_index = [[] for i in range(8)]
        for id, type in enumerate(all_y):
            if len(train_data_index[type]) < train_rate * counts[type]:
                train_data_index[type].append(id)
                continue
            if len(val_data_index[type]) < val_rate * counts[type]:
                val_data_index[type].append(id)
                continue
            test_data_index[type].append(id)

        # training
        train_x = [all_x[j] for i in train_data_index for j in i]
        train_y = [all_y[j] for i in train_data_index for j in i]
        train_three_T_f = [three_T_f[j] for i in train_data_index for j in i]
        train_three_T_cf = [three_T_cf[j] for i in train_data_index for j in i]
        train_three_adj_cf = [three_adj_cf[j] for i in train_data_index for j in i]
        np.save('train_val_test/train_x.npy', train_x)
        np.save('train_val_test/train_y.npy', train_y)
        np.save('train_val_test/train_three_T_f.npy', train_three_T_f)
        np.save('train_val_test/train_three_T_cf.npy', train_three_T_cf)
        np.save('train_val_test/train_three_adj_cf.npy', train_three_adj_cf)
        # validation
        val_x = [all_x[j] for i in val_data_index for j in i]
        val_y = [all_y[j] for i in val_data_index for j in i]
        val_three_T_f = [three_T_f[j] for i in val_data_index for j in i]
        np.save('train_val_test/val_x.npy', val_x)
        np.save('train_val_test/val_y.npy', val_y)
        np.save('train_val_test/val_three_T_f.npy', val_three_T_f)
        # test
        test_x = [all_x[j] for i in test_data_index for j in i]
        test_y = [all_y[j] for i in test_data_index for j in i]
        test_three_T_f = [three_T_f[j] for i in test_data_index for j in i]
        np.save('train_val_test/test_x.npy', test_x)
        np.save('train_val_test/test_y.npy', test_y)
        np.save('train_val_test/test_three_T_f.npy', test_three_T_f)

    


    # move everything to device
    device = args.device
    train_x = torch.LongTensor(train_x).to(device)
    train_y = torch.LongTensor(train_y).to(device)
    val_x = torch.LongTensor(val_x).to(device)
    val_y = torch.LongTensor(val_y).to(device)
    test_x = torch.LongTensor(test_x).to(device)
    test_y = torch.LongTensor(test_y).to(device)
    train_three_T_f = torch.LongTensor(train_three_T_f).to(device)
    train_three_T_cf = torch.LongTensor(train_three_T_cf).to(device)
    train_three_adj_cf = torch.LongTensor(train_three_adj_cf).to(device)



    G = nx.Graph()
    all_nodes = node2type.keys()
    G.add_nodes_from(all_nodes)
    rows, cols = adj_train.nonzero()
    pairs = [i for i in zip(rows, cols)]
    G.add_edges_from(pairs)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    p=args.p
    q=args.q
    dim_feat = args.dim_feat
    walk_num = args.walk_num
    walk_length = args.walk_length
    alias_nodes, alias_edges = preprocess_transition_probs(G, p, q)

    #calculate d
    if os.path.exists('paths_d/train_subgraph_paths_d.npy'):
        train_subgraph_paths_d = np.load('paths_d/train_subgraph_paths_d.npy')
    else:
        train_subgraph_paths_d = []
        for id, motif in enumerate(train_x[:]):
            print('being initialized for the {}th train_x...'.format(id))
            subgraph_paths_d = get_init_feature(G, motif, node2type, dim_feat, walk_num, walk_length, alias_nodes,
                                                alias_edges)
            train_subgraph_paths_d.append(subgraph_paths_d)
        train_subgraph_paths_d = np.array(train_subgraph_paths_d)
        np.save('paths_d/train_subgraph_paths_d.npy', train_subgraph_paths_d)

    if os.path.exists('paths_d/val_subgraph_paths_d.npy'):
        val_subgraph_paths_d = np.load('paths_d/val_subgraph_paths_d.npy')
    else:
        val_subgraph_paths_d = []
        for id, motif in enumerate(val_x[:]):
            print('being initialized for the {}th val_x...'.format(id))
            subgraph_paths_d = get_init_feature(G, motif, node2type, dim_feat, walk_num, walk_length, alias_nodes,
                                                alias_edges)
            val_subgraph_paths_d.append(subgraph_paths_d)
        val_subgraph_paths_d = np.array(val_subgraph_paths_d)
        np.save('paths_d/val_subgraph_paths_d.npy', val_subgraph_paths_d)

    if os.path.exists('paths_d/test_subgraph_paths_d.npy'):
        test_subgraph_paths_d = np.load('paths_d/test_subgraph_paths_d.npy')
    else:
        test_subgraph_paths_d = []
        for id, motif in enumerate(test_x[:]):
            print('being initialized for the {}th test_x'.format(id))
            subgraph_paths_d = get_init_feature(G, motif, node2type, dim_feat, walk_num, walk_length, alias_nodes,
                                                alias_edges)
            test_subgraph_paths_d.append(subgraph_paths_d)
        test_subgraph_paths_d = np.array(test_subgraph_paths_d)
        np.save('paths_d/test_subgraph_paths_d.npy', test_subgraph_paths_d)

    # print(train_subgraph_paths_d.shape)
    # print(val_subgraph_paths_d.shape)
    # print(test_subgraph_paths_d.shape)
    return node2type, train_x, train_y, train_three_T_f, train_three_T_cf, train_three_adj_cf, train_subgraph_paths_d,\
    val_three_T_f,val_y, val_subgraph_paths_d, test_three_T_f,test_y, test_subgraph_paths_d




def train(args,node2type,train_x,train_y,train_three_T_f,train_three_T_cf,train_three_adj_cf,train_subgraph_paths_d,val_three_T_f,val_y,val_subgraph_paths_d,test_three_T_f,test_y,test_subgraph_paths_d, logger):
    dim_feat = args.dim_feat
    walk_num = args.walk_num
    walk_length = args.walk_length
    print('begin model....')
    model = HINCHOR(args, dim_feat, args.dim_h, args.dim_z, walk_num, walk_length, node2type, args.dropout,args.dec)
    model = model.to(args.device)

    optim = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.l2reg)
    optims = MultipleOptimizer(args.lr_scheduler, optim)
    loss_funcution = nn.CrossEntropyLoss()
    best_val_res = 0.0
    cnt_wait = 0
    total_loss = 0.0

    for epoch in range(args.epochs):
        total_examples = 0
        total_loss = 0
        for motifs,labels_f_batch,T_f_batch,T_cf_batch,labels_cf_batch,subgraph_paths_d in DataLoader(MyDataset(train_x[:],train_y[:],train_three_T_f[:],train_three_T_cf[:],train_three_adj_cf[:],train_subgraph_paths_d), args.batch_size, shuffle=True):
            model.train()
            lr=optims.update_lr(args.lr)
            optims.zero_grad()

            z, logits_f,logits_cf= model(subgraph_paths_d, T_f_batch, T_cf_batch)
            loss_f =loss_funcution(logits_f, labels_f_batch)
            loss_cf = loss_funcution(logits_cf, labels_cf_batch)
            loss=loss_f+loss_cf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optims.step()

            total_loss += loss.item() * motifs.shape[0]
            total_examples += motifs.shape[0]
        total_loss /= total_examples

        #evaluation
        model.eval()
        with torch.no_grad():
            z = model.encoder(val_subgraph_paths_d)
            logits_val= model.decoder(z, val_three_T_f).detach().cpu()
            logits_val = torch.tensor(logits_val)
            val_res = eval_ep_batched(logits_val, val_y)
            if val_res[args.metric] >= best_val_res:
                cnt_wait = 0
                best_val_res = val_res[args.metric]
                torch.save(model.state_dict(), './model/best_model.pt')
            else:
                cnt_wait += 1
            if args.verbose:
                logger.info('Epoch {} Loss: {:.4f}  val_auc: {:.4f}'.format(
                    epoch+1, total_loss,  val_res[args.metric]))
            if cnt_wait >= args.patience:
                if args.verbose:
                    print('Early stopping!')
                break

    #Load Best Model
    model.load_state_dict(torch.load('./model/best_model.pt'))
    model.eval()
    logits_test=[]
    label_test=[]
    for subgraph_paths_d,label, three_T_f in DataLoader(testDataset(test_subgraph_paths_d,test_y[:], test_three_T_f[:]),args.test_batch_size,shuffle=True):
        z = model.encoder(subgraph_paths_d)
        temp_logits_test = model.decoder(z, three_T_f).detach().cpu()
        logits_test.extend(temp_logits_test)
        label_test.extend(label)
    logits_test = torch.stack(logits_test, dim=0)
    label_test = torch.stack(label_test, dim=0)
    test_res = eval_ep_batched(logits_test, label_test)
    logger.info('final test Loss: {:.4f} test auc: {:.4f} nmi: {:.4f} macro_f1: {:.4f} micro_f1: {:.4f} acc: {:.4f} precision: {:.4f} recall: {:.4f}'.format(
        total_loss, test_res[args.metric],test_res['nmi'],test_res['macro_f1'],test_res['micro_f1'],test_res['acc'],test_res['precision'],test_res['recall']))
    return test_res



if __name__ == "__main__":
    args = get_args()
    setup_seed(args.seed)
    log_name = f'{args.log_dir}/{args.name}_{args.dataset}_{args.t}_{args.embraw}_{time.strftime("%m-%d_%H-%M")}'
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(log_name)
    logger.info(f'Input argument vector: {args.argv[1:]}')
    logger.info(f'args: {args}')
    node2type, train_x, train_y, train_three_T_f, train_three_T_cf, train_three_adj_cf, train_subgraph_paths_d,\
    val_three_T_f,val_y, val_subgraph_paths_d, test_three_T_f,test_y, test_subgraph_paths_d=precess_data(args,logger)
    res = train(args,node2type,train_x,train_y,train_three_T_f,train_three_T_cf,train_three_adj_cf,train_subgraph_paths_d,\
         val_three_T_f,val_y,val_subgraph_paths_d,test_three_T_f,test_y,test_subgraph_paths_d,logger)
    print('End...')
