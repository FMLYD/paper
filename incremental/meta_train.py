from __future__ import division
from __future__ import print_function

import random
import time
import argparse

import numpy
import numpy as np
import os
from copy import deepcopy

import torch.optim as optim

from data_split import *
from models import *


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dist',default='e')
parser.add_argument('--test_count',default=10,type=int,)
parser.add_argument('--gpu',type=int,default=0)
parser.add_argument('--use_cuda', action='store_true', help='Enable CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--episodes', type=int, default=1000,
                    help='Number of episodes to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--incremental', action='store_true', help='Enable incremental training.')
parser.add_argument('--way', type=int, default=5, help='way.')
parser.add_argument('--shot', type=int, default=5, help='shot.')
parser.add_argument('--qry', type=int, help='k shot for query set', default=20)
parser.add_argument('--dataset', default='Amazon_clothing', help='Dataset:Amazon_clothing/reddit/dblp')
parser.add_argument('--test',action='store_true', required=False,)
parser.add_argument('--adj',action='store_true', required=False,)

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = args.dataset
cache_path = os.path.join("./cache", str(dataset) + ".pkl")
cache = load_object(cache_path)

pretrain_seed = cache["pretrain_seed"]
adj = cache["adj"]
base_adj = cache["pretrain_adj"]
features = cache["features"]
labels = cache["labels"]
degrees = cache["degrees"]
id_by_class = cache["id_by_class"]
novel_id = cache["novel_id"]
base_id = cache["base_id"]
base_train_id = cache["base_train_id"]
base_dev_id = cache["base_dev_id"]
base_test_id = cache["base_test_id"]
novel_train_id  =cache["novel_train_id"]
novel_test_id = cache["novel_test_id"]
torch.autograd.set_detect_anomaly(True)
# Model and optimizer
gpu=args.gpu
device=torch.device(f'cuda:{gpu}')
encoder = GNN_Encoder(nfeat=features.shape[1],nhid=args.hidden,dropout=args.dropout)




optimizer_encoder = optim.Adam(encoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


out_path=f'./out_of_meta_train/dataset_{args.dataset}_way_{args.way}_shot_{args.shot}/{args.model}/{args.name}/lr_{args.lr}_hidden_{args.hidden}/'

text_name=f'output.txt'
if not os.path.exists(out_path):
    os.makedirs(out_path)
if os.path.exists(out_path+text_name):
    with open(out_path+text_name,'w')as f:
        f.write('')
def write_output(s):
    with open(out_path+text_name,'a') as f:
        f.write(s+'\n')
if args.cuda:
    encoder=encoder.to(device)
    features = features.to(device)
    adj = adj.to(device)
    base_adj=base_adj.to(device)
    labels = labels.to(device)
    degrees = degrees.to(device)

def get_base_prototype(id_by_class, curr_adj):

    original_embeddings,embeddings = encoder(features, curr_adj)
    z_dim = embeddings.size()[1]
    base_prototype = torch.zeros((len(id_by_class), z_dim)).to(device)
    original_base_prototype = torch.zeros((len(id_by_class), z_dim)).to(device)
    for cla in list(id_by_class.keys()):
        cla = int(cla)
        if cla  in prototypes.keys():

            base_prototype[cla]=prototypes[cla]
        else:
            base_prototype[cla]=embeddings[id_by_class[cla]].mean(0)
        if args.model=='ours':
            original_base_prototype[cla]=original_embeddings[id_by_class[cla]].mean(0)
    if args.model=='ours':
        return original_base_prototype,base_prototype
    return base_prototype



def incremental_train(curr_adj, base_id_query, novel_id_query, novel_id_support, base_class_selected, novel_class_selected, n_way, k_shot):
    encoder.train()
    original_base_prototype,base_prototype_embeddings = get_base_prototype(id_by_class, curr_adj)
    base_prototype_embeddings = base_prototype_embeddings[base_class_selected]
    for i in range(len(base_class_selected)):
        if base_class_selected[i] not in prototypes.keys():
            prototypes[base_class_selected[i]] =base_prototype_embeddings[i].detach()

    original_base_prototype=original_base_prototype[base_class_selected]
    if args.cuda:
        base_prototype_embeddings = base_prototype_embeddings.to(device)

    optimizer_encoder.zero_grad()
    original_embeddings, embeddings = encoder(features, curr_adj)
    z_dim = embeddings.size()[1]

    # embedding lookup

    novel_support_embeddings = embeddings[novel_id_support]
    novel_support_embeddings = novel_support_embeddings.view([n_way, k_shot, z_dim])
    novel_query_embeddings = embeddings[novel_id_query]
    base_query_embeddings = embeddings[base_id_query]

    # original_embeddings
    original_novel_support_embeddings = original_embeddings[novel_id_support]
    original_novel_support_embeddings = original_novel_support_embeddings.view([n_way, k_shot, z_dim])
    original_novel_query_embeddings = original_embeddings[novel_id_query]
    original_base_query_embeddings = original_embeddings[base_id_query]

    # compute prototype
    novel_prototype_embeddings = novel_support_embeddings.mean(1)
    original_novel_prototype_embeddings = original_novel_support_embeddings.mean(1)

    prototype_embeddings = torch.cat((base_prototype_embeddings, novel_prototype_embeddings), dim=0)
    original_prototype_embeddings = torch.cat((original_base_prototype, original_novel_prototype_embeddings), dim=0)


    # compute loss and acc
    if args.dist == 'e':
        base_dists = euclidean_dist(base_query_embeddings, base_prototype_embeddings)
    elif args.dist == 'c':
        base_dists = cosine_dist(base_query_embeddings, base_prototype_embeddings)

    if args.dist == 'e':
        novel_dists = euclidean_dist(novel_query_embeddings, novel_prototype_embeddings)
    elif args.dist == 'c':
        novel_dists = cosine_dist(novel_query_embeddings, novel_prototype_embeddings)



    query_embeddings = torch.cat((base_query_embeddings, novel_query_embeddings), dim=0)
    if args.dist == 'e':
        dists = euclidean_dist(query_embeddings, prototype_embeddings)
    elif args.dist == 'c':
        dists = cosine_dist(query_embeddings, prototype_embeddings)

    output = F.log_softmax(-dists, dim=1)

    base_labels_new = torch.LongTensor([base_class_selected.index(i) for i in labels[base_id_query]])
    novel_labels_new = torch.LongTensor([novel_class_selected.index(i) for i in labels[novel_id_query]])
    tmp_novel_labels_new = torch.LongTensor([i + len(base_class_selected) for i in novel_labels_new])
    labels_new = torch.cat((base_labels_new, tmp_novel_labels_new))
    del tmp_novel_labels_new

    # Compute attentions


    if args.cuda:
        labels_new = labels_new.to(device)
        # base_labels_new = base_labels_new.cuda()
        #novel_labels_new = novel_labels_new.to(device)

    if args.name=='original' or args.name=='only_hypernet':
       loss_train=NLLLoss(output,labels_new)
    else:
        loss_train =  NLLLoss(output, labels_new)+calculate_distance_variance(prototype_embeddings)


    loss_train_all =loss_train

    # loss_train.backward()
    loss_train_all.backward()
    optimizer_encoder.step()
    
    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()

    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)
    for i in range(len(novel_class_selected)):
        novel_class = novel_class_selected[i]
        prototypes[novel_class] = novel_prototype_embeddings[i].detach()
    return acc_train, f1_train,novel_prototype_embeddings.detach()


def incremental_test(curr_adj, base_id_query, novel_id_query, novel_id_support, base_class_selected, novel_class_selected, n_way, k_shot):
    encoder.eval()

    _,base_prototype_embeddings = get_base_prototype(id_by_class, curr_adj)
    base_prototype_embeddings=base_prototype_embeddings[base_class_selected]
    if args.cuda:
        base_prototype_embeddings = base_prototype_embeddings.to(device)
    _,embeddings = encoder(features, curr_adj)
    z_dim = embeddings.size()[1]

    # embedding lookup
    novel_support_embeddings = embeddings[novel_id_support]
    novel_support_embeddings = novel_support_embeddings.view([n_way, k_shot, z_dim])
    novel_query_embeddings = embeddings[novel_id_query]
    base_query_embeddings = embeddings[base_id_query]

    # node importance


    # compute prototype
    novel_prototype_embeddings=novel_support_embeddings.mean(1)

    prototype_embeddings = torch.cat((base_prototype_embeddings, novel_prototype_embeddings), dim=0)

    # compute loss and acc
    if args.dist == 'e':
        base_dists = euclidean_dist(base_query_embeddings, base_prototype_embeddings)
    elif args.dist == 'c':
        base_dists = cosine_dist(base_query_embeddings, base_prototype_embeddings)

    base_output = F.log_softmax(-base_dists, dim=1)
    if args.dist == 'e':
        novel_dists = euclidean_dist(novel_query_embeddings, novel_prototype_embeddings)
    elif args.dist == 'c':
        novel_dists = cosine_dist(novel_query_embeddings, novel_prototype_embeddings)

    novel_output = F.log_softmax(-novel_dists, dim=1)

    query_embeddings = torch.cat((base_query_embeddings, novel_query_embeddings), dim=0)
    if args.dist == 'e':
        dists = euclidean_dist(query_embeddings, prototype_embeddings)
    elif args.dist == 'c':
        dists = cosine_dist(query_embeddings, prototype_embeddings)

    output = F.log_softmax(-dists, dim=1)

    base_labels_new = torch.LongTensor([base_class_selected.index(i) for i in labels[base_id_query]])
    novel_labels_new = torch.LongTensor([novel_class_selected.index(i) for i in labels[novel_id_query]])
    tmp_novel_labels_new = torch.LongTensor([i + n_way for i in novel_labels_new])
    labels_new = torch.cat((base_labels_new, tmp_novel_labels_new))
    del tmp_novel_labels_new

    if args.cuda:
        labels_new = labels_new.to(device)
    loss_test = NLLLoss(output, labels_new)

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()

        base_output = base_output.cpu().detach()
        base_labels_new = base_labels_new.cpu().detach()

        novel_output = novel_output.cpu().detach()
        novel_labels_new = novel_labels_new.cpu().detach()

    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)

    base_acc_test = accuracy(base_output, base_labels_new)
    base_f1_test = f1(base_output, base_labels_new)

    novel_acc_test = accuracy(novel_output, novel_labels_new)
    novel_f1_test = f1(novel_output, novel_labels_new)

    return acc_test, f1_test, base_acc_test, base_f1_test, novel_acc_test, novel_f1_test

def get_adj(adj, all_base_class_selected, novel_id_support, novel_id_query,
                                          labels,idx):
    incremental_adj = get_incremental_adj(adj.coalesce(), all_base_class_selected, novel_id_support, novel_id_query,
                                          labels)

    
    return incremental_adj
threads = []
prototypes={}
def final_test(base_id,novel_train_id,novel_test_id,n_way, k_shot,m_query):
    meta_test_acc = []
    meta_test_f1 = []

    all_base_class_selected = deepcopy(base_id + novel_train_id)
    novel_class_left = deepcopy(novel_test_id)
    for idx in range(9):


        base_id_query, novel_id_query, novel_id_support, base_class_selected, novel_class_selected = \
            incremental_task_generator(id_by_class, n_way, k_shot, m_query, all_base_class_selected, novel_class_left)


        novel_class_left = list(set(novel_class_left) - set(novel_class_selected))
        
        incremental_adj = get_adj(adj.coalesce().cpu(), [], novel_id_support, np.concatenate([base_id_query,novel_id_query]),
                                  labels.cpu(), idx).to(device)
        if not args.adj:
            
            acc_test, f1_test, base_acc_test, base_f1_test, novel_acc_test, novel_f1_test=incremental_test(incremental_adj, base_id_query, novel_id_query,
                                                  novel_id_support, base_class_selected, novel_class_selected, n_way,
                                                  k_shot)

            all_base_class_selected.extend(novel_class_selected)
            meta_test_acc.append(acc_test)
            meta_test_f1.append(f1_test)

            a = "Session {} Meta test_Accuracy: {}, Meta test_F1: {}".format(idx+1,np.array(meta_test_acc)[-1],
                                                                  np.array(meta_test_f1)[-1])
            print(a)
            write_output(a)
            a = f'meta_novel_test_accuracy:{novel_acc_test},meta_novel_test_f1:{novel_f1_test}'
            print(a)
            write_output(a)
            a = f'meta_base_test_accuracy:{base_acc_test},meta_base_test_f1:{base_f1_test}'
            print(a)
            write_output(a)
        if len(novel_class_left) < n_way:
            break

    return meta_test_acc,meta_test_f1
if __name__ == '__main__':

    n_way = args.way
    k_shot = args.shot
    m_query = args.qry
    meta_test_num = 10
    train_test_num = 10

    # Train model
    t_total = time.time()
    meta_train_acc = []

    all_base_class_selected = deepcopy(base_id)
    novel_class_left = deepcopy(novel_train_id)

    pretrain_test_pool = [task_generator(id_by_class, base_id, n_way, k_shot, m_query) for i in range(train_test_num)]
    if os.path.exists(f'./checkpoints/{args.dataset}.pth'  ) and args.test:

        checkpoint=torch.load(f'./checkpoints/{args.dataset}_{args.model}_{args.name}.pth')
        encoder.load_state_dict(checkpoint['encoder_state_dict'],strict=False)

    else:
        for episode in range(0,args.episodes):
            original_base_prototype = torch.zeros((len(id_by_class), args.hidden)).to(device)
            if args.incremental:
                base_id_query, novel_id_query, novel_id_support, base_class_selected, novel_class_selected = \
                            incremental_task_generator(id_by_class, n_way, k_shot, m_query, all_base_class_selected, novel_class_left)

                acc_train, f1_train,_ = incremental_train(base_adj, base_id_query, novel_id_query,
                                                    novel_id_support, base_class_selected, novel_class_selected, n_way, k_shot)
            all_base_class_selected.extend(novel_class_selected)
            novel_class_left = list(set(novel_class_left) - set(novel_class_selected))
            meta_train_acc.append(acc_train)

            if  episode % 10 == 0:

                # Sampling a pool of tasks for testing
                test_pool = [incremental_task_generator(id_by_class, n_way, k_shot, m_query, all_base_class_selected, novel_train_id) for i
                            in range(meta_test_num)]
                a="-------Episode {}-------".format(episode)
                print(a)
                write_output(a)
                a = "Meta-Train_Accuracy(avg): {}".format(np.array(meta_train_acc).mean(axis=0))
                print(a)
                write_output(a)
                a = "Meta-Train_Accuracy: {}".format(acc_train)
                print(a)
                write_output(a)
                # testing
                meta_test_acc = []
                meta_test_f1 = []
                meta_base_test_acc = []
                meta_base_test_f1 = []
                meta_novel_test_acc = []
                meta_novel_test_f1 = []
                for idx in range(meta_test_num):
                    original_base_prototype = torch.zeros((len(id_by_class), args.hidden)).to(device)

                    base_id_query, novel_id_query, novel_id_support, base_class_selected, novel_class_selected = test_pool[idx]
                    acc_test, f1_test, base_acc_test, base_f1_test, novel_acc_test, novel_f1_test = \
                        incremental_test(base_adj, base_id_query, novel_id_query, novel_id_support,
                                        base_class_selected, novel_class_selected, n_way, k_shot)
                    meta_test_acc.append(acc_test)
                    meta_test_f1.append(f1_test)
                    meta_base_test_acc.append(base_acc_test)
                    meta_base_test_f1.append(base_f1_test)
                    meta_novel_test_acc.append(novel_acc_test)
                    meta_novel_test_f1.append(novel_f1_test)
                a="Meta base test_Accuracy: {}, Meta base test_F1: {}".format(np.array(meta_base_test_acc).mean(axis=0),
                                                                             np.array(meta_base_test_f1).mean(axis=0))
                print(a)
                write_output(a)
                a="Meta novel test_Accuracy: {}, Meta novel test_F1: {}".format(np.array(meta_novel_test_acc).mean(axis=0),
                                                                                np.array(meta_novel_test_f1).mean(axis=0))
                print(a)
                write_output(a)
                a="Meta test_Accuracy: {}, Meta test_F1: {}".format(np.array(meta_test_acc).mean(axis=0),
                                                                        np.array(meta_test_f1).mean(axis=0))
                print(a)
                write_output(a)
            if len(novel_class_left) < n_way:
                prototypes.clear()
                all_base_class_selected = deepcopy(base_id)
                novel_class_left = deepcopy(novel_train_id)
        torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'optimizer_encoder_state_dict': optimizer_encoder.state_dict()
        # 'loss': loss,
    }, f'./checkpoints/{args.dataset}_{args.model}_{args.name}.pth')
    


    acc_list=[[]for i in range(9)]
    f1_list=[[]for i in range(9)]
    test_encoder=deepcopy(encoder).to(device)
    for i in range(args.test_count):
        prototypes.clear()
        #encoder=deepcopy(test_encoder).to(device)
        novel = random.sample(novel_class_left, n_way)
        base_id_query, novel_id_query, novel_id_support, base_class_selected, novel_class_selected = \
            incremental_task_generator(id_by_class, n_way, k_shot, m_query,
                                       all_base_class_selected + list(set(novel_class_left) - set(novel)), novel)

        acc_test, f1_test, base_acc_test, base_f1_test, novel_acc_test, novel_f1_test= incremental_test(base_adj,
                                                        base_id_query,
                                                                                                         novel_id_query,
                                                                                                         novel_id_support,
                                                                                                         base_class_selected,
                                                                                                         novel_class_selected,
                                                                                                         n_way, k_shot)
        a = "Session 0: Meta test_Accuracy: {}, Meta test_F1: {}".format(np.array(acc_test),
                                                                        np.array(f1_test))
        print(a)
        write_output(a)
        acc,f1_=final_test(base_id,novel_train_id,novel_test_id,n_way, k_shot,m_query)
        for idx in range(9):
            acc_list[idx].append(acc[idx])
            f1_list[idx].append(f1_[idx])
        a = "average acc: {}, average test_F1: {}".format(np.array(acc_list).mean(1),
                                                                  np.array(f1_list).mean(1))
        print(a)
        write_output(a)
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
