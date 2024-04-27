from __future__ import division
from __future__ import print_function

import time
import argparse
from copy import deepcopy

import numpy as np
import os

import torch
import torch.optim as optim

from data_split import *
from models import *

# Training settings
parser = argparse.ArgumentParser()


parser.add_argument('--cache',action='store_true',help='')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--dataset', default='Amazon_clothing', help='Dataset:Amazon_clothing/reddit/dblp')


args = parser.parse_args()




random.seed(args.seed)
torch.manual_seed(args.seed)

if not os.path.exists("./cache"):
    os.makedirs("./cache")
cache_path = os.path.join("./cache", (str(args.dataset) + ".pkl"))
dataset = args.dataset

# Load data
if args.cache:
    cache=load_object(cache_path)
    adj, features, labels, degrees, id_by_class, base_id, novel_id, num_nodes, num_all_nodes=cache["adj"], \
                                                                                             cache["features"], cache["labels"], cache["degrees"],\
                                                                                             cache["id_by_class"], cache["base_id"], cache["novel_id"],cache["num_nodes"],cache["num_all_nodes"]
    novel_train_id,novel_test_id=cache["novel_train_id"],cache["novel_test_id"]
    pretrain_id=cache["pretrain_id"]
    pretrain_idx, predev_idx, pretest_idx, base_train_label, base_dev_label, base_test_label, \
    base_train_id, base_dev_id, base_test_id=cache["pretrain_idx"], cache["predev_idx"],\
    cache["pretest_idx"], cache["base_train_label"], cache["base_dev_label"],cache["base_test_label"],\
    cache["base_train_id"],cache["base_dev_id"],cache['base_test_id']
    pretrain_adj=cache["pretrain_adj"]
    print(novel_id)
    print(pretrain_id)
else:
    adj, features, labels, degrees, id_by_class, base_id, novel_id, num_nodes, num_all_nodes = load_raw_data(dataset)
    novel_train_id, novel_test_id = split_novel_data(novel_id, dataset)
    pretrain_id = base_id + novel_train_id
    pretrain_idx, predev_idx, pretest_idx, base_train_label, base_dev_label, base_test_label,\
                                base_train_id, base_dev_id, base_test_id = split_base_data(pretrain_id, id_by_class, labels)
    pretrain_adj = get_base_adj(adj, pretrain_id, labels)

    cache = {"pretrain_seed": args.seed, "adj": adj, "features": features, "labels": labels, "pretrain_adj": pretrain_adj,
         "degrees": degrees, "id_by_class": id_by_class, "base_id": base_id,
         "novel_id": novel_id, "num_nodes": num_nodes, "num_all_nodes": num_all_nodes,
         "base_train_id": base_train_id, "base_dev_id": base_dev_id, "base_test_id": base_test_id,
         "novel_train_id": novel_train_id, "novel_test_id": novel_test_id,"pretrain_id": pretrain_id,"pretrain_idx": pretrain_idx,"predev_idx": predev_idx,
             "pretest_idx": pretest_idx,"base_train_label":base_train_label,"base_dev_label":base_dev_label,
             "base_test_label":base_test_label}


    save_object(cache, cache_path)
del cache

