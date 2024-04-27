import random

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_geometric.nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GNN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid,name='none',model='none',dropout=0):
        super(GNN_Encoder, self).__init__()
        
        self.gcn=torch_geometric.nn.GCN(nfeat, nhid*2, 2,nhid,dropout=dropout)

        self.adjuster=Hypernet(3*nhid,nhid,nhid)
        self.down=nn.Linear(nfeat,2*nhid)
        self.adjuster2=Hypernet(4*nhid,nhid,nhid)
        self.model=model
        self.name=name
        self.nhid=nhid

        
    def our_forward(self,x,adj):

        adj = adj.coalesce().indices()
        hid_original = self.gcn(x, adj)
        res_original=hid_original
        if self.name=='original' or self.name=='only_loss':
            return hid_original,hid_original
        x = self.down(x)
        hid_original=F.relu(hid_original)
        
        mm, bb = self.adjuster(torch.cat([x, hid_original], dim=1))
        hid1 = torch.matmul(mm, F.relu(hid_original).unsqueeze(2))
        hid1 = hid1.squeeze()
        hid = hid1 + bb+hid_original
        hid=F.relu(hid)
       
        return res_original, hid
    def zap_forward(self,x,adj):
        adj = adj.coalesce().indices()

        return None,nn.Linear(self.nhid,self.nhid).to(x.device)(self.gcn(x, adj))
    def gnn_forward(self,x,adj):
        adj=adj.coalesce().indices()
        return None,self.gcn(x,adj)
    def forward(self, x, adj):
        if self.model=='ours':
            return self.our_forward(x,adj)
        elif self.model=='zap':
            return self.zap_forward(x,adj)
        else:
            return self.gnn_forward(x,adj)

class Hypernet(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(Hypernet, self).__init__()
        self.down1 = nn.Linear(nfeat, nhid1 * nhid2 )
  
        self.bias = nn.Linear(nfeat, nhid2)
        self.nhid1 = nhid2
        self.nhid2 = nhid1
    
    def forward(self, x):
        mm = self.down1(x)
        mm = mm.view(-1, self.nhid1, self.nhid2)
        bb = self.bias(x)
        return mm, bb






