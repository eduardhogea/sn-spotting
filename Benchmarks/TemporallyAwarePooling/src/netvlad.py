
import torch
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.neighbors import NearestNeighbors
import numpy as np



from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import math


def _safe_batched_mm(lhs: th.Tensor, rhs: th.Tensor) -> th.Tensor:
    # Torch/cuBLAS on this environment can fail on strided batched GEMM
    # with CUBLAS_STATUS_INVALID_VALUE for valid [B, M, K] x [B, K, N] inputs.
    # Fall back to per-sample mm on CUDA when batch>1.
    if lhs.dim() == 3 and rhs.dim() == 3 and lhs.shape[0] == rhs.shape[0]:
        if lhs.is_cuda and lhs.shape[0] > 1:
            return th.stack([th.mm(lhs[i], rhs[i]) for i in range(lhs.shape[0])], dim=0)
        return th.bmm(lhs, rhs)
    return th.matmul(lhs, rhs)


class NetVLAD(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter((1/math.sqrt(feature_size))
                *th.randn(feature_size, cluster_size))
        self.clusters2 = nn.Parameter((1/math.sqrt(feature_size))
                *th.randn(1, feature_size, cluster_size))

        self.add_batch_norm = add_batch_norm
        self.out_dim = cluster_size*feature_size

    def forward(self,x):
        # x [BS, T, D]
        max_sample = x.size()[1]

        # LOUPE
        if self.add_batch_norm: # normalization along feature dimension
            x = F.normalize(x, p=2, dim=2)

        x = x.reshape(-1,self.feature_size)
        assignment = th.matmul(x,self.clusters) 

        assignment = F.softmax(assignment,dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        a_sum = th.sum(assignment,-2,keepdim=True)
        a = a_sum*self.clusters2

        assignment = assignment.transpose(1,2)

        x = x.view(-1, max_sample, self.feature_size)
        vlad = _safe_batched_mm(assignment, x)
        vlad = vlad.transpose(1,2)
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)
        
        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size*self.feature_size)
        vlad = F.normalize(vlad)

        return vlad


class NetRVLAD(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetRVLAD, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter((1/math.sqrt(feature_size))
                *th.randn(feature_size, cluster_size))
        # self.clusters2 = nn.Parameter((1/math.sqrt(feature_size))
        #         *th.randn(1, feature_size, cluster_size))
        # self.clusters = nn.Parameter(torch.rand(1,feature_size, cluster_size))
        # self.clusters2 = nn.Parameter(torch.rand(1,feature_size, cluster_size))

        self.add_batch_norm = add_batch_norm
        # self.batch_norm = nn.BatchNorm1d(cluster_size)
        self.out_dim = cluster_size*feature_size
        #  (+ 128 params?)
    def forward(self,x):
        max_sample = x.size()[1]

        # LOUPE
        if self.add_batch_norm: # normalization along feature dimension
            x = F.normalize(x, p=2, dim=2)

        x = x.reshape(-1,self.feature_size)
        assignment = th.matmul(x,self.clusters)

        assignment = F.softmax(assignment,dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        # a_sum = th.sum(assignment,-2,keepdim=True)
        # a = a_sum*self.clusters2

        assignment = assignment.transpose(1,2)

        x = x.view(-1, max_sample, self.feature_size)
        rvlad = _safe_batched_mm(assignment, x)
        rvlad = rvlad.transpose(-1,1)

        # vlad = vlad.transpose(1,2)
        # vlad = vlad - a

        # L2 intra norm
        rvlad = F.normalize(rvlad)
        
        # flattening + L2 norm
        rvlad = rvlad.reshape(-1, self.cluster_size*self.feature_size)
        rvlad = F.normalize(rvlad)

        return rvlad


if __name__ == "__main__":
    vlad = NetVLAD(cluster_size=64, feature_size=512)

    feat_in = torch.rand((3,120,512))
    print("in", feat_in.shape)
    feat_out = vlad(feat_in)
    print("out", feat_out.shape)
    print(512*64)
