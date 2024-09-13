import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.pointnet import PointNetfeat
from src.modules.transformer import Encoder_TRANSFORMER

class FeatureFollower(nn.Module):
    def __init__(self, options):
        super(FeatureFollower, self).__init__()

        self.nhidden_kypt = options.nhidden_kypt

        # PointNet
        self.npoint_feat = options.npoint_feat
        self.point_feat = PointNetfeat(self.npoint_feat)

        # Transformer Encoder
        self.nframes = options.Ttot
        self.transformer_encoder = Encoder_TRANSFORMER(options.Ttot, self.npoint_feat, self.nhidden_kypt)


    def forward(self, input_pcd, padding_mask1d=None):
        B, T, N, _ = input_pcd.shape
        
        feat_point, _, _ = self.point_feat(input_pcd.view(B * T, -1, 3).transpose(1, 2))
        feat_point = feat_point.view(B, T, -1)

        # mapped_feat = self.latent_mapper(feat_point)
        mapped_feat, _ = self.transformer_encoder(feat_point, padding_mask1d=padding_mask1d)

        return mapped_feat