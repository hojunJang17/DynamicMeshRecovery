import math
import scipy as sp
import scipy.linalg

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    

class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, mask, lengths):
        time = mask * 1 / (lengths[..., None] - 1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T

        x = x + time[..., None]
        return self.dropout(x)


class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, njoints, nfeats, out_dim,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=8, dropout=0.1, 
                 ablation=None, activation='gelu', **kargs):
        super().__init__()

        self.njoints = njoints
        self.nfeats = nfeats

        self.latent_dim = latent_dim
        self.out_dim = out_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation

        # self.input_feats = self.njoints * self.nfeats
        self.input_feats = self.nfeats

        self.mu_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.out_dim),
            nn.LeakyReLU()
        )
        self.sigma_layer = nn.Linear(self.latent_dim, self.out_dim)

        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

    def forward(self, x, mask=None, padding_mask1d = None):
        '''
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)
        '''
        bs, nframes, nfeats = x.shape
        x = x.permute((1, 0, 2))  # (nframes, bs, nfeats)
        # embedding of the skeleton
        x = self.skelEmbedding(x)
        
        # add positional encoding
        x = self.sequence_pos_encoder(x)

        # transformer layers
        if mask is not None:
            final = self.seqTransEncoder(x, src_key_padding_mask=~mask)
        else:
            if padding_mask1d is not None:
                padding_mask = padding_mask1d.repeat(bs,1).to(x.device)
                final = self.seqTransEncoder(x, src_key_padding_mask=padding_mask)
            else:
                final = self.seqTransEncoder(x)

        final = final.transpose(0, 1)  # (bs, nframes, nfeats)
        
        # extract mu and var
        mu = self.mu_layer(final)
        var = self.sigma_layer(final)

        return mu, var

class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, state_dim,
                 latent_dim=256, ff_size=1024, num_layers=4, 
                 num_heads=8, dropout=0.1, activation='gelu',
                 **kargs):
        super().__init__()

        self.state_dim = state_dim

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation

        self.input_feats = self.state_dim

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        
        self.finalLayer = nn.Linear(self.latent_dim, self.input_feats)
    
    def forward(self, z_pose, feat_trans, padding_mask1d=None):
        SAMPLE_NUM, bs, nframes, _ = z_pose.shape

        z = torch.cat([z_pose, feat_trans], dim=-1)  # (SAMPLE_NUM, bs, nframes, latent_dim)
        z = z.view(SAMPLE_NUM * bs, nframes, -1).permute((1, 0, 2))
        timequeries = torch.zeros(nframes, SAMPLE_NUM * bs, self.latent_dim, device=z_pose.device)
        timequeries = self.sequence_pos_encoder(timequeries)

        if padding_mask1d is not None:
            padding_mask = padding_mask1d.repeat(SAMPLE_NUM*bs,1).to(z.device)             
            output = self.seqTransDecoder(tgt=timequeries, memory=z, memory_key_padding_mask=padding_mask)

        else:
            output = self.seqTransDecoder(tgt=timequeries, memory=z)

        output = output.transpose(0, 1)
        output = output.view(SAMPLE_NUM, bs, nframes, -1)

        output = self.finalLayer(output)

        return output
