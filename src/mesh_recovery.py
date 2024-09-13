import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import random

import sys
sys.path.append('src/dataset')
from human_body_prior.body_model.body_model import BodyModel
from manopth.manolayer import ManoLayer

from src.utils.geo_utils import *
from src.utils.mask_utils import generate_padding_mask1d
from src.modules.pointnet import PointNetfeat
from src.modules.transformer import Encoder_TRANSFORMER, Decoder_TRANSFORMER
from src.feature_follower import FeatureFollower

import pytorch_lightning as pl



class MeshRecovery(pl.LightningModule):
    def __init__(self, options):
        super().__init__()
        
        self.lr = options.lrate
        self.firstdecay = options.firstdecay
        self.seconddecay = options.seconddecay

        self.nkeypoints = options.nkeypoints
        self.nlatent_kypt = options.nlatent_kypt
        self.nhidden_kypt = options.nhidden_kypt
        self.input_dim = options.input_dim

        # weights of loss terms
        self.aux_weight = options.aux_weight
        self.kl_pose_weight = options.kl_pose_weight
        self.pose_recon_weight = options.pose_recon_weight
        self.joint_recon_weight = options.joint_recon_weight
        self.vol_fit_weight = options.vol_fit_weight
        self.shape_weight = options.shape_weight
        self.vertex_weight = options.vertex_weight

        self.log_gif_num = options.log_gif_num
        self.logger_path = './output/%s/%s' % (options.training_id, options.exp_name)
        self.pretrained_mode = options.pretrained_mode
        self.fps = options.fps
        self.dataset = options.dataset

        self.SAMPLE_NUM = 1

        state_dim = options.state_dim * 2

        # PointNet
        self.npoint_feat = options.npoint_feat
        self.point_feat = PointNetfeat(self.npoint_feat)

        # MK experiment
        self.Ttot = options.Ttot
        self.br_initial = options.br_initial
        self.maximum_nblock = int(self.Ttot * self.br_initial)
        self.maximum_nblock_validation = int(self.Ttot * self.br_initial)

        # Transformer Encoder
        self.transformer_encoder = Encoder_TRANSFORMER(self.Ttot, self.npoint_feat, self.nhidden_kypt)

        # post / prior network
        self.extract_post_dist = nn.Sequential(
            nn.Linear(in_features=self.nhidden_kypt + state_dim, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=self.nlatent_kypt * 2)
        )
        self.extract_prior_dist = nn.Sequential(
            nn.Linear(in_features=self.nhidden_kypt, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=self.nlatent_kypt * 2)
        )

        # translation decoder
        self.trans_decoder = nn.Sequential(
            nn.Linear(in_features=self.nhidden_kypt, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=3)
        )

        # pose decoder
        self.pose_decoder = Decoder_TRANSFORMER(state_dim, latent_dim=self.nlatent_kypt + self.nhidden_kypt)

        
        # auxiliary task
        self.trans_to_pose = nn.Sequential(
            nn.Linear(in_features=self.nhidden_kypt, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=state_dim)
        )

        if self.dataset.startswith('human'):
            bm_smpl_fname = 'data/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'
            self.body_model = BodyModel(bm_fname=bm_smpl_fname, num_betas=options.shape_num)
        elif self.dataset.startswith('hand'):
            self.hand_model = ManoLayer(mano_root='data/models', use_pca=False, ncomps=45, flat_hand_mean=False, center_idx=9)

        self.feature_weight = options.feature_weight
        self.feature_follower = FeatureFollower(options)

        self.shape_estimator = nn.Sequential(
            nn.Linear(in_features=self.Ttot * self.nhidden_kypt, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=options.shape_num)
        )
    
    def configure_optimizers(self):
        if self.pretrained_mode == 0:
            for child in self.feature_follower.modules():
                for param in child.parameters():
                    param.requires_grad = False
        elif self.pretrained_mode == 1:
            for child in self.modules():
                for param in child.parameters():
                    param.requires_grad = False
            for child in self.feature_follower.modules():
                for param in child.parameters():
                    param.requires_grad = True
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

        return [optimizer]
    

    def training_step(self, batch, batch_nb):

        batch_ids, input_pcd, joints, body_pose, body_shape, trans, partial_pcd = batch
        input_pcd = input_pcd.float()
        joints = joints.float()
        body_pose = body_pose.float()
        body_shape = body_shape.float()
        trans = trans.float()
        partial_pcd = partial_pcd.float()

        B, T, P = body_pose.shape

        nblock = random.randint(0, self.maximum_nblock)

        if nblock == 0:
            padding_mask1d = None
        else:
            padding_mask1d = generate_padding_mask1d(nblock, self.Ttot)
        

        if self.pretrained_mode == 0:
            body_pose_6d = axis_angle_to_rotation_6d(body_pose.reshape(B, T, -1, 3)).view(B, T, -1)

            # PointNet feature extraction
            feat_point, _, _ = self.point_feat(input_pcd.view(B * T, -1, 3).transpose(1, 2))  # (B * T, 3, N) --> (B * T, N_feat)
            feat_point = feat_point.view(B, T, -1)  # (B, T, N_feat)
            
            # feature encoding using Transformer encoder
            feat_trans, _ = self.transformer_encoder(feat_point, padding_mask1d=padding_mask1d)  # (B, T, N_hidden)

            # auxiliary pose estimation
            aux_pose_6d = self.trans_to_pose(feat_trans)  # (B, T, P * 2)
            aux_pose_loss = (aux_pose_6d - body_pose_6d).pow(2).sum(dim=-1).mean()

            # shape parameter estimation
            best_shape = self.shape_estimator(feat_trans.view(B, -1))  # (B, shape_num)
            best_shape = best_shape[:, None]  # (B, 1, shape_num)
            best_shape = best_shape.repeat_interleave(T, 1)  # (B, T, shape_num)
            best_shape_flat = best_shape.view(B * T, -1)

            # prior distribution
            params_prior = self.extract_prior_dist(feat_trans)
            prior_mean, prior_std = torch.chunk(params_prior, 2, dim=-1)
            prior_std = F.softplus(prior_std) + 1e-4
            z_pose_prior_dist = Normal(prior_mean, prior_std)

            # posterior distribution
            params_post = self.extract_post_dist(torch.cat([feat_trans, body_pose_6d], dim=-1))
            post_mean, post_std = torch.chunk(params_post, 2, dim=-1)
            post_std = F.softplus(post_std) + 1e-4
            z_pose_post_dist = Normal(post_mean, post_std)

            # latent sampling
            z_pose_sampled = z_pose_post_dist.rsample(sample_shape=(self.SAMPLE_NUM, ))

            # Transformer decoder
            feat_trans_dec = feat_trans.expand(self.SAMPLE_NUM, -1, -1, -1)

            pose_sampled = self.pose_decoder(z_pose_sampled, feat_trans_dec, padding_mask1d=padding_mask1d)
            pose_distance = (body_pose_6d[None] - pose_sampled).pow(2).sum((2, 3))
            min_sample_idx = pose_distance.argmin(dim=0)
            batch_idx = torch.arange(0, B, dtype=torch.int64, device=body_pose.device)

            best_pose_6d = pose_sampled[min_sample_idx, batch_idx]

            # root translation estimation
            best_trans = self.trans_decoder(feat_trans)


            # generating smpl/mano body
            best_pose_axis = rotation_6d_to_axis_angle(best_pose_6d.view(B, T, -1, 6)).view(B, T, -1)
            best_pose_flat = best_pose_axis.view(B * T, -1)
            best_trans_flat = best_trans.view(B * T, -1)
            best_shape_flat = best_shape.view(B * T, -1)

            body_pose_flat = body_pose.view(B * T, -1)
            trans_flat = trans.view(B * T, -1)
            body_shape_flat = body_shape.view(B * T, -1)

            if self.dataset.startswith('human'):
                best_body = self.body_model(root_orient=best_pose_flat[:, :3],
                                            pose_body=best_pose_flat[:, 3:66],
                                            trans=best_trans_flat,
                                            betas=best_shape_flat)
                best_keypoint = best_body.Jtr.view(B, T, -1, 3)
                best_vertices = best_body.v.view(B, T, -1, 3)

                gt_body = self.body_model(root_orient=body_pose_flat[:, :3],
                                          pose_body=body_pose_flat[:, 3:66],
                                          trans=trans_flat,
                                          betas=body_shape_flat)
                gt_keypoint = gt_body.Jtr.view(B, T, -1, 3)
                gt_vertices = gt_body.v.view(B, T, -1, 3)

            elif self.dataset.startswith('hand'):
                best_vertices, best_keypoint = self.hand_model(best_pose_flat, best_shape_flat, best_trans_flat)
                best_keypoint = best_keypoint.view(B, T, -1, 3) * 10
                best_keypoint = torch.stack([best_keypoint[..., 2], best_keypoint[..., 1], -best_keypoint[..., 0]], -1)
                best_vertices = best_vertices.view(B, T, -1, 3) * 10
                best_vertices = torch.stack([best_vertices[..., 2], best_vertices[..., 1], -best_vertices[..., 0]], -1)
                
                gt_vertices, gt_keypoint = self.hand_model(body_pose_flat, body_shape_flat, trans_flat)
                gt_keypoint = gt_keypoint.view(B, T, -1, 3) * 10
                gt_keypoint = torch.stack([gt_keypoint[..., 2], gt_keypoint[..., 1], -gt_keypoint[..., 0]], -1)
                gt_vertices = gt_vertices.view(B, T, -1, 3) * 10
                gt_vertices = torch.stack([gt_vertices[..., 2], gt_vertices[..., 1], -gt_vertices[..., 0]], -1)
            
            
            # kl divergence loss
            kl_pose = kl_divergence(z_pose_post_dist, z_pose_prior_dist).mean()

            # pose reconstruction loss
            pose_recon_loss = (best_pose_6d - body_pose_6d).pow(2).sum(dim=-1).mean()

            # joint reconstruction loss
            joint_recon_loss = (best_keypoint - joints).pow(2).sum(dim=(-2, -1)).mean()

            # volume fitting loss
            dist = (input_pcd[:, :, :, None] - best_keypoint[:, :, None]).pow(2)  # (B, T, N, K, 3)
            dist = dist.sum(dim=-1)  # (B, T, N, K)
            dist = dist.min(dim=-1).values  # (B, T, N)
            vol_fit_loss = dist.mean()

            # shape loss
            shape_loss = (best_shape - body_shape).pow(2).sum(-1).mean()

            # vertex displacement loss
            vertex_loss = (best_vertices - gt_vertices).pow(2).sum(dim=(-2, -1)).mean()

            loss = 0.0
            loss += aux_pose_loss * self.aux_weight
            loss += kl_pose * self.kl_pose_weight
            loss += pose_recon_loss * self.pose_recon_weight
            loss += joint_recon_loss * self.joint_recon_weight
            loss += vol_fit_loss * self.vol_fit_weight
            loss += shape_loss * self.shape_weight
            loss += vertex_loss * self.vertex_weight

            return {'loss': loss,
                    'auxiliary_pose': aux_pose_loss,
                    'kl_pose': kl_pose,
                    'pose_recon': pose_recon_loss,
                    'joint_recon': joint_recon_loss,
                    'vol_fit': vol_fit_loss,
                    'shape': shape_loss,
                    'vertex': vertex_loss}
        
        elif self.pretrained_mode == 1:
            body_pose_6d = axis_angle_to_rotation_6d(body_pose.reshape(B, T, -1, 3)).view(B, T, -1)

            # PointNet feature extraction
            feat_point, _, _ = self.point_feat(input_pcd.view(B * T, -1, 3).transpose(1, 2))
            feat_point = feat_point.view(B, T, -1)  # (B, T, N_feat)

            # feature encoding using Transformer encoder
            feat_trans, _ = self.transformer_encoder(feat_point, padding_mask1d=padding_mask1d)  # (B, T, N_hidden)

            # feature encoding from the feature follower
            partial_pcd_feat = self.feature_follower(partial_pcd, padding_mask1d)  # (B, T, N_hidden)

            # auxiliary pose estimation
            aux_pose_6d = self.trans_to_pose(partial_pcd_feat)  # (B, T, P * 2)
            aux_pose_loss = (aux_pose_6d - body_pose_6d).pow(2).sum(dim=-1).mean()

            # shape parameter estimation
            part_shape = self.shape_estimator(partial_pcd_feat.view(B, -1))  # (B, shape_num)
            part_shape = part_shape[:, None]  # (B, 1, shape_num)
            part_shape = part_shape.repeat_interleave(T, 1)  # (B, T, shape_num)
            
            # prior distribution
            params_prior = self.extract_prior_dist(partial_pcd_feat)
            prior_mean, prior_std = torch.chunk(params_prior, 2, dim=-1)
            prior_std = F.softplus(prior_std) + 1e-4
            z_pose_prior_dist = Normal(prior_mean, prior_std)

            # latent sampling
            z_pose_sampled = z_pose_prior_dist.rsample(sample_shape=(self.SAMPLE_NUM, ))

            # Transformer decoder
            feat_trans_dec = partial_pcd_feat.expand(self.SAMPLE_NUM, -1, -1, -1)

            pose_sampled = self.pose_decoder(z_pose_sampled, feat_trans_dec, padding_mask1d=padding_mask1d)
            pose_distance = (body_pose_6d[None] - pose_sampled).pow(2).sum((2, 3))
            min_sample_idx = pose_distance.argmin(dim=0)
            batch_idx = torch.arange(0, B, dtype=torch.int64, device=body_pose.device)

            part_pose_6d = pose_sampled[min_sample_idx, batch_idx]

            # root translation estimation
            part_trans = self.trans_decoder(partial_pcd_feat)


            # generating smpl/mano body
            part_pose_axis = rotation_6d_to_axis_angle(part_pose_6d.view(B, T, -1, 6)).view(B, T, -1)
            part_pose_flat = part_pose_axis.view(B * T, -1)
            part_trans_flat = part_trans.view(B * T, -1)
            part_shape_flat = part_shape.view(B * T, -1)

            body_pose_flat = body_pose.view(B * T, -1)
            trans_flat = trans.view(B * T, -1)
            body_shape_flat = body_shape.view(B * T, -1)

            if self.dataset.startswith('human'):
                part_body = self.body_model(root_orient=part_pose_flat[:, :3],
                                            pose_body=part_pose_flat[:, 3:66],
                                            trans=part_trans_flat,
                                            betas=part_shape_flat)
                part_keypoint = part_body.Jtr.view(B, T, -1, 3)
                part_vertices = part_body.v.view(B, T, -1, 3)

                gt_body = self.body_model(root_orient=body_pose_flat[:, :3],
                                            pose_body=body_pose_flat[:, 3:66],
                                            trans=trans_flat,
                                            betas=body_shape_flat)
                gt_keypoint = gt_body.Jtr.view(B, T, -1, 3)
                gt_vertices = gt_body.v.view(B, T, -1, 3)

            elif self.dataset.startswith('hand'):
                part_vertices, part_keypoint = self.hand_model(part_pose_flat, part_shape_flat, part_trans_flat)
                part_keypoint = part_keypoint.view(B, T, -1, 3) * 10
                part_keypoint = torch.stack([part_keypoint[..., 2], part_keypoint[..., 1], -part_keypoint[..., 0]], -1)
                part_vertices = part_vertices.view(B, T, -1, 3) * 10
                part_vertices = torch.stack([part_vertices[..., 2], part_vertices[..., 1], -part_vertices[..., 0]], -1)
                
                gt_vertices, gt_keypoint = self.hand_model(body_pose_flat, body_shape_flat, trans_flat)
                gt_keypoint = gt_keypoint.view(B, T, -1, 3) * 10
                gt_keypoint = torch.stack([gt_keypoint[..., 2], gt_keypoint[..., 1], -gt_keypoint[..., 0]], -1)
                gt_vertices = gt_vertices.view(B, T, -1, 3) * 10
                gt_vertices = torch.stack([gt_vertices[..., 2], gt_vertices[..., 1], -gt_vertices[..., 0]], -1)
            

            # feature following loss
            feature_loss = (partial_pcd_feat - feat_trans).pow(2).sum(-1).mean()

            # pose reconstruction loss
            pose_recon_loss = (part_pose_6d - body_pose_6d).pow(2).sum(-1).mean()

            # joint reconstruction loss
            joint_recon_loss = (part_keypoint - joints).pow(2).sum((-2, -1)).mean()

            # shape loss
            shape_loss = (part_shape - body_shape).pow(2).sum(-1).mean()

            # vertex displacement loss
            vertex_loss = (part_vertices - gt_vertices).pow(2).sum((-2, -1)).mean()

            loss = 0.0
            loss += aux_pose_loss * self.aux_weight
            loss += feature_loss * self.feature_weight
            loss += pose_recon_loss * self.pose_recon_weight
            loss += joint_recon_loss * self.joint_recon_weight
            loss += shape_loss * self.shape_weight
            loss += vertex_loss * self.vertex_weight

            return {'loss': loss,
                    'auxiliary_pose': aux_pose_loss,
                    'feature': feature_loss,
                    'pose_recon': pose_recon_loss,
                    'joint_recon': joint_recon_loss,
                    'shape': shape_loss,
                    'vertex': vertex_loss}
        
    
    def training_epoch_end(self, outputs):
        if self.pretrained_mode == 0:
            aux_pose = []
            kl_pose = []
            pose_recon = []
            joint_recon = []
            vol_fit = []
            shape = []
            vertex = []

            for output in outputs:
                aux_pose.append(output['auxiliary_pose'].clone().detach().cpu())
                kl_pose.append(output['kl_pose'].clone().detach().cpu())
                pose_recon.append(output['pose_recon'].clone().detach().cpu())
                joint_recon.append(output['joint_recon'].clone().detach().cpu())
                vol_fit.append(output['vol_fit'].clone().detach().cpu())
                shape.append(output['shape'].clone().detach().cpu())
                vertex.append(output['vertex'].clone().detach().cpu())

            aux_pose = torch.stack(aux_pose, 0).mean()
            kl_pose = torch.stack(kl_pose, 0).mean()
            pose_recon = torch.stack(pose_recon, 0).mean()
            joint_recon = torch.stack(joint_recon, 0).mean()
            vol_fit = torch.stack(vol_fit, 0).mean()
            shape = torch.stack(shape, 0).mean()
            vertex = torch.stack(vertex, 0).mean()

            self.loss_terms = dict(
                aux_pose=aux_pose,
                kl_pose=kl_pose,
                pose_recon=pose_recon,
                joint_recon=joint_recon,
                vol_fit=vol_fit,
                shape=shape,
                vertex=vertex
            )
        
        elif self.pretraned_mode == 1:
            aux_pose = []
            feature = []
            pose_recon = []
            joint_recon = []
            shape = []
            vertex = []

            for output in outputs:
                aux_pose.append(output['auxiliary_pose'].clone().detach().cpu())
                feature.append(output['feature'].clone().detach().cpu())
                pose_recon.append(output['pose_recon'].clone().detach().cpu())
                joint_recon.append(output['joint_recon'].clone().detach().cpu())
                shape.append(output['shape'].clone().detach().cpu())
                vertex.append(output['vertex'].clone().detach().cpu())
            
            aux_pose = torch.stack(aux_pose, 0).mean()
            feature = torch.stack(feature, 0).mean()
            pose_recon = torch.stack(pose_recon, 0).mean()
            joint_recon = torch.stack(joint_recon, 0).mean()
            shape = torch.stack(shape, 0).mean()
            vertex = torch.stack(vertex, 0).mean()

            self.loss_terms = dict(
                aux_pose=aux_pose,
                feature=feature,
                pose_recon=pose_recon,
                joint_recon=joint_recon,
                shape=shape,
                vertex=vertex
            )

    
    def validation_step(self, batch, batch_nb):
        batch_ids, input_pcd, joints, body_pose, body_shape, trans, partial_pcd = batch
        input_pcd = input_pcd.float()
        joints = joints.float()
        body_pose = body_pose.float()
        body_shape = body_shape.float()
        trans = trans.float()
        partial_pcd = partial_pcd.float()

        B, T, P = body_pose.shape

        nblock = random.randint(0, self.maximum_nblock)

        if nblock == 0:
            padding_mask1d = None
        else:
            padding_mask1d = generate_padding_mask1d(nblock, self.Ttot)
        

        if self.pretrained_mode == 0:
            body_pose_6d = axis_angle_to_rotation_6d(body_pose.reshape(B, T, -1, 3)).view(B, T, -1)

            # PointNet feature extraction
            feat_point, _, _ = self.point_feat(input_pcd.view(B * T, -1, 3).transpose(1, 2))
            feat_point = feat_point.view(B, T, -1)  # (B, T, N_feat)

            # feature encoding using Transformer encoder
            feat_trans, _ = self.transformer_encoder(feat_point, padding_mask1d=padding_mask1d)

            # auxiliary pose estimation
            aux_pose_6d = self.trans_to_pose(feat_trans)  # (B, T, P * 2)
            aux_pose_loss = rotation_6d_difference(body_pose_6d, aux_pose_6d).sum(-1).mean()

            # shape parameter estimation
            best_shape = self.shape_estimator(feat_trans.view(B, -1))  # (B, shape_num)
            best_shape = best_shape[:, None]  # (B, 1, shape_num)
            best_shape = best_shape.repeat_interleave(T, 1)  # (B, T, shape_num)
            best_shape_flat = best_shape.view(B * T, -1)

            # prior distribution
            params_prior = self.extract_prior_dist(feat_trans)
            prior_mean, prior_std = torch.chunk(params_prior, 2, dim=-1)
            prior_std = F.softplus(prior_std) + 1e-4
            z_pose_prior_dist = Normal(prior_mean, prior_std)

            # posterior distribution
            params_post = self.extract_post_dist(torch.cat([feat_trans, body_pose_6d], dim=-1))
            post_mean, post_std = torch.chunk(params_post, 2, dim=-1)
            post_std = F.softplus(post_std) + 1e-4
            z_pose_post_dist = Normal(post_mean, post_std)

            # latent sampling
            z_pose_prior_sampled = z_pose_prior_dist.rsample(sample_shape=(1, ))
            z_pose_post_sampled = z_pose_post_dist.rsample(sample_shape=(1, ))

            # Transformer decoder
            feat_trans_dec = feat_trans.expand(1, -1, -1, -1)

            pose_prior_sampled = self.pose_decoder(z_pose_prior_sampled, feat_trans_dec, padding_mask1d=padding_mask1d)
            prior_pose_6d = pose_prior_sampled.squeeze(0)

            pose_post_sampled = self.pose_decoder(z_pose_post_sampled, feat_trans_dec, padding_mask1d=padding_mask1d)
            post_pose_6d = pose_post_sampled.squeeze(0)

            # root translation estimation
            best_trans = self.trans_decoder(feat_trans)


            # generating smpl/mano body
            prior_pose_axis = rotation_6d_to_axis_angle(prior_pose_6d.view(B, T, -1, 6)).view(B, T, -1)
            prior_pose_flat = prior_pose_axis.view(B * T, -1)
            post_pose_axis = rotation_6d_to_axis_angle(post_pose_6d.view(B, T, -1, 6)).view(B, T, -1)
            post_pose_flat = post_pose_axis.view(B * T, -1)
            best_trans_flat = best_trans.view(B * T, -1)
            best_shape_flat = best_shape.view(B * T, -1)

            body_pose_flat = body_pose.view(B * T, -1)
            trans_flat = trans.view(B * T, -1)
            body_shape_flat = body_shape.view(B * T, -1)

            if self.dataset.startswith('human'):
                prior_body = self.body_model(root_orient=prior_pose_flat[:, :3],
                                             pose_body=prior_pose_flat[:, 3:66],
                                             trans=best_trans_flat,
                                             betas=best_shape_flat)
                prior_keypoint = prior_body.Jtr.view(B, T, -1, 3)
                prior_vertices = prior_body.v.view(B, T, -1, 3)

                post_body = self.body_model(root_orient=post_pose_flat[:, :3],
                                             pose_body=post_pose_flat[:, 3:66],
                                             trans=best_trans_flat,
                                             betas=best_shape_flat)
                post_keypoint = post_body.Jtr.view(B, T, -1, 3)
                post_vertices = post_body.v.view(B, T, -1, 3)

                gt_body = self.body_model(root_orient=body_pose_flat[:, :3],
                                          pose_body=body_pose_flat[:, 3:66],
                                          trans=trans_flat,
                                          betas=body_shape_flat)
                gt_keypoint = gt_body.Jtr.view(B, T, -1, 3)
                gt_vertices = gt_body.v.view(B, T, -1, 3)

            elif self.dataset.startswith('hand'):
                prior_vertices, prior_keypoint = self.hand_model(prior_pose_flat, best_shape_flat, best_trans_flat)
                prior_keypoint = prior_keypoint.view(B, T, -1, 3) * 10
                prior_keypoint = torch.stack([prior_keypoint[..., 2], prior_keypoint[..., 1], -prior_keypoint[..., 0]], -1)
                prior_vertices = prior_vertices.view(B, T, -1, 3) * 10
                prior_vertices = torch.stack([prior_vertices[..., 2], prior_vertices[..., 1], -prior_vertices[..., 0]], -1)

                post_vertices, post_keypoint = self.hand_model(post_pose_flat, best_shape_flat, best_trans_flat)
                post_keypoint = post_keypoint.view(B, T, -1, 3) * 10
                post_keypoint = torch.stack([post_keypoint[..., 2], post_keypoint[..., 1], -post_keypoint[..., 0]], -1)
                post_vertices = post_vertices.view(B, T, -1, 3) * 10
                post_vertices = torch.stack([post_vertices[..., 2], post_vertices[..., 1], -post_vertices[..., 0]], -1)
                
                gt_vertices, gt_keypoint = self.hand_model(body_pose_flat, body_shape_flat, trans_flat)
                gt_keypoint = gt_keypoint.view(B, T, -1, 3) * 10
                gt_keypoint = torch.stack([gt_keypoint[..., 2], gt_keypoint[..., 1], -gt_keypoint[..., 0]], -1)
                gt_vertices = gt_vertices.view(B, T, -1, 3) * 10
                gt_vertices = torch.stack([gt_vertices[..., 2], gt_vertices[..., 1], -gt_vertices[..., 0]], -1)
            
            # kl divergence loss
            kl_pose = kl_divergence(z_pose_post_dist, z_pose_prior_dist).mean()

            # pose reconstruction loss
            pose_recon_loss = rotation_6d_difference(body_pose_6d, prior_pose_6d).sum(-1).mean()

            # joint reconstruction loss
            joint_recon_loss = (prior_keypoint - joints).pow(2).sum(dim=(-2, -1)).mean()

            # volume fitting loss
            dist = (input_pcd[:, :, :, None] - prior_keypoint[:, :, None]).pow(2)  # (B, T, N, K, 3)
            dist = dist.sum(dim=-1)  # (B, T, N, K)
            dist = dist.min(dim=-1).values  # (B, T, N)
            vol_fit_loss = dist.mean()

            # shape loss
            shape_loss = (best_shape - body_shape).pow(2).sum(-1).mean()

            # vertex displacement loss
            vertex_loss = (prior_vertices - gt_vertices).pow(2).sum(dim=(-2, -1)).mean()

            return dict(
                input_pcd=input_pcd,
                joints=joints,
                body_pose=body_pose,
                body_shape=body_shape,
                trans=trans,
                pose_infer=prior_pose_axis,
                pose_recon=post_pose_axis,
                trans_infer=best_trans,
                trans_recon=best_trans,
                joints_infer=prior_keypoint,
                joints_recon=post_keypoint,
                shape_infer=best_shape,
                shape_recon=best_shape,
                partial_pcd=partial_pcd,
                aux_pose_loss=aux_pose_loss,
                kl_pose_loss=kl_pose,
                pose_recon_loss=pose_recon_loss,
                joint_recon_loss=joint_recon_loss,
                vol_fit_loss=vol_fit_loss,
                shape_loss=shape_loss,
                vertex_loss=vertex_loss,
                padding_mask=padding_mask1d
            )

        elif self.pretrained_mode == 1:
            body_pose_6d = axis_angle_to_rotation_6d(body_pose.reshape(B, T, -1, 3)).view(B, T, -1)

            # PointNet feature extraction
            feat_point, _, _ = self.point_feat(input_pcd.view(B * T, -1, 3).transpose(1, 2))
            feat_point = feat_point.view(B, T, -1)  # (B, T, N_feat)

            # feature encoding using Transformer encoder
            feat_trans, _ = self.transformer_encoder(feat_point, padding_mask1d=padding_mask1d)

            # feature encoding from the feature follower
            partial_pcd_feat = self.feature_follower(partial_pcd, padding_mask1d=padding_mask1d)

            # auxiliary pose estimation
            aux_pose_6d = self.trans_to_pose(partial_pcd_feat)  # (B, T, P * 2)
            aux_pose_loss = rotation_6d_difference(body_pose_6d, aux_pose_6d).sum(dim=-1).mean()

            # shape parameter estimation
            part_shape = self.shape_estimator(partial_pcd_feat.view(B, -1))  # (B, shape_num)
            part_shape = part_shape[:, None]  # (B, 1, shape_num)
            part_shape = part_shape.repeat_interleave(T, 1)  # (B, T, shape_num)

            full_shape = self.shape_estimator(feat_trans.view(B, -1))  # (B, shape_num)
            full_shape = full_shape[:, None]  # (B, 1, shape_num)
            full_shape = full_shape.repeat_interleave(T, 1)  # (B, T, shape_num)

            # prior distribution
            params_prior_part = self.extract_prior_dist(partial_pcd_feat)
            prior_mean_part, prior_std_part = torch.chunk(params_prior_part, 2, dim=-1)
            prior_std_part = F.softplus(prior_std_part) + 1e-4
            z_pose_prior_dist_part = Normal(prior_mean_part, prior_std_part)

            params_prior_full = self.extract_prior_dist(feat_trans)
            prior_mean_full, prior_std_full = torch.chunk(params_prior_full, 2, dim=-1)
            prior_std_full = F.softplus(prior_std_full) + 1e-4
            z_pose_prior_dist_full = Normal(prior_mean_full, prior_std_full)

            # latent sampling
            z_pose_sampled_part = z_pose_prior_dist_part.rsample(sample_shape=(1, ))
            z_pose_sampled_full = z_pose_prior_dist_full.rsample(sample_shape=(1, ))

            # Transformer decoder
            feat_trans_dec = feat_trans.expand(1, -1, -1, -1)
            partial_pcd_feat_dec = partial_pcd_feat.expand(1, -1, -1, -1)

            pose_sampled_part = self.pose_decoder(z_pose_sampled_part, partial_pcd_feat_dec, padding_mask1d=padding_mask1d)
            part_pose_6d = pose_sampled_part.squeeze(0)
            pose_sampled_full = self.pose_decoder(z_pose_sampled_full, feat_trans_dec, padding_mask1d=padding_mask1d)
            full_pose_6d = pose_sampled_full.squeeze(0)

            # root translation estimation
            part_trans = self.trans_decoder(partial_pcd_feat)
            full_trans = self.trans_decoder(feat_trans)


            # generating smpl/mano body
            part_pose_axis = rotation_6d_to_axis_angle(part_pose_6d.view(B, T, -1, 6)).view(B, T, -1)
            part_pose_flat = part_pose_axis.view(B * T, -1)
            part_trans_flat = part_trans.view(B * T, -1)
            part_shape_flat = part_shape.view(B * T, -1)

            full_pose_axis = rotation_6d_to_axis_angle(full_pose_6d.view(B, T, -1, 6)).view(B, T, -1)
            full_pose_flat = full_pose_axis.view(B * T, -1)
            full_trans_flat = full_trans.view(B * T, -1)
            full_shape_flat = full_shape.view(B * T, -1)

            body_pose_flat = body_pose.view(B * T, -1)
            trans_flat = trans.view(B * T, -1)
            body_shape_flat = body_shape.view(B * T, -1)

            if self.dataset.startswith('human'):
                part_body = self.body_model(root_orient=part_pose_flat[:, :3],
                                            pose_body=part_pose_flat[:, 3:66],
                                            trans=part_trans_flat,
                                            betas=part_shape_flat)
                part_keypoint = part_body.Jtr.view(B, T, -1, 3)
                part_vertices = part_body.v.view(B, T, -1, 3)

                full_body = self.body_model(root_orient=full_pose_flat[:, :3],
                                            pose_body=full_pose_flat[:, 3:66],
                                            trans=full_trans_flat,
                                            betas=full_shape_flat)
                full_keypoint = full_body.Jtr.view(B, T, -1, 3)
                full_vertices = full_body.v.view(B, T, -1, 3)

                gt_body = self.body_model(root_orient=body_pose_flat[:, :3],
                                            pose_body=body_pose_flat[:, 3:66],
                                            trans=trans_flat,
                                            betas=body_shape_flat)
                gt_keypoint = gt_body.Jtr.view(B, T, -1, 3)
                gt_vertices = gt_body.v.view(B, T, -1, 3)

            elif self.dataset.startswith('hand'):
                part_vertices, part_keypoint = self.hand_model(part_pose_flat, part_shape_flat, part_trans_flat)
                part_keypoint = part_keypoint.view(B, T, -1, 3) * 10
                part_keypoint = torch.stack([part_keypoint[..., 2], part_keypoint[..., 1], -part_keypoint[..., 0]], -1)
                part_vertices = part_vertices.view(B, T, -1, 3) * 10
                part_vertices = torch.stack([part_vertices[..., 2], part_vertices[..., 1], -part_vertices[..., 0]], -1)

                full_vertices, full_keypoint = self.hand_model(full_pose_flat, full_shape_flat, full_trans_flat)
                full_keypoint = full_keypoint.view(B, T, -1, 3) * 10
                full_keypoint = torch.stack([full_keypoint[..., 2], full_keypoint[..., 1], -full_keypoint[..., 0]], -1)
                full_vertices = full_vertices.view(B, T, -1, 3) * 10
                full_vertices = torch.stack([full_vertices[..., 2], full_vertices[..., 1], -full_vertices[..., 0]], -1)
                
                gt_vertices, gt_keypoint = self.hand_model(body_pose_flat, body_shape_flat, trans_flat)
                gt_keypoint = gt_keypoint.view(B, T, -1, 3) * 10
                gt_keypoint = torch.stack([gt_keypoint[..., 2], gt_keypoint[..., 1], -gt_keypoint[..., 0]], -1)
                gt_vertices = gt_vertices.view(B, T, -1, 3) * 10
                gt_vertices = torch.stack([gt_vertices[..., 2], gt_vertices[..., 1], -gt_vertices[..., 0]], -1)
            

            # feature following loss
            feature_loss = (partial_pcd_feat - feat_trans).pow(2).sum(-1).mean()

            # pose reconstruction loss
            pose_recon_loss = rotation_6d_difference(body_pose_6d, part_pose_6d).sum(-1).mean()

            # joint reconstruction loss
            joint_recon_loss = (part_keypoint - joints).pow(2).sum((-2, -1)).mean()

            # shape loss
            shape_loss = (part_shape - body_shape).pow(2).sum(-1).mean()

            # vertex displacement loss
            vertex_loss = (part_vertices - gt_vertices).pow(2).sum((-2, -1)).mean()

            return dict(
                input_pcd=input_pcd,
                joints=joints,
                body_pose=body_pose,
                body_shape=body_shape,
                trans=trans,
                pose_infer=full_pose_axis,
                pose_recon=part_pose_axis,
                trans_infer=full_trans,
                trans_recon=part_trans,
                joints_infer=full_keypoint,
                joints_recon=part_keypoint,
                shape_infer=full_shape,
                shape_recon=part_shape,
                partial_pcd=partial_pcd,
                aux_pose_loss=aux_pose_loss,
                feature_loss=feature_loss,
                pose_recon_loss=pose_recon_loss,
                joint_recon_loss=joint_recon_loss,
                shape_loss=shape_loss,
                vertex_loss=vertex_loss,
                padding_mask=padding_mask1d
            )
        
    def validation_epoch_end(self, outputs):
        input_pcd = outputs[0]['input_pcd'][:self.log_gif_num].cpu()
        joints = outputs[0]['joints'][:self.log_gif_num].cpu()
        body_pose = outputs[0]['body_pose'][:self.log_gif_num].cpu()
        body_shape = outputs[0]['body_shape'][:self.log_gif_num].cpu()
        trans = outputs[0]['trans'][:self.log_gif_num].cpu()
        pose_infer = outputs[0]['pose_infer'][:self.log_gif_num].cpu()
        pose_recon = outputs[0]['pose_recon'][:self.log_gif_num].cpu()
        trans_infer = outputs[0]['trans_infer'][:self.log_gif_num].cpu()
        trans_recon = outputs[0]['trans_recon'][:self.log_gif_num].cpu()
        joints_infer = outputs[0]['joints_infer'][:self.log_gif_num].cpu()
        joints_recon = outputs[0]['joints_recon'][:self.log_gif_num].cpu()
        shape_infer = outputs[0]['shape_infer'][:self.log_gif_num].cpu()
        shape_recon = outputs[0]['shape_recon'][:self.log_gif_num].cpu()
        partial_pcd = outputs[0]['partial_pcd'][:self.log_gif_num].cpu()

        padding_mask = outputs[0]['padding_mask']
        if padding_mask is not None:
            padding_mask = padding_mask.cpu()
        
        if self.pretrained_mode == 0:
            aux_pose_loss = []
            kl_pose_loss = []
            pose_recon_loss = []
            joint_recon_loss = []
            vol_fit_loss = []
            shape_loss = []
            vertex_loss = []

            for output in outputs:
                aux_pose_loss.append(output['aux_pose_loss'])
                kl_pose_loss.append(output['kl_pose_loss'])
                pose_recon_loss.append(output['pose_recon_loss'])
                joint_recon_loss.append(output['joint_recon_loss'])
                vol_fit_loss.append(output['vol_fit_loss'])
                shape_loss.append(output['shape_loss'])
                vertex_loss.append(output['vertex_loss'])
            
            aux_pose_loss = torch.stack(aux_pose_loss, 0).mean()
            kl_pose_loss = torch.stack(kl_pose_loss, 0).mean()
            pose_recon_loss = torch.stack(pose_recon_loss, 0).mean()
            joint_recon_loss = torch.stack(joint_recon_loss, 0).mean()
            vol_fit_loss = torch.stack(vol_fit_loss, 0).mean()
            shape_loss = torch.stack(shape_loss, 0).mean()
            vertex_loss = torch.stack(vertex_loss, 0).mean()

            self.val_loss_terms = dict(
                aux_pose_loss=aux_pose_loss,
                kl_pose_loss=kl_pose_loss,
                pose_recon_loss=pose_recon_loss,
                joint_recon_loss=joint_recon_loss,
                vol_fit_loss=vol_fit_loss,
                shape_loss=shape_loss,
                vertex_loss=vertex_loss
            )
        
        elif self.pretrained_mode == 1:
            aux_pose_loss = []
            feature_loss = []
            pose_recon_loss = []
            joint_recon_loss = []
            shape_loss = []
            vertex_loss = []

            for output in outputs:
                aux_pose_loss.append(output['aux_pose_loss'])
                feature_loss.append(output['feature_loss'])
                pose_recon_loss.append(output['pose_recon_loss'])
                joint_recon_loss.append(output['joint_recon_loss'])
                shape_loss.append(output['shape_loss'])
                vertex_loss.append(output['vertex_loss'])
            
            aux_pose_loss = torch.stack(aux_pose_loss, 0).mean()
            feature_loss = torch.stack(feature_loss, 0).mean()
            pose_recon_loss = torch.stack(pose_recon_loss, 0).mean()
            joint_recon_loss = torch.stack(joint_recon_loss, 0).mean()
            shape_loss = torch.stack(shape_loss, 0).mean()
            vertex_loss = torch.stack(vertex_loss, 0).mean()

            self.val_loss_terms = dict(
                aux_pose_loss=aux_pose_loss,
                feature_loss=feature_loss,
                pose_recon_loss=pose_recon_loss,
                joint_recon_loss=joint_recon_loss,
                shape_loss=shape_loss,
                vertex_loss=vertex_loss
            )
    
        
        self.val_outputs = dict(
            input_pcd=input_pcd,
            joints=joints,
            body_pose=body_pose,
            body_shape=body_shape,
            trans=trans,
            pose_infer=pose_infer,
            pose_recon=pose_recon,
            trans_infer=trans_infer,
            trans_recon=trans_recon,
            joints_infer=joints_infer,
            joints_recon=joints_recon,
            shape_infer=shape_infer,
            shape_recon=shape_recon,
            partial_pcd=partial_pcd,
            padding_mask=padding_mask
        )