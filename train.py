import argparse
import sys
import torch
import os
import numpy as np
import collections
from glob import glob
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(1, '.')

from src.dataset.config import adjust_config
from src.dataset.dataset_pl import DATA_PL
from src.mesh_recovery import MeshRecovery

import pytorch_lightning as pl
from src.utils.pl_callbacks import *


#========================================================================================
#                                   argument parsing
#========================================================================================
parser = argparse.ArgumentParser()
# about training itself
parser.add_argument('--seed', type=int, default=2, help='seed for random')
parser.add_argument('--nepoch', type=int, default=2000, help='')
parser.add_argument('--lrate', type=float, default=5e-4, help='')
parser.add_argument('--firstdecay', type=int, default=1200, help='epoch for first lr decay')
parser.add_argument('--seconddecay', type=int, default=1600, help='epoch for second lr decay')
parser.add_argument('--max_grad_norm', type=float, default=30.0, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

# about saving & logging
parser.add_argument('--exp_name', type=str, default='default', help='exp name')
parser.add_argument('--training_id', type=str, default='default', help='')
parser.add_argument('--save_every', type=int, default=10, help='saving epoch frequency')
parser.add_argument('--log_gif_num', type=int, default=8, help='sequence to log with tensorboard')

# about dataset
parser.add_argument('--dataset', type=str, default='human', help='dataset name')
parser.add_argument('--nbatch', type=int, default=32, help='')
parser.add_argument('--input_dim', type=int, default=6, help='')
parser.add_argument('--state_dim', type=int, default=66, help='')
parser.add_argument('--inputpc_size', type=int, default=1024, help='point cloud size')
parser.add_argument('--Ttot', type=int, default=40, help='total frame number')
parser.add_argument('--fps', type=int, default=10, help='input fps')
parser.add_argument('--shape_num', type=int, default=10, help='')

# dataset adjustment
parser.add_argument('--br_initial', type=float, default=0.0, help='initial key-padding mask block ratio')
parser.add_argument('--random_crop', type=int, default=1, help='random crop sequence from total data sequence')
parser.add_argument('--add_random_noise', type=bool, default=False, help='')
parser.add_argument('--noise_sigma', type=float, default=0.0, help='')

# about architecture
parser.add_argument('--nlatent_kypt', type=int, default=128, help='kypt latent dimension')
parser.add_argument('--nhidden_kypt', type=int, default=128, help='kypt rnn state dimension')
parser.add_argument('--npoint_feat', type=int, default=512, help='')

# loss weights
parser.add_argument('--aux_weight', type=float, default=0.5, help='')
parser.add_argument('--kl_pose_weight', type=float, default=1.0, help='')
parser.add_argument('--pose_recon_weight', type=float, default=0.5, help='')
parser.add_argument('--joint_recon_weight', type=float, default=0.5, help='')
parser.add_argument('--vol_fit_weight', type=float, default=1.0, help='')
parser.add_argument('--shape_weight', type=float, default=0.1, help='')
parser.add_argument('--vertex_weight', type=float, default=0.1, help='')
parser.add_argument('--feature_weight', type=float, default=1.0, help='')

# pretrain
parser.add_argument('--pretrained_mode', type=int, default=0, help='')
parser.add_argument('--pretrained_pth', type=str, default=None, help='')

opt = parser.parse_args()
opt = adjust_config(opt)

# make network deterministic
torch.backends.cudnn.deterministic = True
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)
pl.seed_everything(opt.seed)

# set train id
if opt.pretrained_mode == 0:
    opt.training_id = 'kinematics_learner/%s' % (opt.dataset)
elif opt.pretrained_mode == 1:
    opt.training_id = 'feature_follower/%s' % (opt.dataset)
elif opt.pretrained_mode == 2:
    opt.training_id = 'test/%s' % (opt.dataset)

# adjust log gif num
if opt.log_gif_num > opt.nbatch:
    opt.log_gif_num = opt.nbatch


if not os.path.exists('output'):
    os.makedirs('output')

logger_path = './output/%s/%s' % (opt.training_id, opt.exp_name)
ckpt_path = None

if not os.path.exists(logger_path):
    os.makedirs(os.path.join(logger_path))
else:
    ckpt_path_list = glob(os.path.join(logger_path, 'lightning_logs', 'version_0', 'checkpoints', '*'))
    if len(ckpt_path_list) > 0:
        ckpt_path = sorted(ckpt_path_list)[-1]

dataset = DATA_PL(opt)
model = MeshRecovery(opt)

# loading checkpoints
if opt.pretrained_mode == 1:
    checkpoint = torch.load(opt.pretrained_pth)
    checkpoint = checkpoint['state_dict']
    modules = [
        'extract_post_dist',
        'extract_prior_dist',
        'point_feat',
        'pose_decoder',
        'trans_decoder',
        'trans_to_pose',
        'transformer_encoder',
        'shape_estimator'
    ]
    for module in modules:
        ckpt_part = collections.OrderedDict(filter(lambda p: p[0].split('.')[0] == module, checkpoint.items()))
        ckpt_part = collections.OrderedDict({k[len(module)+1:]: v for k, v in ckpt_part.items()})
        if module == 'extract_post_dist':
            model.extract_post_dist.load_state_dict(ckpt_part)
        elif module == 'extract_prior_dist':
            model.extract_prior_dist.load_state_dict(ckpt_part)
        elif module == 'point_feat':
            model.point_feat.load_state_dict(ckpt_part)
        elif module == 'pose_decoder':
            model.pose_decoder.load_state_dict(ckpt_part)
        elif module == 'trans_decoder':
            model.trans_decoder.load_state_dict(ckpt_part)
        elif module == 'trans_to_pose':
            model.trans_to_pose.load_state_dict(ckpt_part)
        elif module == 'transformer_encoder':
            model.transformer_encoder.load_state_dict(ckpt_part)
        elif module == 'shape_estimator':
            model.shape_estimator.load_state_dict(ckpt_part)

elif opt.pretrained_mode == 2:
    checkpoint = torch.load(opt.pretrained_pth)
    checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)

logger = pl.loggers.TensorBoardLogger(save_dir=logger_path, version=0)

if opt.pretrained_mode == 0 or opt.pretrained_mode == 1:
    trainer = pl.Trainer(accelerator='gpu', devices=1,
                        max_epochs=opt.nepoch,
                        deterministic=True,
                        detect_anomaly=True,
                        gradient_clip_val=opt.max_grad_norm,
                        check_val_every_n_epoch=opt.save_every,
                        log_every_n_steps=1,
                        num_sanity_val_steps=0,
                        default_root_dir=logger_path,
                        logger=logger,
                        callbacks=[GifLogCallback(), LRDecayCallback(), TrainLogCallback(), ValLogCallback()])

    trainer.fit(model, dataset, ckpt_path=ckpt_path)
    
elif opt.pretrained_mode == 2:
    trainer = pl.Trainer(accelerator='gpu', devices=1,
                         deterministic=True,
                         detect_anomaly=True,
                         default_root_dir=logger_path,
                         logger=logger,
                         callbacks=[GifLogCallback(), ValLogCallback()])

    trainer.validate(model, dataset)