import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append('src/dataset')
from human_body_prior.body_model.body_model import BodyModel
from manopth.manolayer import ManoLayer


def vis_keypoints(input_pcd, keypoints, logger_path, log_num=8, group='track', padding_mask=None):
    '''
    input_pcd: (B, T, N, 3)
    keypoints: (B, T, K, 3)
    logger_path: str
    log_num: int
    group: str
    padding_mask: None or (T, )
    '''

    save_gif_root = os.path.join(logger_path, 'gifs', group)
    os.makedirs(save_gif_root, exist_ok=True)

    B, T, K, _ = keypoints.shape

    if log_num > B:
        log_num = B
    
    input_pcd = input_pcd[:log_num].detach().cpu().numpy()
    keypoints = keypoints[:log_num].detach().cpu().numpy()

    if K == 24:
        parents = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])
    elif K == 21:
        parents = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('X-axis', fontweight='bold')
    ax.set_ylabel('Z-axis', fontweight='bold')
    ax.set_zlabel('Y-axis', fontweight='bold')

    gif = []
    for b in range(log_num):
        gif_b = []
        for t in range(T):
            coords = input_pcd[b, t]  # (N, 3)
            keypoint = keypoints[b, t]  # (K, 3)

            point_size = int(2048 / len(coords))
            if padding_mask is not None:
                if not padding_mask[t]:
                    ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2], color='grey', s=point_size, alpha=0.3)
            else:
                ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2], color='grey', s=point_size, alpha=0.3)
            
            if group == 'track':
                color = 'red'
            elif group == 'gen':
                color = 'blue'
            
            for k in range(K):
                ax.plot(keypoint[k, 0], keypoint[k, 1], keypoint[k, 2], color=color, marker='o', markersize=6, linewidth=0)
                
                if k > 0:
                    pairs = np.stack([keypoint[parents[k]], keypoint[k]], axis=0)
                    ax.plot([pairs[0, 0], pairs[1, 0]], [pairs[0, 1], pairs[1, 1]], [pairs[0, 2], pairs[1, 2]],
                            color='green', linewidth=1.7)
            
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.view_init(elev=10, azim=-60)

            save_file = os.path.join(save_gif_root, 'keypoints_%d_%d.png' % (b, t))
            fig.savefig(save_file)
            ax.cla()

            image = cv2.imread(save_file)[100:730, 120:750]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gif_b.append(torch.from_numpy(image))
        gif_b = torch.stack(gif_b, dim=0)
        gif.append(gif_b)
    gif = torch.stack(gif, dim=0)  # (B, T, H, W, C)
    plt.close(fig)
    return gif.permute(0, 1, 4, 2, 3)


def vis_mesh(pose, shape, trans, logger_path, log_num=8, group='mesh', padding_mask=None):

    save_gif_root = os.path.join(logger_path, 'gifs', group)
    os.makedirs(save_gif_root, exist_ok=True)

    B, T, P = pose.shape

    if log_num > B:
        log_num = B
    
    pose_cpu = pose[:log_num].detach().cpu()
    shape_cpu = shape[:log_num].detach().cpu()
    trans_cpu = trans[:log_num].detach().cpu()

    if P == 66:
        bm_smpl_fname = 'data/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'
        body_model = BodyModel(bm_fname=bm_smpl_fname, num_betas=shape_cpu.shape[-1])
    elif P == 48:
        hand_model = ManoLayer(mano_root='data/models', use_pca=False, ncomps=45, flat_hand_mean=False, center_idx=9)

    if group == 'track':
        color = [0.5, 1.0, 0.5]
    elif group == 'gen':
        color = [0.5, 0.5, 1.0]
    elif group == 'input':
        color = [1, 1, 1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('X-axis', fontweight='bold')
    ax.set_ylabel('Z-axis', fontweight='bold')
    ax.set_zlabel('Y-axis', fontweight='bold')

    gif = []

    for b in range(log_num):
        gif_b = []

        if P == 66:
            body = body_model(root_orient=pose_cpu[b, :, :3],
                            pose_body=pose_cpu[b, :, 3:66],
                            trans=trans_cpu[b],
                            betas=shape_cpu[b, :T])
            
            faces = body.f
        elif P == 48:
            verts, _ = hand_model(pose_cpu[b], shape_cpu[b, :T], trans_cpu[b])
            faces = hand_model.th_faces

        for t in range(T):
            if P == 66:
                vertices = body.v[t] - trans_cpu[b, 0]
            elif P == 48:
                vertices = verts[t] * 10
                vertices = torch.stack([vertices[..., 2], vertices[..., 1], -vertices[..., 0]], -1)

            if padding_mask is not None:
                if not padding_mask[t]:
                    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color=color)
            else:
                ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color=color)

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.view_init(elev=10, azim=-60)

            save_file = os.path.join(save_gif_root, 'mesh_%d_%d.png' % (b, t))
            fig.savefig(save_file)
            ax.cla()

            image = cv2.imread(save_file)[100:730, 120:750]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gif_b.append(torch.from_numpy(image))
        
        gif_b = torch.stack(gif_b, dim=0)
        gif.append(gif_b)

    plt.close(fig)
    gif = torch.stack(gif, dim=0)

    return gif.permute(0, 1, 4, 2, 3)


def vis_points(input_pcd, logger_path, log_num=8, group='track', padding_mask=None):

    save_gif_root = os.path.join(logger_path, 'gifs', group)
    os.makedirs(save_gif_root, exist_ok=True)

    B, T, N, _ = input_pcd.shape

    if log_num > B:
        log_num = B
    
    input_pcd = input_pcd[:log_num].detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('X-axis', fontweight='bold')
    ax.set_ylabel('Z-axis', fontweight='bold')
    ax.set_zlabel('Y-axis', fontweight='bold')

    gif = []
    for b in range(log_num):
        gif_b = []
        for t in range(T):
            coords = input_pcd[b, t]
            point_size = int(4096 / len(coords))

            if padding_mask is not None:
                if not padding_mask[t]:
                    ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2], color='grey', s=point_size, alpha=0.5)
            else:
                ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2], color='grey', s=point_size, alpha=0.5)
                
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.view_init(elev=10, azim=-60)

            save_file = os.path.join(save_gif_root, 'points_%d_%d.png' % (b, t))
            fig.savefig(save_file)
            ax.cla()

            image = cv2.imread(save_file)[100:730, 120:750]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gif_b.append(torch.from_numpy(image))
        gif_b = torch.stack(gif_b, dim=0)
        gif.append(gif_b)
    
    plt.close(fig)
    gif = torch.stack(gif, dim=0)  # (B, T, H, W, C)

    return gif.permute(0, 1, 4, 2, 3)