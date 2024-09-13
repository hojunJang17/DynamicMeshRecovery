import torch
from pytorch_lightning.callbacks import Callback
from src.visualize.visualize import *

class LRDecayCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if pl_module.current_epoch >= pl_module.firstdecay and pl_module.current_epoch < pl_module.seconddecay:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, pl_module.parameters()), lr=pl_module.lr / 4.)
            trainer.optimizers = [optimizer]
        if pl_module.current_epoch >= pl_module.seconddecay:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, pl_module.parameters()), lr=pl_module.lr / 10.)
            trainer.optimizers = [optimizer]

class TrainLogCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.logger and hasattr(pl_module, 'loss_terms'):
            outputs = pl_module.loss_terms
            if pl_module.pretrained_mode == 0:
                aux_pose = outputs['aux_pose']
                kl_pose = outputs['kl_pose']
                pose_recon = outputs['pose_recon']
                joint_recon = outputs['joint_recon']
                vol_fit = outputs['vol_fit']
                shape = outputs['shape']

                trainer.logger.experiment.add_scalar('a_train/aux_pose', aux_pose, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('a_train/kl_pose', kl_pose, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('a_train/pose_recon', pose_recon, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('a_train/joint_recon', joint_recon, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('a_train/vol_fit', vol_fit, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('a_train/shape', shape, pl_module.current_epoch)

            elif pl_module.pretrained_mode == 1:
                latent_diff = outputs['latent_diff']
                shape = outputs['shape']
                aux_pose = outputs['aux_pose']
                pose_recon = outputs['pose_recon']
                joint_recon = outputs['joint_recon']

                trainer.logger.experiment.add_scalar('b_train/latent_diff', latent_diff, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('b_train/aux_pose', aux_pose, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('b_train/shape', shape, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('b_train/joint_recon', joint_recon, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('b_train/pose_recon', pose_recon, pl_module.current_epoch)

class ValLogCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.logger:
            outputs = pl_module.val_loss_terms
            if pl_module.pretrained_mode == 0:
                kl_pose = outputs['kl_pose_loss']
                pose_recon = outputs['pose_recon_loss']
                joint_recon = outputs['joint_recon_loss']
                vol_fit = outputs['vol_fit_loss']
                shape = outputs['shape_loss']

                trainer.logger.experiment.add_scalar('a_val/kl_pose', kl_pose, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('a_val/pose_recon', pose_recon, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('a_val/joint_recon', joint_recon, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('a_val/vol_fit', vol_fit, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('a_val/shape', shape, pl_module.current_epoch)

            elif pl_module.pretrained_mode == 1 or pl_module.pretrained_mode == 2:
                latent_diff = outputs['latent_diff_loss']
                shape = outputs['shape_loss']
                pose_recon = outputs['pose_recon_loss']
                joint_recon = outputs['joint_recon_loss']
                vertex = outputs['vertex_loss']

                trainer.logger.experiment.add_scalar('b_val/pose_recon', pose_recon, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('b_val/joint_recon', joint_recon, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('b_val/latent_diff', latent_diff, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('b_val/shape', shape, pl_module.current_epoch)
                trainer.logger.experiment.add_scalar('b_val/vertex', vertex, pl_module.current_epoch)
                

class GifLogCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.logger:
            outputs = pl_module.val_outputs
            input_pcd = outputs['input_pcd']
            joints = outputs['joints']
            body_pose = outputs['body_pose']
            body_shape = outputs['body_shape']
            trans = outputs['trans']
            pose_infer = outputs['pose_infer']
            pose_recon = outputs['pose_recon']
            trans_infer = outputs['trans_infer']
            trans_recon = outputs['trans_recon']
            joints_infer = outputs['joints_infer']
            joints_recon = outputs['joints_recon']
            shape_infer = outputs['shape_infer']
            shape_recon = outputs['shape_recon']
            partial_pcd = outputs['partial_pcd']
            padding_mask = outputs['padding_mask']

            fps = pl_module.fps

            gif_input_mesh = vis_mesh(pose=body_pose,
                                      shape=body_shape,
                                      trans=trans,
                                      logger_path=pl_module.logger_path,
                                      log_num=pl_module.log_gif_num,
                                      group='input')
            
            gif_keypoints_gt = vis_keypoints(input_pcd=input_pcd,
                                             keypoints=joints,
                                             logger_path=pl_module.logger_path,
                                             log_num=pl_module.log_gif_num,
                                             group='track')

            gif_keypoints_infer = vis_keypoints(input_pcd=input_pcd,
                                                keypoints=joints_infer,
                                                logger_path=pl_module.logger_path,
                                                log_num=pl_module.log_gif_num,
                                                group='track',
                                                padding_mask=padding_mask)

            gif_keypoints_recon = vis_keypoints(input_pcd=partial_pcd,
                                                keypoints=joints_recon,
                                                logger_path=pl_module.logger_path,
                                                log_num=pl_module.log_gif_num,
                                                group='gen',
                                                padding_mask=padding_mask)
            
            gif_mesh_infer = vis_mesh(pose=pose_infer,
                                     shape=shape_infer,
                                     trans=trans_infer,
                                     logger_path=pl_module.logger_path,
                                     log_num=pl_module.log_gif_num,
                                     group='track')

            gif_mesh_recon = vis_mesh(pose=pose_recon,
                                     shape=shape_recon,
                                     trans=trans_recon,
                                     logger_path=pl_module.logger_path,
                                     log_num=pl_module.log_gif_num,
                                     group='gen')

            gif_input_point = vis_points(input_pcd=partial_pcd,
                                         logger_path=pl_module.logger_path,
                                         log_num=pl_module.log_gif_num,
                                         group='input',
                                         padding_mask=padding_mask)
            
            for i in range(pl_module.log_gif_num):
                trainer.logger.experiment.add_video(f'input_mesh/mesh_{i}',  gif_input_mesh[i][None], pl_module.current_epoch, fps=fps)
                trainer.logger.experiment.add_video(f'input_pcd/pcd_{i}', gif_input_point[i][None], pl_module.current_epoch, fps=fps)
                trainer.logger.experiment.add_video(f'gt_kypt/kypt_{i}', gif_keypoints_gt[i][None], pl_module.current_epoch, fps=fps)
                trainer.logger.experiment.add_video(f'infer_kypt/kypt_{i}', gif_keypoints_infer[i][None], pl_module.current_epoch, fps=fps)
                trainer.logger.experiment.add_video(f'infer_mesh/mesh_{i}', gif_mesh_infer[i][None], pl_module.current_epoch, fps=30)
                trainer.logger.experiment.add_video(f'gen_kypt/kypt_{i}', gif_keypoints_recon[i][None], pl_module.current_epoch, fps=fps)
                trainer.logger.experiment.add_video(f'gen_mesh/mesh_{i}', gif_mesh_recon[i][None], pl_module.current_epoch, fps=30)