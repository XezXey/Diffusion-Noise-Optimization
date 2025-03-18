import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class CondReporjectionLoss:
    def __init__(self,
                 target=None,
                 target_joints_2d=None,
                 target_mask=None,
                 motion_length=None,
                 transform=None,
                 inv_transform=None,
                 abs_3d=False,
                 reward_model=None,
                 reward_model_args=None,
                 use_mse_loss=False,
                 guidance_style='xstart',
                 stop_cond_from=0,
                 use_rand_projection=False,
                 print_every=None,
                 obs_list=[],
                 w_colli=1.0,
                 ):
        self.target = target
        self.target_mask = target_mask
        self.motion_length = motion_length
        self.transform = transform
        self.inv_transform = inv_transform
        self.abs_3d = abs_3d
        self.reward_model = reward_model
        self.reward_model_args = reward_model_args
        self.use_mse_loss = use_mse_loss
        self.guidance_style = guidance_style
        self.stop_cond_from = stop_cond_from
        self.use_rand_projection = use_rand_projection
        self.print_every = print_every
        self.obs_list = obs_list
        self.w_colli = w_colli
        # Reprojection loss
        self.target_joints_2d = target_joints_2d
        
        self.n_joints = 22
        # NOTE: optimizing to the whole trajectory is not good.
        self.gt_style = 'target'  # 'inpainting_motion'

    def __call__(self, xstart_in, cam_dict, y=None,): # *args, **kwds):
        """
        Args:
            xstart_in: [bs, 263, 1, 120]
            target: [bs, 120, 22, 3]
            motion_length: [bs]
            target_mask: [bs, 120, 22, 3]
            cam_dict: Camera parameters (camera_R, camera_t, camera_center, focal_length) (Need to be optimized)
        """
        target = self.target
        target_joints_2d = self.target_joints_2d
        target_mask = self.target_mask
        motion_length = self.motion_length

        # motion_mask shape [bs, 120, 22, 3]
        if motion_length is None:
            motion_mask = torch.ones_like(target_mask)
        else:
            # the mask is only for the first motion_length frames
            motion_mask = torch.zeros_like(target_mask)
            for i, m_len in enumerate(motion_length):
                motion_mask[i, :m_len, :, :] = 1.

        assert y is not None
        # x shape [bs, 263, 1, len]
        with torch.enable_grad():
            if self.gt_style == 'target':
                if self.guidance_style == 'xstart':
                    xstart_in = xstart_in
                elif self.guidance_style == 'eps':
                    # using epsilon style guidance
                    raise NotImplementedError()
                else:
                    raise NotImplementedError()
                # import pdb; pdb.set_trace()
                if y['traj_model']:
                    use_rand_proj = False  # x contains only (pose,x,z,y)
                else:
                    use_rand_proj = self.use_rand_projection
                x_in_pose_space = self.inv_transform(
                    xstart_in.permute(0, 2, 3, 1),
                    traject_only=y['traj_model'],
                    use_rand_proj=use_rand_proj
                )  # [bs, 1, 120, 263]
                # x_in_adjust[:,:,:, [1,2]] == x_in_joints[:, :, :, 0, [0,2]]
                # Compute (x,y,z) shape [bs, 1, 120, njoints=22, nfeat=3]
                x_in_joints = recover_from_ric(x_in_pose_space, self.n_joints,
                                            abs_3d=self.abs_3d)  
                # Assume the target has dimention [bs, 120, 22, 3] in case we do key poses instead of key location
                # Only care about XZ position for now. Y-axis is going up from the ground
                # remove the feature dim
                x_in_joints = x_in_joints.squeeze(1)

                loss_fn = reprojection_loss(x_in_joints, target_joints_2d, cam_dict)
                loss_sum = loss_fn(x_in_joints, target, reduction="none") * target_mask * motion_mask 

                adaptive_mean = True
                if adaptive_mean: 
                    # average the loss over the number of valid joints
                    loss_sum = loss_sum.sum(dim=[1, 2, 3]) / (target_mask * motion_mask).sum(dim=[1, 2, 3])
                else:
                    # average naively over the batch
                    loss_sum = loss_sum.mean(dim=[1,2,3])
            else:   
                raise NotImplementedError()
            
            #NOTE: This is for collision avoidance
            # assert len(self.obs_list) == 0, "not implemented"
            loss_colli = 0.0
            for ((c_x, c_z), rad) in self.obs_list:
                # import pdb; pdb.set_trace()
                cent = torch.tensor([c_x, c_z], device=x_in_joints.device)
                dist = torch.norm(x_in_joints[:, :, 0, [0, 2]] - cent, dim=2)
                dist = torch.clamp(rad - dist, min=0.0)
                # import pdb; pdb.set_trace()
                loss_colli += dist.sum(1) / x_in_joints.shape[1] * self.w_colli
                print(loss_colli)
                # loss_colli *= 0.1
            # import pdb; pdb.set_trace()

            loss_sum += loss_colli
            # # if self.print_every is not None a:
            # #     print("%03d: %f" % (int(t[0]), float(loss_sum) / batch_size))

            return loss_sum

def reprojection_loss(x_in_joints, target_joints_2d, cam_dict):
    """
    Reprojection loss
    Input:
        x_in_joints (bs, N, 3, T): 3D model joints; N is the number of joints, T is the number of frames
        cam_dict: Camera parameters (camera_R, camera_t, camera_center, focal_length)
        
    """
    camera_T = cam_dict['camera_T'] # BS x 3 x T  # 3D-Translation of T frames
    camera_R = cam_dict['camera_R'] # BS x 6 x T  # 6D-Rotation of T frames 
    camera_R = rotation_6d_to_matrix(camera_R.permute(0, 2, 1))
    camera_center = cam_dict['camera_center']   # BS x 2  # 2D-Center of the camera fixed for all frames
    focal_length = cam_dict['focal_length']  # BS x 1  # Focal length of the camera fixed for all frames
    
    
    # Project model joints
    B = x_in_joints.shape[0]
    x_in_joints_projected = perspective_projection(x_in_joints, camera_R, camera_T,
                                            focal_length, camera_center)

    # Compute reprojection error
    use_mse_loss = False
    loss_fn = F.mse_loss if use_mse_loss else F.l1_loss
    loss_sum = loss_fn(x_in_joints_projected, target_joints_2d, reduction="none") # * target_mask * motion_mask 
    total_loss = loss_sum

    return total_loss.sum()

def perspective_projection(points, rotation, translation,
                        focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)