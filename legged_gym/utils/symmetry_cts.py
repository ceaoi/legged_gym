from __future__ import annotations

from typing import Optional, Tuple
import torch


_POLICY_OBS_DIM = 53
_NUM_ACTIONS = 16
_CMD_MIRROR_EPS = 0.01

# 假设 action/dof 排列为: [FL(3), FR(3), RL(3), RR(3)]
# 左右镜像: FL<->FR, RL<->RR
_PERM_12_leg = [3, 4, 5, 
                0, 1, 2, 
                9, 10, 11, 
                6, 7, 8]
_PERM_16 = [3, 4, 5, 
            0, 1, 2, 
            9, 10, 11, 
            6, 7, 8, 
            13, 12, 15, 14]  # 包含轮部关节
# hip 关节左右符号相反（从 default_joint_angles: FL_hip=-, FR_hip=+ 可推断）
_SIGN_12_leg = [-1.0, 1.0, 1.0, 
                -1.0, 1.0, 1.0, 
                -1.0, 1.0, 1.0, 
                -1.0, 1.0, 1.0]
_SIGN_16 = [-1.0, 1.0, 1.0, 
            -1.0, 1.0, 1.0, 
            -1.0, 1.0, 1.0, 
            -1.0, 1.0, 1.0, 
            1.0, 1.0, 1.0, 1.0]  # 包含轮部关节
_PERM_16_orig = [4, 5, 6, 7, 
                 0, 1, 2, 3, 
                 12, 13, 14, 15, 
                 8, 9, 10, 11]  # 包含轮部关节
_SIGN_16_orig = [-1.0, 1.0, 1.0, 1.0,
                 -1.0, 1.0, 1.0, 1.0,
                 -1.0, 1.0, 1.0, 1.0,
                 -1.0, 1.0, 1.0, 1.0]  # 包含轮部关节

def _mirror_dof_leg(x: torch.Tensor) -> torch.Tensor:
    perm = torch.as_tensor(_PERM_12_leg, device=x.device, dtype=torch.long)
    sign = x.new_tensor(_SIGN_12_leg)
    return x.index_select(dim=-1, index=perm) * sign

def _mirror_dof_all_orig(x: torch.Tensor) -> torch.Tensor:
    perm = torch.as_tensor(_PERM_16_orig, device=x.device, dtype=torch.long)
    sign = x.new_tensor(_SIGN_16_orig)
    return x.index_select(dim=-1, index=perm) * sign

def _mirror_dof_all(x: torch.Tensor) -> torch.Tensor:
    perm = torch.as_tensor(_PERM_16, device=x.device, dtype=torch.long)
    sign = x.new_tensor(_SIGN_16)
    return x.index_select(dim=-1, index=perm) * sign

def _should_skip_mirroring_from_cmd(cmd: torch.Tensor, eps: float = _CMD_MIRROR_EPS) -> torch.Tensor:
    """
    根据命令中的 [side, yaw] 两个分量判断是否跳过镜像。
    返回 shape 为 obs/actions 前缀 batch 维度的 bool mask：
    True  -> 跳过镜像，直接返回原样
    False -> 执行镜像
    """
    # cmd[..., 1:3] == [side_cmd, yaw_cmd]
    return torch.linalg.norm(cmd[..., 1:3], dim=-1) <= eps

def _mirror_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] != _POLICY_OBS_DIM:
        raise ValueError(f"policy obs dim should be {_POLICY_OBS_DIM}, got {obs.shape[-1]}")

    w = obs[..., 0:3] # base_ang_vel
    g = obs[..., 3:6] # projected_gravity
    q_leg = obs[..., 6:18]
    dq = obs[..., 18:34]
    actions = obs[..., 34:50]
    cmd = obs[..., 50:53] # commands

    w_m = w * w.new_tensor([-1.0, 1.0, -1.0])
    g_m = g * g.new_tensor([1.0, -1.0, 1.0])
    cmd_m = cmd * cmd.new_tensor([1.0, -1.0, -1.0])

    q_leg_m = _mirror_dof_leg(q_leg)
    dq_m = _mirror_dof_all(dq)
    actions_m = _mirror_dof_all(actions)

    return torch.cat([w_m, g_m, q_leg_m, dq_m, actions_m, cmd_m], dim=-1)


def _get_height_grid_shape() -> Tuple[int, int]:
    # 默认来自 LeggedRobotCfg.terrain.measured_points_{x,y}
    # nx = len(env.cfg.terrain.measured_points_x)
    # ny = len(env.cfg.terrain.measured_points_y)
    return 17, 11


def _mirror_privileged_obs(privileged_obs: torch.Tensor) -> torch.Tensor:
    # head parts
    w = privileged_obs[..., 0:3] # base_ang_vel
    g = privileged_obs[..., 3:6] # projected_gravity
    q_leg = privileged_obs[..., 6:18]
    dq = privileged_obs[..., 18:34]
    actions = privileged_obs[..., 34:50]
    cmd = privileged_obs[..., 50:53] # commands

    w_m = w * w.new_tensor([-1.0, 1.0, -1.0])
    g_m = g * g.new_tensor([1.0, -1.0, 1.0])
    cmd_m = cmd * cmd.new_tensor([1.0, -1.0, -1.0])

    q_leg_m = _mirror_dof_leg(q_leg)
    dq_m = _mirror_dof_all(dq)
    actions_m = _mirror_dof_all(actions)

    base_vel = privileged_obs[..., 53:56] # base_lin_vel
    base_vel_m = base_vel * base_vel.new_tensor([1.0, -1.0, 1.0])

    foot_contact_forces = privileged_obs[..., 56:68] # foot_contact_forces 4*3
    foot_contact_forces_m = foot_contact_forces.view(-1, 4, 3) * torch.tensor([1.0, -1.0, 1.0], device=privileged_obs.device)
    foot_contact_forces_m = foot_contact_forces_m.index_select(dim=1, index=torch.tensor([1, 0, 3, 2], device=privileged_obs.device)) # swap left-right feet])
    foot_contact_forces_m = foot_contact_forces_m.view(-1, 12)

    tau = privileged_obs[..., 68:84] # joint_torques
    tau_m = _mirror_dof_all_orig(tau)

    nx, ny = _get_height_grid_shape()
    heights = privileged_obs[..., 84:]

    # heights: reshape (nx, ny) and flip along y
    h = heights.view(*heights.shape[:-1], nx, ny)
    h = torch.flip(h, dims=[-1])
    heights_m = h.reshape(*heights.shape)

    return torch.cat(
        (w_m, 
         g_m, 
         q_leg_m, 
         dq_m, 
         actions_m, 
         cmd_m, 
         base_vel_m, 
         foot_contact_forces_m, 
         tau_m, 
         heights_m),
        dim=-1,
    )


def go2w_symmetry(
    *,
    obs: Optional[torch.Tensor],
    actions: Optional[torch.Tensor],
    # env,
    obs_type: str,
):
    """
    rsl_rl symmetry data_augmentation_func.

    Returns augmented batch: concat([orig, mirrored]) for whichever tensor is provided.
    - When obs is provided: returns (obs_aug, actions_aug_or_None)
    - When only actions is provided: returns (None, actions_aug)
    """
    obs_aug = None
    actions_aug = None
    skip_mask = None

    if obs is not None:
        if obs_type == "obs":
            mirrored = _mirror_obs(obs)
            skip_mask = _should_skip_mirroring_from_cmd(obs[..., 50:53])
            mirrored = torch.where(skip_mask.unsqueeze(-1), obs, mirrored)
        elif obs_type == "obs_hist":
            if obs.shape[-1] % _POLICY_OBS_DIM != 0:
                raise ValueError(f"obs_history last dim must be multiple of {_POLICY_OBS_DIM}, got {obs.shape[-1]}")
            h = obs.shape[-1] // _POLICY_OBS_DIM
            x = obs.view(obs.shape[0], h, _POLICY_OBS_DIM)
            x_m = _mirror_obs(x.reshape(-1, _POLICY_OBS_DIM)).view(obs.shape[0], h, _POLICY_OBS_DIM)
            mirrored = x_m.reshape(obs.shape[0], h * _POLICY_OBS_DIM)

            # 用最后一帧的 cmd 判定是否跳过镜像
            cmd_ = x[:, -1, 50:53]
            skip_mask = _should_skip_mirroring_from_cmd(cmd_)
            mirrored = torch.where(skip_mask.unsqueeze(-1), obs, mirrored)
        elif obs_type == "privileged_obs":
            mirrored = _mirror_privileged_obs(obs)
            skip_mask = _should_skip_mirroring_from_cmd(obs[..., 50:53])
            mirrored =  torch.where(skip_mask.unsqueeze(-1), obs, mirrored)
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")
    else:
        mirrored = None
        # obs_aug = torch.cat([obs, mirrored], dim=0)

    if actions is not None:
        if actions.shape[-1] != _NUM_ACTIONS:
            raise ValueError(f"actions dim should be {_NUM_ACTIONS}, got {actions.shape[-1]}")
        
        actions_m = _mirror_dof_all(actions)
        if skip_mask is not None:
            actions_m = torch.where(skip_mask.unsqueeze(-1), actions, actions_m)
    else:
        actions_m = None
        # actions_aug = torch.cat([actions, actions_m], dim=0)

    return mirrored, actions_m