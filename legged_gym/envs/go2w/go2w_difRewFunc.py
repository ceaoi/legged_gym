
from legged_gym.envs.go2w.go2w_env import GO2wRobot
from legged_gym.envs.go2w.go2w_config import GO2wRoughCfg

from isaacgym.torch_utils import *
import torch
import random

class GO2wCfg_difRewFunc( GO2wRoughCfg ):
    class rewards():
        base_height_target = 0.38 # 0.391.2
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        class scales():
            # 运动跟踪
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            # 稳定性
            base_height = 1.0
            orientation = 1.0
            lin_vel_z = 1.0
            ang_vel_xy = 1.0
            # 动作和平滑性
            dof_vel_leg = 1.0
            # dof_acc_leg = 1.0
            # dof_acc_wheel = 1.0
            action_rate_leg = 1.0
            action_rate_wheel = 1.0
            joint_power = 1.0
            # 特定奖励
            feet_stumble = 1.0

class GO2wRobot_difRewFunc(GO2wRobot):
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0. # step total reward for RL

        # 运动跟踪
        tracking_cmd_xy = self._reward_tracking_lin_vel()
        tracking_cmd_yaw = self._reward_tracking_ang_vel()
        # 稳定性
        height = self._reward_base_height()
        orientation = self._reward_orientation()
        lin_vel_z = self._reward_lin_vel_z()
        ang_vel_xy = self._reward_ang_vel_xy()
        # 动作和平滑性
        dof_vel_leg = self._reward_dof_vel_leg()
        # dof_acc_leg = self._reward_dof_acc_leg() * self.reward_scales["dof_acc_leg"]
        # dof_acc_wheel = self._reward_dof_acc_wheel() * self.reward_scales["dof_acc_wheel"]
        action_rate_leg = self._reward_action_rate_leg()
        action_rate_wheel = self._reward_action_rate_wheel()
        joint_power = self._reward_joint_power()
        # 特定奖励
        stumble = self._reward_feet_stumble()

        self.rew_buf[:] = 0.99 * (tracking_cmd_xy * 0.6 + tracking_cmd_yaw * 0.4) * (
                9.0 + height * 0.5 + lin_vel_z * 0.4 + ang_vel_xy * 0.01 + orientation * 0.07
            ) / 10.0 + (dof_vel_leg + action_rate_leg + action_rate_wheel + joint_power * 0.01) * 0.01

        self.episode_sums["tracking_lin_vel"] += tracking_cmd_xy * self.reward_scales["tracking_lin_vel"]
        self.episode_sums["tracking_ang_vel"] += tracking_cmd_yaw * self.reward_scales["tracking_ang_vel"]
        self.episode_sums["base_height"] += height * self.reward_scales["base_height"]
        self.episode_sums["orientation"] += orientation * self.reward_scales["orientation"]
        self.episode_sums["lin_vel_z"] += lin_vel_z * self.reward_scales["lin_vel_z"]
        self.episode_sums["ang_vel_xy"] += ang_vel_xy * self.reward_scales["ang_vel_xy"]
        self.episode_sums["dof_vel_leg"] += dof_vel_leg * self.reward_scales["dof_vel_leg"]
        # self.episode_sums["dof_acc_leg"] += dof_acc_leg
        # self.episode_sums["dof_acc_wheel"] += dof_acc_wheel
        self.episode_sums["action_rate_leg"] += action_rate_leg * self.reward_scales["action_rate_leg"]
        self.episode_sums["action_rate_wheel"] += action_rate_wheel * self.reward_scales["action_rate_wheel"]
        self.episode_sums["joint_power"] += joint_power * self.reward_scales["joint_power"]
        self.episode_sums["feet_stumble"] += stumble * self.reward_scales["feet_stumble"]
        self.episode_sums["total_rewards"] += self.rew_buf * self.dt

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        self.reward_scales["total_rewards"] = 1.0  # for logging total rewards
        # prepare list of functions
        # self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            # self.reward_functions.append(getattr(self, name)) # 根据字符串 name，在对象 self 上动态获取同名属性

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        pass
    
    
    # 运动跟踪
    def _reward_tracking_lin_vel(self): # 0~1
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self): # 0~1
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    # def _reward_tracking_lin_vel(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     cmd_xy = self.commands[:, :2]
    #     vel_xy = self.base_lin_vel[:, :2]

    #     denom = cmd_xy.abs().clamp_min(9e-3)
    #     lin_vel_error = ((vel_xy - cmd_xy) / denom).square().sum(1)
    #     return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    # def _reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw)
    #     cmd_ang_vel = self.commands[:, 2]
    #     ang_vel = self.base_ang_vel[:, 2]

    #     denom = cmd_ang_vel.abs().clamp_min(9e-3)
    #     ang_vel_error = ((ang_vel - cmd_ang_vel) / denom).square()
    #     return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    # 稳定性
    def _reward_base_height(self): # 0~1
        # 惩罚偏离目标高度的base高度,范围 -1~0,|误差|<0.1时, 惩罚 < 0.05
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        error = base_height - self.cfg.rewards.base_height_target
        return self._bell_shaped_func(error, w=0.2, s=4)

    def _reward_orientation(self): # 0~1
        # Penalize non flat base orientation
        rew1 = self._cheap_func(self.projected_gravity[:, 0], k=100)
        rew2 = self._cheap_func(self.projected_gravity[:, 1], k=100)
        return (rew1 + rew2) / 2.0

    def _reward_lin_vel_z(self): # 0~1
        # Penalize z axis base linear velocity
        return self._cheap_func(self.base_lin_vel[:, 2], k=20)
    
    def _reward_ang_vel_xy(self): # 0~1
        # Penalize xy axes base angular velocity
        rew1 = self._cheap_func(self.base_ang_vel[:, 0], k=2)
        rew2 = self._cheap_func(self.base_ang_vel[:, 1], k=2)
        return (rew1 + rew2) / 2.0
    
    # 动作和平滑性
    def _reward_dof_vel_leg(self): # 0~1
        # Penalize dof velocities
        x = torch.mean(torch.square(self.dof_vel[:, self.dof_idx_leg]), dim=1)
        return 1 / (1 + 0.5 * x)

    def _reward_dof_acc_leg(self): # 0~1
        # Penalize dof accelerations
        x = torch.mean(torch.square((self.last_dof_vel[:, self.dof_idx_leg] - self.dof_vel[:, self.dof_idx_leg]) / self.dt), dim=1)
        return 1 / (1 + 0.05 * x)
    
    def _reward_dof_acc_wheel(self): # 0~1
        # Penalize dof accelerations
        x = torch.mean(torch.square((self.last_dof_vel[:, self.dof_idx_wheel] - self.dof_vel[:, self.dof_idx_wheel]) / self.dt), dim=1)
        return 1 / (1 + 0.05 * x)
    
    def _reward_action_rate_leg(self): # 0~1
        # Penalize changes in actions
        x = torch.mean(torch.square(self.last_actions[:, 0:12] - self.actions[:, 0:12]), dim=1)
        return 1 / (1 + 0.5 * x)
    
    def _reward_action_rate_wheel(self): # 0~1
        # Penalize changes in actions
        x = torch.mean(torch.square(self.last_actions[:, 12:16] - self.actions[:, 12:16]), dim=1)
        return 1 / (1 + 0.5 * x)
    
    def _reward_joint_power(self): # 0~1
        # Penalize joint power consumption
        x = torch.mean(torch.square(self.torques * self.dof_vel), dim=1)
        return 1 / (1 + 0.5 * x)
    
    # 特定奖励
    def _reward_feet_stumble(self): # 0~1
        # contact_forces: [N, num_bodies, 3] -> feet: [N, F, 3]
        forces = self.contact_forces[:, self.feet_indices, :]

        xy_sum = forces[..., :2].square().sum(dim=-1).sum(dim=-1)   # [N]
        xyz_sum = forces.square().sum(dim=-1).sum(dim=-1)           # [N]

        ratio = xy_sum / xyz_sum.clamp_min(1e-12)                   # [N], in [0,1]
        return 1.0 - ratio
    
    def _bell_shaped_func(self, error, w, s):
        return 1 / (1 + torch.pow(error / w, 2*s))
    
    def _cheap_func(self, error, k):
        return 1 / (1 + k * torch.pow(error, 2))