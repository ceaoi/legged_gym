
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.go2w.go2w_config import GO2wRoughCfg, GO2wRoughCfgPPO

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import random
from legged_gym.utils.math import quat_apply_yaw

class GO2wRobot(LeggedRobot):
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
        """
        obs_dim = self.obs_buf.shape[1] if len(self.obs_buf.shape) > 1 else self.obs_buf[0].numel()
        noise_vec = torch.zeros(obs_dim, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:6+self.num_actions_leg] = noise_scales.dof_pos_leg * noise_level * self.obs_scales.dof_pos_leg
        noise_vec[6+self.num_actions_leg:6+2*self.num_actions_leg] = noise_scales.dof_vel_leg * noise_level * self.obs_scales.dof_vel_leg
        noise_vec[6+2*self.num_actions_leg:6+2*self.num_actions_leg + self.num_actions_wheel] = noise_scales.dof_vel_wheel * noise_level * self.obs_scales.dof_vel_wheel
        noise_vec[6+2*self.num_actions_leg + self.num_actions_wheel:] = 0.  # previous actions + commands have no noise
        return noise_vec

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        resample_nums = len(env_ids)
        ten_percent_nums = resample_nums // 10
        env_list = list(range(resample_nums))
        half_env_list = random.sample(env_list, resample_nums // 2)
        rest_env_list = list(set(env_list) - set(half_env_list))
        # 50% 直走
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat[env_ids[half_env_list]], \
                                 self.forward_vec[env_ids[half_env_list]])
            heading = torch.atan2(forward[:,1], forward[:,0])
            self.commands[env_ids[half_env_list], 3] = heading
        else:
            self.commands[env_ids[half_env_list], 2] = 0.0
        # 其中 10% 沿着x轴正方向走
        ten_percent_env_list = random.sample(half_env_list, ten_percent_nums)
        self.commands[env_ids[ten_percent_env_list], 0] = 0.0
        half_env_list = list(set(half_env_list) - set(ten_percent_env_list))
        # 其中 10% 沿着y轴正方向走
        ten_percent_env_list = random.sample(half_env_list, ten_percent_nums)
        self.commands[env_ids[ten_percent_env_list], 1] = 0.0
        half_env_list = list(set(half_env_list) - set(ten_percent_env_list))
        # # 其中 0.1% 静止不动
        # ten_percent_env_list = random.sample(half_env_list, ten_percent_nums // 100)
        # self.commands[env_ids[ten_percent_env_list], :2] = 0.0

        # 另 10% 纯转弯
        ten_percent_env_list = random.sample(rest_env_list, ten_percent_nums)
        self.commands[env_ids[ten_percent_env_list], :2] = 0.0

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_lin_vel).unsqueeze(1)

    def compute_observations(self):
        """ Computes observations
        """
        feet_contact_forces = self.contact_forces[:, self.feet_indices, :].view(-1, 3) # num_envs x 4 x 3
        _, _, yaw = get_euler_xyz(self.base_quat)
        quat_yaw = quat_from_euler_xyz(torch.zeros_like(yaw),
                                    torch.zeros_like(yaw),
                                    yaw).unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
        feet_contact_forces = quat_rotate_inverse(quat_yaw, feet_contact_forces).view(self.num_envs, -1)


        idx_imu = (self._obs_imu_latency_head - self.obs_imu_latency_simstep) % self.obs_imu_latency_ring.shape[1]
        idx_motor = (self._obs_motor_latency_head - self.obs_motor_latency_simstep) % self.obs_motor_latency_ring.shape[1]
        self.obs_buf = torch.cat((  self.obs_imu_latency_ring[self._all_envs_ids, idx_imu], # base角速度 3 + 重力 3
                                    self.obs_motor_latency_ring[self._all_envs_ids, idx_motor], # 关节位置速度 12+12 + 轮子速度 4
                                    self.actions, # 上一步的动作 16
                                    self.commands[:, :3] * self.commands_scale, # 运动指令 xy+yaw 3
                                    self.gait_phase.unsqueeze(1), # gait phase 1
                                    self.gait_clock, # gait clock for each leg 4
                                    ),dim=-1)
        
        self.privileged_obs_buf = torch.cat((   self.obs_imu_latency_ring[self._all_envs_ids, self._obs_imu_latency_head], # base角速度 3 + 重力 3
                                                self.obs_motor_latency_ring[self._all_envs_ids, self._obs_motor_latency_head], # 关节位置速度 12+12 + 轮子速度 4
                                                self.actions, # 上一步的动作 16
                                                self.commands[:, :3] * self.commands_scale, # 运动指令 xy+yaw 3
                                                self.gait_phase.unsqueeze(1), # gait phase 1
                                                self.gait_clock, # gait clock for each leg 4
                                                self.base_lin_vel * self.obs_scales.lin_vel, # base线速度 3
                                                feet_contact_forces * self.obs_scales.contact_forces, # 足端接触力 4*3
                                                self.torques * self.obs_scales.torques, # 关节力矩 16
                                            ), dim=-1)
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        # print("feet contact forces:", self.contact_forces[0, self.feet_indices, :])
        # print("torques shape:", self.torques.shape)
        if self.cfg.terrain.measure_heights:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _compute_torques(self, actions):
        # actions：0：11 position for leg joints, 12：15 velocity for wheel joints
        pos_action = actions[:, 0:12] * self.cfg.control.action_scale_pos
        vel_action = actions[:, 12:16]* self.cfg.control.action_scale_vel

        pos_torques = \
            self.pLeg_gains*(pos_action + self.default_dof_pos_leg + self.motor_zero_offsets_leg - self.dof_pos_leg) \
            - self.dLeg_gains*self.dof_vel_leg
        vel_torques = self.pWheel_gains*(vel_action - self.dof_vel_wheel) - self.dWheel_gains*(self.dof_vel_wheel - self.last_dof_vel_wheel)/self.sim_params.dt
        
        # pos_torques = self.pd_control(
        #     pos_action + self.default_dof_pos_leg + self.motor_zero_offsets_leg - self.dof_pos_leg,
        #     -self.dof_vel_leg,
        #     self.pLeg_gains,
        #     self.dLeg_gains
        #     )
        # vel_torques = self.pd_control(
        #     vel_action - self.dof_vel_wheel,
        #     -(self.dof_vel_wheel - self.last_dof_vel_wheel)/self.sim_params.dt,
        #     self.pWheel_gains,
        #     self.dWheel_gains
        #     )
        # print("pWheel_gains:", self.pWheel_gains[:4, :])
        # print("dWheel_gains:", self.dWheel_gains[:4, :])
            
        torques = torch.cat(
                        (
                            pos_torques[:, 0:3], vel_torques[:, 0].view(self.num_envs, 1),
                            pos_torques[:, 3:6], vel_torques[:, 1].view(self.num_envs, 1),
                            pos_torques[:, 6:9], vel_torques[:, 2].view(self.num_envs, 1),
                            pos_torques[:, 9:12], vel_torques[:, 3].view(self.num_envs, 1),
                        ),
                        axis=1,
                    )
        return torch.clip(torques, -self.torque_limits, self.torque_limits)    
    
    def pd_control(self, err, err_dot, kp, kd):
        return kp * err + kd * err_dot
    
    ########################################################## reward functions ##########################################################
    
    def _reward_dof_pos_leg_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos_leg - self.dof_pos_leg_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos_leg - self.dof_pos_leg_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos_leg - self.default_dof_pos_leg), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    def _reward_dof_acc_leg(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel[:, self.dof_idx_leg] - self.dof_vel[:, self.dof_idx_leg]) / self.dt), dim=1)
    
    def _reward_dof_acc_wheel(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel[:, self.dof_idx_wheel] - self.dof_vel[:, self.dof_idx_wheel]) / self.dt), dim=1)
    
    def _reward_action_rate_leg(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:, 0:12] - self.actions[:, 0:12]), dim=1)
    
    def _reward_action_rate_wheel(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:, 12:16] - self.actions[:, 12:16]), dim=1)
    
    def _reward_action_smoothness(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions[:, 0:12] - self.last_last_actions[:, 0:12] - 2 * self.last_actions[:, 0:12]), dim=1)

    def _reward_base_height(self):
        # 惩罚偏离目标高度的base高度,范围 -1~0,|误差|<0.1时, 惩罚 < 0.05
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        error = base_height - self.cfg.rewards.base_height_target
        return self._reward_bell_shaped_function(error, w=0.2, s=4) - 1
    
    def _reward_wheel_air(self):
        # 当无y or yaw 命令时，惩罚轮子离地
        isAir = self.contact_forces[:, self.feet_indices, 2] < 1.
        isNoYorYawCmd = torch.norm(self.commands[:, 1:3], dim=1) < 0.1
        rew_Air = torch.zeros_like(self.commands[:, 1])
        rew_Air[isNoYorYawCmd] = torch.sum(isAir[isNoYorYawCmd].float(), dim=1)
        return rew_Air
    
    def _reward_encourage_wheel_for_x(self):
        # 惩罚不使用wheel来满足xVel指令
        xVelCmd = self.commands[:, 0]
        xVelCmd = xVelCmd.unsqueeze(1)
        wheelVel = self.dof_vel_wheel * self.cfg.asset.wheelRadius        
        # 为鼓励使用轮子，不惩罚过大的轮子速度误差
        posCmdIdx = (xVelCmd >= 0).squeeze()
        negCmdIdx = (xVelCmd <= 0).squeeze()
        wheelVel[posCmdIdx] = torch.min(wheelVel, xVelCmd)[posCmdIdx]
        wheelVel[negCmdIdx] = torch.max(wheelVel, xVelCmd)[negCmdIdx]
        error = torch.abs(xVelCmd - wheelVel)
        errorAbsMean = torch.mean(error, dim=1)
        # print("vVelCmd:", xVelCmd[1, 0])
        print("wheelVel:", wheelVel[0, :])
        print("--------------------------------------------")
        return errorAbsMean
    
    def _reward_joint_power(self):
        # Penalize joint power consumption
        joint_powers = torch.abs(self.torques * self.dof_vel)
        return torch.sum(joint_powers, dim=1)
    
    def _reward_same_wheel_x_position(self):
        # 惩罚前后轮子x位置不一致
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_x_position_err = torch.abs(foot_positions_base[:,0,0] - foot_positions_base[:,1,0]) \
                                + torch.abs(foot_positions_base[:,2,0] - foot_positions_base[:,3,0])
        # print("foot_position:", foot_positions_base[0, :, 0])
        return foot_x_position_err
    
    def _reward_feet_stumble(self): # -1~0
        # contact_forces: [N, num_bodies, 3] -> feet: [N, F, 3]
        forces = self.contact_forces[:, self.feet_indices, :]

        z_sum = forces[..., 2].square().sum(dim=-1)   # [N]
        xyz_sum = forces.square().sum(dim=-1).sum(dim=-1)           # [N]

        ratio = z_sum / xyz_sum.clamp_min(1e-12)                   # [N], in [0,1]
        return 1.0 - ratio
    
    def _reward_tracking_gait(self):
        foot_forces_z = self.contact_forces[:, self.feet_indices, 2]
        # self.gait_clock > 0 期望支撑状态
        desired = (self.gait_clock > 0.0).float()
        reward_stance = desired * (1 - torch.exp(-1 * foot_forces_z**2 / 50.0))
        reward_swing = (1 - desired) * torch.exp(-1 * foot_forces_z**2 / 50.0)
        return torch.sum(reward_stance + reward_swing, dim=1) / 4.0
    
    def _reward_bell_shaped_function(self, error, w, s):
        return 1 / (1 + torch.pow(error / w, 2*s))