from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # 新增 obs_history_buf， 放 step 后，hist 包含最新的 obs
        # self.obs_hist_buf = self.obs_hist_buf[:,self.num_obs:]
        # self.obs_hist_buf = torch.cat((self.obs_hist_buf,self.obs_buf),dim = -1)

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # 新增的延迟buffer
        self.actions = self.update_cmd_action_latency_buffer()
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation): # 高频
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.post_sim_step()
            self.update_obs_latency_buffer()
        self.post_physics_step()
        if self.cfg.rewards.isGaitInput:
            self.update_gait_phase()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # 新增 obs_history_buf (oldest -> newest)
        # self.obs_hist_buf = self.obs_hist_buf[:,self.num_obs:]
        # self.obs_hist_buf = torch.cat((self.obs_hist_buf,self.obs_buf),dim = -1)
        self.update_obs_hist_buf()

        self.extras["obs_hist_buf"] = self.obs_hist_buf
        self.extras["episode_length"] = self.episode_length_buf

        self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.extras["privileged_obs_buf"] = self.privileged_obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def update_obs_hist_buf(self): # (oldest -> newest)
        self._obs_hist_ring[:, self._obs_hist_head].copy_(self.obs_buf)
        self._obs_hist_head = (self._obs_hist_head + 1) % self.num_obs_hist

        if self._obs_hist_head == 0:
            # ring 中本来就是 [oldest ... newest]
            self._obs_hist_buf3.copy_(self._obs_hist_ring)
        else:
            # 第一段：ring[head : H] -> hist[0 : H-head]
            self._obs_hist_buf3[:, :self.num_obs_hist-self._obs_hist_head].copy_(self._obs_hist_ring[:, self._obs_hist_head:])
            # 第二段：ring[0 : head] -> hist[H-head : H]
            self._obs_hist_buf3[:, self.num_obs_hist-self._obs_hist_head:].copy_(self._obs_hist_ring[:, :self._obs_hist_head])
        # check
        # print("++++++++++++++++++++check obs_hist_buf+++++++++++++++++++")
        # assert torch.allclose( # 如果 torch.allclose = False，立刻抛出 AssertionError 并中断程序
        #     self._obs_hist_buf3[:, -1],
        #     self.obs_buf,
        #     atol=1e-6
        # )
    
    def post_sim_step(self): # 高频
        self.last_dof_vel_wheel[:] = self.dof_vel_wheel[:] # [:]是切片赋值，避免指针引用

        self.dof_pos_leg.copy_(self.dof_pos[:, self.dof_idx_leg])
        self.dof_vel_leg.copy_(self.dof_vel[:, self.dof_idx_leg])
        self.dof_vel_wheel.copy_(self.dof_vel[:, self.dof_idx_wheel])


        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
    
    def post_physics_step(self): # 低频
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """

        # self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self._reset_obs_history_buffer(env_ids) # 重置 obs history buffer
        
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        self.foot_positions = self.rigid_body_state[:, self.feet_indices, 0:3]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    # def compute_feet_state(self):
    #     rb_state = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)
    #     self.feet_state = rb_state[:, self.feet_indices, :] 
    #     self.foot_positions = self.feet_state[:, :, 0:3]
    #     self.foot_quat = self.feet_state[:, :, 3:7]
    #     self.foot_velocities_sim = self.feet_state[:, :, 7:10]
    #     self.foot_ang_vel_sim = self.feet_state[:, :, 10:13]
    #     # self.foot_positions / self.foot_quat / self.foot_velocities_sim are views refreshed by Isaac Gym
    #     foot_velocities = (self.foot_positions - self.last_foot_positions) / self.dt

    #     tmp_foot_pos = self.foot_positions - self.root_states[:, :3].unsqueeze(1)
    #     # foot_pos_in_base = quat_rotate_inverse(self.base_quat.unsqueeze(1), tmp_foot_pos)

    #     forward = quat_apply(self.base_quat, self.forward_vec)
    #     heading = torch.atan2(forward[:, 1], forward[:, 0])
    #     for i in range(len(self.feet_indices)):
    #         self.foot_ang_vel[:, i] = quat_rotate_inverse(
    #             self.foot_quat[:, i], self.foot_ang_vel[:, i]
    #         )
    #         self.foot_velocities[:, i] = quat_rotate_inverse(
    #             quat_from_euler_xyz(heading * 0, heading * 0, heading),
    #             foot_velocities[:, i],
    #         )
    #         self.foot_acc[:, i] = (
    #             self.foot_velocities[:, i] - self.last_foot_velocities[:, i]
    #         ) / self.dt
    #         self.foot_velocities_f[:, i] = quat_rotate_inverse(
    #             self.foot_quat[:, i], foot_velocities[:, i]
    #         )
    #         self.foot_positions_base[:, i] = quat_rotate_inverse(self.base_quat, tmp_foot_pos[:, i])
    #     self.foot_heights = torch.clip(self.foot_positions[:, :, 2] - self._get_foot_heights(), 0, 1)

    #     # print(self.foot_positions_base)
    #     self.last_foot_positions[:] = self.foot_positions
    #     self.last_foot_velocities[:] = self.foot_velocities

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        #+
        self.last_dof_vel_wheel[env_ids] = 0.
        self.dof_pos_leg[env_ids] = self.dof_pos[env_ids][:, self.dof_idx_leg]
        self.dof_vel_leg[env_ids] = self.dof_vel[env_ids][:, self.dof_idx_leg]
        self.dof_vel_wheel[env_ids] = self.dof_vel[env_ids][:, self.dof_idx_wheel]

        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        #+

        self._reset_latency_buffer(env_ids) # reset latency buffer 并 随机化

        # 随机化
        self._randomize_motor_params(env_ids)
        # print("Randomized motor params for envs:", self.dWheel_gains[env_ids[0]])

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_lin"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["max_command_ang"] = self.command_ranges["ang_vel_yaw"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # 新增 gait phase reset
        if self.cfg.rewards.isGaitInput:
            self.gait_phase[env_ids] = 0.0
            self.stop_timer[env_ids] = 0.0
            self.gait_offsets[env_ids] = 0.0
            self.gait_clock[env_ids] = 0.0
            self.last_should_run_clock[env_ids] = False
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0. # step total reward for RL
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # print("obs_buf shape:", self.obs_buf.shape)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            # print(f"heights: {heights[:20]}")
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # print("obs_buf shape:", self.obs_buf.shape)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id): # 此处进行随机化
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0: # thus only work once
            self.dof_pos_leg_limits = torch.zeros(self.num_actions_leg, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                if i in self.dof_idx_leg:
                    idxLeg = i
                    if i > 3: # skip foot joints
                        idxLeg -= (i // 4)
                    self.dof_pos_leg_limits[idxLeg, 0] = props["lower"][i].item()
                    self.dof_pos_leg_limits[idxLeg, 1] = props["upper"][i].item()
                    # soft limits
                    m = (self.dof_pos_leg_limits[idxLeg, 0] + self.dof_pos_leg_limits[idxLeg, 1]) / 2
                    r = self.dof_pos_leg_limits[idxLeg, 1] - self.dof_pos_leg_limits[idxLeg, 0]
                    self.dof_pos_leg_limits[idxLeg, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                    self.dof_pos_leg_limits[idxLeg, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()

        #randomization of pd gains
        # if self.cfg.domain_rand.randomize_pd_gains:
        #     self.p_gains_multiplier[env_id, :] = torch_rand_float(self.cfg.domain_rand.stiffness_multiplier_range[0], self.cfg.domain_rand.stiffness_multiplier_range[1], (1,self.num_actions), device=self.device)
        #     self.d_gains_multiplier[env_id, :] =  torch_rand_float(self.cfg.domain_rand.damping_multiplier_range[0], self.cfg.domain_rand.damping_multiplier_range[1], (1,self.num_actions), device=self.device)   
        
        # randomization of the motor zero calibration for real machine
        # if self.cfg.domain_rand.randomize_motor_zero_offset:
        #     self.motor_zero_offsets[env_id, :] = torch_rand_float(self.cfg.domain_rand.motor_zero_offset_range[0], self.cfg.domain_rand.motor_zero_offset_range[1], (1,self.num_actions), device=self.device)

        # randomization of the motor properties
        randomize_cfg = [
            ("friction", "randomize_joint_friction", "joint_friction_range", "set"),
            ("stiffness", "randomize_joint_stiffness", "joint_stiffness_range", "set"),
            ("damping", "randomize_joint_damping", "joint_damping_range", "set"),
            ("armature", "randomize_joint_armature", "joint_armature_range", "set"),
        ]

        for i in range(len(props)):
            for attr, flag, rng_name, mode in randomize_cfg:
                # attr 参数名、 flag 是否随机化标志、rng_name 范围名称、 mode 乘法or赋值
                if getattr(self.cfg.domain_rand, flag, False):
                    rng = getattr(self.cfg.domain_rand, rng_name)
                    rnd = np.random.uniform(rng[0], rng[1])
                    if mode == "mul":
                        props[attr][i] *= rnd
                    elif mode == "set":
                        props[attr][i] = rnd
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        enforce_sym = getattr(self.cfg.domain_rand, "enforce_left_right_symmetry", False)

        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])

        # randomize link masses
        if self.cfg.domain_rand.randomize_link_mass:
            multiplied_link_masses_ratio = self.cfg.domain_rand.multiplied_link_mass_range
    
            for i in range(1, len(props)):
                props[i].mass *= np.random.uniform(multiplied_link_masses_ratio[0], multiplied_link_masses_ratio[1])

            if self.cfg.domain_rand.enforce_left_right_symmetry:
                for idxL, idxR in zip(self.cfg.domain_rand.left_body_index, self.cfg.domain_rand.right_body_index):
                    props[idxR].mass = props[idxL].mass

        if self.cfg.domain_rand.randomize_base_com:
            rng = self.cfg.domain_rand.base_com_range
            rand_vec = np.random.uniform(rng[0], rng[1], size=3)
            if enforce_sym:
                rand_vec[1] = 0.0
            props[0].com.x += rand_vec[0]
            props[0].com.y += rand_vec[1]
            props[0].com.z += rand_vec[2]

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        # 新增不动 push
        if self.cfg.domain_rand.push_robots_still & (self.extras["episode"]["max_command_lin"] < self.cfg.commands.ranges.lin_vel_x[1] + 0.2):
            cmd_active = self.commands[:, :2].square().sum(-1) > 1e-6
            is_static = self.base_lin_vel[:, :2].square().sum(-1) < 1e-2
            push_env_ids = (self.episode_length_buf % self.cfg.domain_rand.push_if_still_interval == 0) & cmd_active & is_static
            self._push_robots(push_env_ids.nonzero(as_tuple=False).flatten())
            # print("pushed still robots:", push_env_ids.nonzero(as_tuple=False).flatten())


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

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        p_gains = self.p_gains * self.p_gains_multiplier
        d_gains = self.d_gains * self.d_gains_multiplier

        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_zero_offsets) - d_gains*self.dof_vel
        elif control_type=="V":
            torques = p_gains*(actions_scaled - self.dof_vel) - d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self, env_ids = None):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        if env_ids is None:
            self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        else:
            # self.root_states[env_ids, 7:9] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 2), device=self.device) # lin vel x/y
            scale = torch_rand_float(0.5, 1.5, (len(env_ids), 1), device=self.device).squeeze(1)
            self.root_states[env_ids, 7] = self.commands[env_ids, 0] * scale
            self.root_states[env_ids, 8] = self.commands[env_ids, 1] * scale
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_states),
                                                         gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if len(env_ids) == 0:
            return
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        # if (self.command_ranges["lin_vel_x"][1] == self.cfg.commands.max_curriculum_x):
        #     pass
        if (torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 1.0 * self.reward_scales["tracking_lin_vel"]):
            if (self.command_ranges["lin_vel_x"][1] < self.cfg.commands.max_curriculum_x) :
                self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.001, -self.cfg.commands.max_curriculum_x, 0.)
                self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.001, 0., self.cfg.commands.max_curriculum_x)
                
                self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.001, -self.cfg.commands.max_curriculum_y, 0.)
                self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.001, 0., self.cfg.commands.max_curriculum_y)
        # elif (torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length < 0.5 * self.reward_scales["tracking_lin_vel"]):
        #     if (self.command_ranges["lin_vel_x"][1] > 0.5):
        #         self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] + 0.1, -self.cfg.commands.max_curriculum_x, 0.)
        #         self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] - 0.1, 0., self.cfg.commands.max_curriculum_x)
                
        #         self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] + 0.1, -self.cfg.commands.max_curriculum_y, 0.)
        #         self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] - 0.1, 0., self.cfg.commands.max_curriculum_y)

        
        if (self.command_ranges["ang_vel_yaw"][1] == self.cfg.commands.max_curriculum_yawRate):
            pass
        elif (torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length > 1.0 * self.reward_scales["tracking_ang_vel"]):
            self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 0.001, -self.cfg.commands.max_curriculum_yawRate, 0.)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 0.001, 0., self.cfg.commands.max_curriculum_yawRate)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_position = self.root_states[:, 0:3] #+
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        #+ 新增足端信息
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, -1)
        self.foot_positions = self.rigid_body_state[:, self.feet_indices, 0:3] # 使用了高级索引，需手动刷新
        self.feet_contact_forces = torch.zeros(self.num_envs, len(self.feet_indices) * 3, dtype=torch.float, device=self.device, requires_grad=False) # 足端接触力

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.extras["privileged_obs_buf"] = self.privileged_obs_buf
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        
        # self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.pLeg_gains_init = torch.zeros(1, self.num_actions_leg, dtype=torch.float, device=self.device, requires_grad=False)
        self.dLeg_gains_init = torch.zeros(1, self.num_actions_leg, dtype=torch.float, device=self.device, requires_grad=False)
        self.pWheel_gains_init = torch.zeros(1, self.num_actions - self.num_actions_leg, dtype=torch.float, device=self.device, requires_grad=False)
        self.dWheel_gains_init = torch.zeros(1, self.num_actions - self.num_actions_leg, dtype=torch.float, device=self.device, requires_grad=False)

        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_vel_wheel = torch.zeros_like(self.dof_vel_wheel) #+ 
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_leg = torch.zeros_like(self. dof_pos_leg[0])
        self.motor_zero_offsets_leg = torch.zeros_like(self.dof_pos_leg)
        idxWheel = 0
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            found = False
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle

            if i in self.dof_idx_leg:
                idxLeg = i
                if i > 3: # skip foot joints
                    idxLeg -= (i // 4)
                self.default_dof_pos_leg[idxLeg] = angle

                if name in self.cfg.control.stiffnessLeg.keys():
                    self.pLeg_gains_init[0, idxLeg] = self.cfg.control.stiffnessLeg[name]
                    self.dLeg_gains_init[0, idxLeg] = self.cfg.control.dampingLeg[name]
                    found = True
            elif name in self.cfg.control.stiffnessWheel.keys():
                self.pWheel_gains_init[0, idxWheel] = self.cfg.control.stiffnessWheel[name]
                self.dWheel_gains_init[0, idxWheel] = self.cfg.control.dampingWheel[name]
                idxWheel +=1
                found = True

            if not found:
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

        self.default_dof_pos_leg = self.default_dof_pos_leg.unsqueeze(0) # 维度从(num_dof)变成(1, num_dof)
        self.pLeg_gains = self.pLeg_gains_init.repeat(self.num_envs, 1).clone()
        self.dLeg_gains = self.dLeg_gains_init.repeat(self.num_envs, 1).clone()
        self.pWheel_gains = self.pWheel_gains_init.repeat(self.num_envs, 1).clone()
        self.dWheel_gains = self.dWheel_gains_init.repeat(self.num_envs, 1).clone()

        # gait phase buffer
        # 0~1 的循环相位
        self.gait_phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.stop_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.gait_offsets = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.gait_clock = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.last_should_run_clock = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gait_offset_leftward = torch.tensor(self.cfg.rewards.gait_offset_leftward, device=self.device, dtype=torch.float).unsqueeze(0).expand(self.num_envs, -1)
        self.gait_offset_rightward = torch.tensor(self.cfg.rewards.gait_offset_rightward, device=self.device, dtype=torch.float).unsqueeze(0).expand(self.num_envs, -1)

        # terrain check
        self.is_in_terrain = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

    # 新增 延迟buf
    def update_cmd_action_latency_buffer(self): # old -> new
        actions = self.actions
        if not self.cfg.domain_rand.add_cmd_action_latency:
            return actions
        
        # 更新 head
        self._cmd_action_latency_head = (self._cmd_action_latency_head + 1) % self.cmd_action_latency_ring.shape[1]
        
        self.cmd_action_latency_ring[:, self._cmd_action_latency_head].copy_(actions)
        # 读取每个 env 的延迟动作
        idx = (self._cmd_action_latency_head - self.cmd_action_latency_simstep) % self.cmd_action_latency_ring.shape[1]  # [N]
        action_delayed = self.cmd_action_latency_ring[self._all_envs_ids, idx]

        return action_delayed
    
    def update_obs_latency_buffer(self):
        if self.cfg.domain_rand.randomize_obs_motor_latency:
            # 更新 head
            self._obs_motor_latency_head = (self._obs_motor_latency_head + 1) % self.obs_motor_latency_ring.shape[1]

            obs_motor = torch.cat((
                (self.dof_pos_leg - self.default_dof_pos_leg) * self.obs_scales.dof_pos_leg, 
                self.dof_vel_leg * self.obs_scales.dof_vel_leg,
                self.dof_vel_wheel * self.obs_scales.dof_vel_wheel
            ), dim=1)
            self.obs_motor_latency_ring[:, self._obs_motor_latency_head].copy_(obs_motor)
        else:
            ValueError("Not implemented obs motor latency false!")

        if self.cfg.domain_rand.randomize_obs_imu_latency:
            # 更新 head
            self._obs_imu_latency_head = (self._obs_imu_latency_head + 1) % self.obs_imu_latency_ring.shape[1]

            obs_imu = torch.cat((
                self.base_ang_vel * self.obs_scales.ang_vel, 
                self.projected_gravity
                ), dim=1)            
            self.obs_imu_latency_ring[:, self._obs_imu_latency_head].copy_(obs_imu)
        else:
            ValueError("Not implemented obs latency false!")
    
    def _reset_latency_buffer(self, env_ids):
        if self.cfg.domain_rand.add_cmd_action_latency:
            cur_a = self.actions[env_ids].clone()
            T_a = self.cfg.domain_rand.range_cmd_action_latency[1] + 1
            self.cmd_action_latency_ring[env_ids].copy_(
                cur_a.unsqueeze(1).expand(-1, T_a, -1)              # [num_envs, T, A]
            )
            if self.cfg.domain_rand.randomize_cmd_action_latency:
                self.cmd_action_latency_simstep[env_ids] = torch.randint(
                    self.cfg.domain_rand.range_cmd_action_latency[0],
                    self.cfg.domain_rand.range_cmd_action_latency[1] + 1, # 左开右闭，右加1
                    (len(env_ids),), device=self.device
                )
            else:
                self.cmd_action_latency_simstep[env_ids] = self.cfg.domain_rand.range_cmd_action_latency[1]
        
        if self.cfg.domain_rand.add_obs_latency:
            # 设置当前的观测值到延迟buf中
            # q = (self.dof_pos[env_ids] - self.default_dof_pos) * self.obs_scales.dof_pos
            # dq = self.dof_vel[env_ids] * self.obs_scales.dof_vel
            cur_motor = torch.cat((
                (self.dof_pos_leg[env_ids] - self.default_dof_pos_leg) * self.obs_scales.dof_pos_leg, 
                self.dof_vel_leg[env_ids] * self.obs_scales.dof_vel_leg,
                self.dof_vel_wheel[env_ids] * self.obs_scales.dof_vel_wheel
            ), dim=1)
            
            cur_imu = torch.cat((
                self.base_ang_vel[env_ids] * self.obs_scales.ang_vel, 
                self.projected_gravity[env_ids]
            ), dim=1)   

            T_m = self.cfg.domain_rand.range_obs_motor_latency[1] + 1
            if self.cfg.domain_rand.randomize_obs_motor_latency:
                self.obs_motor_latency_ring[env_ids].copy_(
                    cur_motor.unsqueeze(1).expand(-1, T_m, -1)              # [num_envs, T, A]
                )

            T_i = self.cfg.domain_rand.range_obs_imu_latency[1] + 1
            if self.cfg.domain_rand.randomize_obs_imu_latency:
                self.obs_imu_latency_ring[env_ids].copy_(
                    cur_imu.unsqueeze(1).expand(-1, T_i, -1)              # [num_envs, T, A]
                )
            
            # 设置延迟步数
            if self.cfg.domain_rand.randomize_obs_motor_latency:
                self.obs_motor_latency_simstep[env_ids] = torch.randint(
                    self.cfg.domain_rand.range_obs_motor_latency[0],
                    self.cfg.domain_rand.range_obs_motor_latency[1] + 1,
                    (len(env_ids),), device=self.device
                )
            else:
                self.obs_motor_latency_simstep[env_ids] = self.cfg.domain_rand.range_obs_motor_latency[1]

            if self.cfg.domain_rand.randomize_obs_imu_latency:
                self.obs_imu_latency_simstep[env_ids] = torch.randint(
                    self.cfg.domain_rand.range_obs_imu_latency[0],
                    self.cfg.domain_rand.range_obs_imu_latency[1] + 1,
                    (len(env_ids),), device=self.device
                )
            else:
                self.obs_imu_latency_simstep[env_ids] = self.cfg.domain_rand.range_obs_imu_latency[1]

    # 新增 reset obs_history buffer
    def _reset_obs_history_buffer(self, env_ids):
        # 当前 obs: [K, O]
        obs = self.obs_buf[env_ids]
        # 填满 ring: [K, H, O]
        self._obs_hist_ring[env_ids].copy_(
            obs.unsqueeze(1).expand(-1, self.num_obs_hist, -1)
        )
            # # 重置 head（从 0 开始写）
            # self._obs_hist_head = 0
        # 同步导出到扁平 obs_hist_buf
        self._obs_hist_buf3[env_ids].copy_(self._obs_hist_ring[env_ids])

    def update_gait_phase(self):        
        isActive = torch.norm(self.commands[:, 1:3], dim=1) > 0.0
        # isActive = isActive | (torch.norm(self.base_ang_vel[:, 1:3], dim=1) > 2.0)
        isNoConstraint = ~isActive & (torch.norm(self.base_ang_vel[:, 1:3], dim=1) > 1.0)

        self.stop_timer = torch.where(
            isActive,
            torch.ones_like(self.stop_timer) * (1.5 / self.cfg.rewards.gait_freq),  # reset to 0.4s when active
            torch.clip(self.stop_timer - self.dt, min=0.)
        )
        isNoConstraint = isNoConstraint | (~isActive & (self.stop_timer > 0.))
        should_run_clock = isActive# | (self.stop_timer > 0.)
        self.gait_phase = torch.where(
            should_run_clock,
            torch.remainder(self.gait_phase + self.dt * self.cfg.rewards.gait_freq, 1.0),
            torch.ones_like(self.gait_phase) * -1e-9
        )

        self.gait_phase = self.gait_phase * (~isNoConstraint) - 2e-9 * isNoConstraint # 处于无约束状态时，phase设为一个小负数，保证sin波为负，表示悬空相

        # 区分起步是向左走还是向右走
        isLeftward = (self.commands[:,1] > 0) | ((self.commands[:,1] == 0) & (self.commands[:,2] > 0))
        new_offsets = torch.where(
            isLeftward.unsqueeze(1),
            self.gait_offset_leftward,
            self.gait_offset_rightward
        )        
        self.gait_offsets = torch.where(
            ((~self.last_should_run_clock) & should_run_clock).unsqueeze(1),
            new_offsets,
            self.gait_offsets
        )

        # phase_per_leg < 0.5 时，sin波为正，表示该腿处于支撑相
        phase_per_leg = (self.gait_phase.unsqueeze(1) + self.gait_offsets) * (self.gait_phase >= 0).unsqueeze(1)
        self.gait_clock = torch.sin(2 * np.pi * phase_per_leg)
        self.gait_clock = self.gait_clock * (~isNoConstraint).unsqueeze(1) # 无约束状态时，clock设为0

        self.last_should_run_clock = should_run_clock.clone()
        # print("gait phase:", self.gait_clock[:8])

    def _randomize_motor_params(self, env_ids):
        #randomization of pd gains
        if self.cfg.domain_rand.randomize_pd_gains:
            leg_p_mult = torch_rand_float(
                *self.cfg.domain_rand.stiffness_multiplier_range,
                (len(env_ids), self.num_actions_leg),
                device=self.device,
            )
            leg_d_mult = torch_rand_float(
                *self.cfg.domain_rand.damping_multiplier_range,
                (len(env_ids), self.num_actions_leg),
                device=self.device,
            )
            wheel_p_mult = torch_rand_float(
                *self.cfg.domain_rand.stiffness_multiplier_range,
                (len(env_ids), self.num_actions - self.num_actions_leg),
                device=self.device,
            )
            wheel_d_mult = torch_rand_float(
                *self.cfg.domain_rand.damping_multiplier_range,
                (len(env_ids), self.num_actions - self.num_actions_leg),
                device=self.device,
            )

            # 如果开启了对称增强/镜像损失，则必须保证“环境动力学/参数随机化”也满足左右对称。
            # 否则镜像样本的动力学与奖励不再等价，会在训练后期引入偏差，表现为无命令抖动/乱动。
            if getattr(self.cfg.domain_rand, "enforce_left_right_symmetry", False):
                if self.num_actions_leg == 12:
                    leg_p_mult = leg_p_mult.view(-1, 4, 3)
                    leg_d_mult = leg_d_mult.view(-1, 4, 3)
                    # order assumed: [FL, FR, RL, RR] x [hipx, hipy, knee]
                    leg_p_mult[:, 1, :] = leg_p_mult[:, 0, :]
                    leg_p_mult[:, 3, :] = leg_p_mult[:, 2, :]
                    leg_d_mult[:, 1, :] = leg_d_mult[:, 0, :]
                    leg_d_mult[:, 3, :] = leg_d_mult[:, 2, :]
                    leg_p_mult = leg_p_mult.view(-1, 12)
                    leg_d_mult = leg_d_mult.view(-1, 12)

                if (self.num_actions - self.num_actions_leg) == 4:
                    # order assumed: [FL, FR, RL, RR]
                    wheel_p_mult[:, 1] = wheel_p_mult[:, 0]
                    wheel_p_mult[:, 3] = wheel_p_mult[:, 2]
                    wheel_d_mult[:, 1] = wheel_d_mult[:, 0]
                    wheel_d_mult[:, 3] = wheel_d_mult[:, 2]

            self.pLeg_gains[env_ids] = self.pLeg_gains_init * leg_p_mult
            self.dLeg_gains[env_ids] = self.dLeg_gains_init * leg_d_mult
            self.pWheel_gains[env_ids] = self.pWheel_gains_init * wheel_p_mult
            self.dWheel_gains[env_ids] = self.dWheel_gains_init * wheel_d_mult
            # print(f"Randomized PD gains for envs:", leg_p_mult[:5], leg_d_mult[:5])

        # randomization of the motor zero calibration for real machine
        if self.cfg.domain_rand.randomize_legMotor_zero_offset:
            motor_zero_offsets_leg = torch_rand_float(
                *self.cfg.domain_rand.legMotor_zero_offset_range,
                (len(env_ids), self.num_actions_leg),
                device=self.device,
            )
            if getattr(self.cfg.domain_rand, "enforce_left_right_symmetry", False):
                if self.num_actions_leg == 12:
                    motor_zero_offsets_leg = motor_zero_offsets_leg.view(-1, 4, 3)
                    # order assumed: [FL, FR, RL, RR] x [hipx, hipy, knee]
                    # Note: hipx joint is mirrored with sign flip (see symmetry mapping);
                    # preserving physical L/R symmetry requires opposite sign in joint coordinates.
                    motor_zero_offsets_leg[:, 1, :] = motor_zero_offsets_leg[:, 0, :]
                    motor_zero_offsets_leg[:, 3, :] = motor_zero_offsets_leg[:, 2, :]
                    motor_zero_offsets_leg[:, 1, 0] = -motor_zero_offsets_leg[:, 0, 0] # hipx 取反值
                    motor_zero_offsets_leg[:, 3, 0] = -motor_zero_offsets_leg[:, 2, 0]
                    motor_zero_offsets_leg = motor_zero_offsets_leg.view(-1, 12)
            self.motor_zero_offsets_leg[env_ids] = motor_zero_offsets_leg
            # print(f"Randomized motor zero offsets:", motor_zero_offsets_leg[:5])

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
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # print("Asset DOF names:", self.dof_names)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        print("!!!!!!!!!!!!!!!!!!!!!! If gait phase is added, the order of foot indices must be checked !!!!!!!!")
        print("feet names:", feet_names)
        print("dof names:", self.dof_names)
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.p_gains_multiplier = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains_multiplier = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_zero_offsets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                         requires_grad=False) 

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids is not None:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        # 检查是否位于复杂地形
        heights_grid = heights.view(-1, len(self.cfg.terrain.measured_points_x), len(self.cfg.terrain.measured_points_y))
        # x:  0:17 -> 3:14 
        # y:  0:11 -> 2:9
        center_heights = heights_grid[:, 3:14, 2:9]   # [N, 11, 7]
        # 用最大高度差判断
        height_span = center_heights.amax(dim=(1, 2)) - center_heights.amin(dim=(1, 2))
        # 中间区域最大高差超过 5cm 就认为是复杂地形
        self.is_in_terrain = height_span > 0.05
        # print("is_in_terrain:", self.is_in_terrain)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        # print("Projected gravity:", self.projected_gravity[:5, :2])
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * ~self.is_in_terrain

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_action_smoothness(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_last_actions - 2 * self.last_actions), dim=1)
    
    def _reward_joint_power(self):
        # Penalize joint power consumption
        joint_powers = torch.abs(self.torques * self.dof_vel)
        return torch.sum(joint_powers, dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # print(torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma))
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma) * (1 + 0.5 * self.is_in_terrain)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma) * (1 + 0.5 * self.is_in_terrain)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        # rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime = torch.sum(self.feet_air_time * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, 1:3], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
            torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)