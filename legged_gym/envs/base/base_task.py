
import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        self.num_privileged_group = cfg.env.num_privileged_group
        self.num_proprio_group = cfg.env.num_proprio_group
        #+ 
        self.num_actions_leg = cfg.env.num_actions_leg
        self.num_actions_wheel = cfg.env.num_actions_wheel
        self.num_obs_hist = cfg.env.num_obs_hist
        #+ 轮足区分 足部关节和轮部关节的索引
        self.dof_idx_leg = torch.tensor(self.cfg.init_state.dof_idx_leg, device=sim_device, dtype=torch.long)
        self.dof_idx_wheel = torch.tensor(self.cfg.init_state.dof_idx_wheel, device=sim_device, dtype=torch.long)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs
        # 新增 obs_history_buf + latency buffer
        self.obs_hist_buf = torch.zeros(self.num_envs, self.num_obs_hist * self.num_obs, device=self.device, dtype=torch.float)
        self._obs_hist_ring = torch.zeros(
            self.num_envs,
            self.num_obs_hist,
            self.num_obs,
            device=self.device,
            dtype=torch.float,
        )
        self._obs_hist_head = 0
        self._obs_hist_buf3 = self.obs_hist_buf.view( # view obs_hist_buf，实际上是以 3d 修改 obs_hist_buf
            self.num_envs,
            self.num_obs_hist,
            self.num_obs,
        )

        self.cmd_action_latency_ring = torch.zeros(self.num_envs, cfg.domain_rand.range_cmd_action_latency[1]+1, self.num_actions, device=self.device)
        self.obs_motor_latency_ring = torch.zeros(self.num_envs, cfg.domain_rand.range_obs_motor_latency[1]+1, self.num_actions_leg + self.num_actions, device=self.device)
        self.obs_imu_latency_ring = torch.zeros(self.num_envs, cfg.domain_rand.range_obs_imu_latency[1]+1, 6, device=self.device)
        self._cmd_action_latency_head = 0
        self._obs_motor_latency_head = 0
        self._obs_imu_latency_head = 0
        self.cmd_action_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.obs_motor_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.obs_imu_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        #+ 优化参数
        self._all_envs_ids = torch.arange(self.num_envs, device=self.device)        
        #+ 轮足信息
        self.dof_pos_leg = torch.zeros(self.num_envs, self.num_actions_leg, device=self.device)
        self.dof_vel_leg = torch.zeros(self.num_envs, self.num_actions_leg, device=self.device)
        self.dof_vel_wheel = torch.zeros(self.num_envs, self.num_actions_wheel, device=self.device)

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def get_observations(self):
        return self.obs_buf, self.extras
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, extras = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        # 新增 obs_history_buf 初始化
        obs_hist_buf = extras.get("obs_hist_buf")
        privileged_obs = extras.get("privileged_obs_buf")
        return obs, privileged_obs, obs_hist_buf

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)