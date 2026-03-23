import os
import sys
import numpy as np

# NumPy compatibility for IsaacGym (NumPy>=1.24 removed deprecated aliases like np.float)
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from legged_gym import LEGGED_GYM_ROOT_DIR
from isaacgym import gymapi

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, export_student_teacher_policy_as_jit, export_policy_as_onnx,task_registry, Logger, \
                                export_dwaq_as_onnx, export_cts_as_onnx
from legged_gym.utils.XboxController import XboxController
import torch


def _update_follow_camera(env, env_id: int = 0, distance: float = 2.0, yaw_offset_deg: float = 45.0, target_height: float = 0.5):
    if getattr(env, "viewer", None) is None:
        return
    if env_id >= getattr(env, "num_envs", 1):
        return

    root = env.root_states[env_id]
    x, y, z = root[0].item(), root[1].item(), root[2].item()
    qx, qy, qz, qw = root[3].item(), root[4].item(), root[5].item(), root[6].item()

    # yaw (z-axis) from quaternion (x,y,z,w)
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    yaw = yaw + np.deg2rad(yaw_offset_deg)

    cam_x = x - distance * np.cos(yaw)
    cam_y = y - distance * np.sin(yaw)
    cam_z = z + distance  # distance==height => 约45°俯视

    env.gym.viewer_camera_look_at(
        env.viewer,
        None,
        gymapi.Vec3(cam_x, cam_y, cam_z),
        gymapi.Vec3(x, y, z + target_height),
    )

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing    
    if REMOTE_CONTROL:
        rc = XboxController()
        rc.start()
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
        env_cfg.commands.heading_command = False
        env_cfg.env.max_episode_length = int(1e9)
        env_cfg.env.episode_length_s = 1e9
    else:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_pd_gains = False
    env_cfg.domain_rand.randomize_legMotor_zero_offset = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # env.debug_viz = False
    env.reset()
    obs, exteras = env.get_observations()[:2]
    obs_hist = exteras.get("obs_hist_buf", None)
    critic_obs = exteras["privileged_obs_buf"]
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    privileged_encoder = (
        ppo_runner.get_inference_teacher_student_module_privileged_encoder(
            device=env.device
        )
    )
    proprio_encoder = ppo_runner.get_inference_teacher_student_module_proprio_encoder(
        device=env.device
    )
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        if( train_cfg.algorithm.class_name == 'Distillation' ):
            export_student_teacher_policy_as_jit(ppo_runner.alg.policy, path)
        else:
            export_policy_as_jit(ppo_runner.alg.policy, path)
            # export_policy_as_onnx(ppo_runner.alg.policy, path)
            export_cts_as_onnx(
                ppo_runner.alg.proprioceptive_encoder,
                ppo_runner.alg.policy,
                path,
                "policy.onnx",
            )
        print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):
        if REMOTE_CONTROL:
            cmd = rc.get_cmd()
            cmd[0] *= 2.0
            cmd[1] *= 1.0
            cmd[2] *= 1.0
            env.commands[0, :3] = torch.tensor(cmd, device=env.commands.device, dtype=env.commands.dtype)
            _update_follow_camera(env)
        if args.test_obj == "teacher":
            latent = privileged_encoder(critic_obs)
        elif args.test_obj == "student":
            latent = proprio_encoder(obs_hist)
        else:
            raise ValueError("test_obj: \"teacher\" / \"student\"")
        actions = policy(torch.cat((obs, latent), dim=-1).detach())
        obs, rews, dones, infos = env.step(actions.detach())
        # print("gait phase", env.gait_phase)
        obs_hist = infos.get("obs_hist_buf", None)
        critic_obs = infos.get("privileged_obs_buf", None)
        # print("cmd:", obs[0, 50:])

if __name__ == '__main__':
    # test_obj = "student" # "teacher" / "student"
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    REMOTE_CONTROL = False
    args = get_args()
    if hasattr(args, 'remote_ctrl') and args.remote_ctrl:
        REMOTE_CONTROL = True
    print("REMOTE_CONTROL:", REMOTE_CONTROL)
    print("TEST_OBJ:", args.test_obj)
    play(args)
