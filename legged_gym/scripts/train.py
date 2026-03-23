import os
import numpy as np
from datetime import datetime
import sys

# NumPy compatibility for IsaacGym (NumPy>=1.24 removed deprecated aliases like np.float)
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    task_registry.save_cfgs(name=args.task)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
