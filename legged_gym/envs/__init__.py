from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from .base.legged_robot import LeggedRobot

from legged_gym.envs.go2w.go2w_env import GO2wRobot
from legged_gym.envs.go2w.go2w_config import GO2wRoughCfg, GO2wRoughCfgPPO
from legged_gym.envs.go2w.go2w_difRewFunc import GO2wRobot_difRewFunc, GO2wCfg_difRewFunc

from legged_gym.envs.m20.m20_env import M20Robot
from legged_gym.envs.m20.m20_config import M20RoughCfg, M20RoughCfgPPO
from legged_gym.envs.m20.m20_config_plane import M20RoughCfg as M20RoughCfgPlane
from legged_gym.envs.m20.m20_config_plane import M20RoughCfgPPO as M20RoughCfgPPoPlane

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2w", GO2wRobot, GO2wRoughCfg(), GO2wRoughCfgPPO())
task_registry.register( "go2w_difRew", GO2wRobot_difRewFunc, GO2wCfg_difRewFunc(), GO2wRoughCfgPPO())

task_registry.register( "m20_plane", M20Robot, M20RoughCfgPlane(), M20RoughCfgPPoPlane())
task_registry.register( "m20", M20Robot, M20RoughCfg(), M20RoughCfgPPO())
