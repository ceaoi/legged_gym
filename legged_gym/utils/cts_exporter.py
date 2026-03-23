import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

class PolicyCTS(torch.nn.Module):
    """
    CTS 学生端部署版 (History-only):
    自动适配传入的是 Proprioceptive_Encoder 类实例还是内部的 Sequential
    """
    def __init__(
        self,
        proprioceptive_encoder,
        actor_critic,
        obs_from_history: str = "tail",   # "tail" | "head" | "index"
        frame_index: Optional[int] = None # 当 obs_from_history="index" 时生效
    ):
        super().__init__()
        
        # 深拷贝并转 CPU 推理模式
        # 注意：这里先拷贝，后续根据类型判断如何使用
        self.proprio_encoder_module = copy.deepcopy(proprioceptive_encoder).to("cpu").eval()
        ac = copy.deepcopy(actor_critic).to("cpu").eval()
        self.actor = ac.actor

        # --- 1. 智能识别 Encoder 类型并推断维度 ---
        self.is_wrapper_class = False

        # If user passed the custom Proprioceptive_Encoder instance
        if hasattr(self.proprio_encoder_module, "proprio_encoder_input_dim") or hasattr(
            self.proprio_encoder_module, "proprioceptive_encoder_input_dim"
        ):
            self.is_wrapper_class = True
            # input dim of history encoder
            self.hist_dim = getattr(
                self.proprio_encoder_module, "proprio_encoder_input_dim", None
            ) or getattr(self.proprio_encoder_module, "proprioceptive_encoder_input_dim")
            # latent dim exposed by encoder
            if hasattr(self.proprio_encoder_module, "num_latent_dim"):
                self.latent_dim = self.proprio_encoder_module.num_latent_dim
            else:
                # fallback: if forward returns vel+latent, we cannot infer split automatically
                raise ValueError(
                    "无法自动推断自定义 encoder 的 latent 维度，请在 encoder 中暴露 'num_latent_dim'."
                )
            if hasattr(self.proprio_encoder_module, "num_est_vel_dim"):
                self.est_vel_dim = self.proprio_encoder_module.num_est_vel_dim
            else:
                # fallback: if forward returns vel+latent, we cannot infer split automatically
                raise ValueError(
                    "无法自动推断自定义 encoder 的 estimated velocity 维度，请在 encoder 中暴露 'num_est_vel_dim'."
                )

        # If user passed an nn.Sequential or raw Sequential-like module
        elif isinstance(self.proprio_encoder_module, torch.nn.Sequential):
            self.is_wrapper_class = False
            self.hist_dim = self.proprio_encoder_module[0].in_features
            self.latent_dim = self.proprio_encoder_module[-1].out_features

        else:
            # last resort: try to introspect common attributes
            if hasattr(self.proprio_encoder_module, "in_features"):
                self.hist_dim = self.proprio_encoder_module.in_features
                # guess latent dim
                if hasattr(self.proprio_encoder_module, "out_features"):
                    self.latent_dim = self.proprio_encoder_module.out_features
                    self.is_wrapper_class = False
                else:
                    raise ValueError(
                        f"无法识别 proprioceptive_encoder 的结构，类型为: {type(self.proprio_encoder_module)}"
                    )
            else:
                raise ValueError(
                    f"无法识别 proprioceptive_encoder 的结构，类型为: {type(self.proprio_encoder_module)}"
                )

        # --- 2. 推断其他维度 ---
        self.actor_in_dim = self.actor[0].in_features
        # Actor 输入 = cat(obs, latent)  --> obs_dim = actor_in - latent
        self.obs_dim = self.actor_in_dim - self.latent_dim - self.est_vel_dim

        # --- 3. 校验 ---
        if self.obs_dim <= 0:
            raise ValueError(
                f"推断出的 obs_dim 无效: actor_in({self.actor_in_dim}) - latent({self.latent_dim}) = {self.obs_dim}"
            )
        if self.hist_dim % self.obs_dim != 0:
            raise ValueError(
                f"obs_history dim ({self.hist_dim}) 不是 obs_dim ({self.obs_dim}) 的整数倍。"
            )
        
        self.num_frames = self.hist_dim // self.obs_dim
        
        # 配置切片逻辑
        self.obs_from_history = obs_from_history
        self.frame_index = frame_index
        if self.obs_from_history == "index" and frame_index is None:
             raise ValueError("当 obs_from_history='index' 时必须提供 frame_index")

    def _slice_obs(self, obs_history: torch.Tensor) -> torch.Tensor:
        """从 history 中切出单帧 obs"""
        if obs_history.dim() == 1:
            if self.obs_from_history == "tail":
                return obs_history[-self.obs_dim:]
            if self.obs_from_history == "head":
                return obs_history[:self.obs_dim]
            # index
            start = self.frame_index * self.obs_dim
            end = start + self.obs_dim
            return obs_history[start:end]

        if obs_history.dim() == 2:
            if self.obs_from_history == "tail":
                return obs_history[:, -self.obs_dim:]
            if self.obs_from_history == "head":
                return obs_history[:, :self.obs_dim]
            # index
            start = self.frame_index * self.obs_dim
            end = start + self.obs_dim
            return obs_history[:, start:end]

        raise ValueError(f"obs_history shape 错误: {tuple(obs_history.shape)}")

    @torch.inference_mode()
    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass:
        1. 切片获取当前 Obs
        2. 编码 History 获取 Latent (必须包含 normalize)
        3. 拼接并决策
        """
        # 1. Slice Obs
        obs = self._slice_obs(obs_history)

        # 2. Encode History
        # We need both estimated velocity head and latent, because the actor expects
        # `actor_in = concat(obs, vel, latent)` as in training.
        if self.is_wrapper_class:
            if hasattr(self.proprio_encoder_module, "get_vel_and_latent"):
                vel, latent = self.proprio_encoder_module.get_vel_and_latent(obs_history)
            elif hasattr(self.proprio_encoder_module, "get_proprioceptive_latent"):
                # no vel head exposed; create zeros for vel
                latent = self.proprio_encoder_module.get_proprioceptive_latent(obs_history)
                vel_dim = getattr(self.proprio_encoder_module, "num_est_vel_dim", 0)
                vel = torch.zeros(latent.shape[0], vel_dim, dtype=latent.dtype, device=latent.device)
            else:
                out = self.proprio_encoder_module(obs_history)
                if hasattr(self.proprio_encoder_module, "num_est_vel_dim") and hasattr(
                    self.proprio_encoder_module, "num_latent_dim"
                ):
                    vel = out[:, : self.proprio_encoder_module.num_est_vel_dim]
                    latent = out[:, self.proprio_encoder_module.num_est_vel_dim :]
                else:
                    # unknown layout: assume no vel head
                    latent = out
                    vel = torch.zeros(latent.shape[0], max(0, self.actor_in_dim - (self.obs_dim + self.latent_dim)), dtype=latent.dtype, device=latent.device)
        else:
            # 如果是 Sequential，直接调用并手动补充 normalize
            # 对应 proprioceptive_encoder.py 中的 F.normalize(latent, p=2, dim=-1)
            latent = self.proprio_encoder_module(obs_history)
            latent = F.normalize(latent, p=2, dim=-1)
            # infer vel_dim required by actor
            vel_dim = max(0, self.actor_in_dim - (self.obs_dim + self.latent_dim))
            vel = torch.zeros(latent.shape[0], vel_dim, dtype=latent.dtype, device=latent.device)

        # 3. Actor Inference (先 Obs, 再 Vel, 再 Latent，与 PPO 训练逻辑一致)
        actor_in = torch.cat([obs, vel, latent], dim=-1)
        action_mean = self.actor(actor_in)
        return action_mean


def export_cts_as_onnx(
    proprioceptive_encoder,
    actor_critic,
    path: str,
    file_name: str = "policy_cts.onnx",
    opset_version: int = 11,
    obs_from_history: str = "tail",
    frame_index: Optional[int] = None,
) -> str:
    import torch.onnx

    onnx_dir = os.path.join(path, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, file_name)

    print(f"--- 正在导出 CTS 模型 ---")
    
    try:
        model = PolicyCTS(
            proprioceptive_encoder, 
            actor_critic, 
            obs_from_history=obs_from_history, 
            frame_index=frame_index
        ).to("cpu").eval()
    except Exception as e:
        print(f"初始化 PolicyCTS 失败: {e}")
        # 打印更多调试信息
        print(f"传入的 encoder 类型: {type(proprioceptive_encoder)}")
        if isinstance(proprioceptive_encoder, torch.nn.Sequential):
            print("检测到传入的是 Sequential，可能是导致 Attribute Error 的原因，新代码已修复此问题。")
        raise

    # 构造 Dummy Input
    dummy_input = torch.randn(model.hist_dim, dtype=torch.float32)

    print(f"  Hist Dim (输入): {model.hist_dim}")
    print(f"  Obs Dim (切片): {model.obs_dim}")
    print(f"  Latent Dim: {model.latent_dim}")
    print(f"  Wrapper Mode: {model.is_wrapper_class}")

    # Forward Test
    try:
        with torch.no_grad():
            out = model(dummy_input)
        print(f"  前向测试: 成功, 输出维度 {tuple(out.shape)}")
    except Exception as e:
        print(f"  前向测试: 失败 ({e})")
        raise

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actions"],
    )

    print(f"[export_cts_as_onnx] 导出成功: {onnx_path}")
    return onnx_path