import os
import copy
import torch
import numpy as np
from typing import Optional


class PolicyDWAQ(torch.nn.Module):
    """
    DWAQ 部署版（History-only）：
      输入:  obs_history (1D 或 2D)
      输出:  action_mean (1D 或 2D)
    说明：
    - 训练时 actor 输入为 [code, obs] 拼接 :contentReference[oaicite:2]{index=2}
    - 这里不额外输入 obs，而是从 obs_history 中切出一帧作为 obs
      （默认取最后一帧：obs_history[-num_obs:]）
    - 为了部署稳定/可导出 ONNX，code 采用均值 mu（确定性），不做随机 reparameterise
    """
    def __init__(
        self,
        actor_critic,
        obs_from_history: str = "tail",   # "tail" | "head" | "index"
        frame_index: Optional[int] = None # 当 obs_from_history="index" 时生效：0..num_frames-1
    ):
        super().__init__()
        ac = copy.deepcopy(actor_critic).to("cpu").eval()

        # 只保留推理需要的模块
        self.encoder = ac.encoder
        self.encode_mean_latent = ac.encode_mean_latent
        self.encode_mean_vel = ac.encode_mean_vel
        self.actor = ac.actor

        # 推断维度（来自网络结构）
        self.hist_dim = self.encoder[0].in_features                # cenet_in_dim :contentReference[oaicite:3]{index=3}
        self.actor_in_dim = self.actor[0].in_features              # num_obs + cenet_out_dim（训练时 runner 这么构造）:contentReference[oaicite:4]{index=4}
        # code_dim = vel(3) + latent(cenet_out_dim-3) = cenet_out_dim
        self.code_dim = self.encode_mean_vel.out_features + self.encode_mean_latent.out_features
        self.obs_dim = self.actor_in_dim - self.code_dim

        if self.obs_dim <= 0:
            raise ValueError(
                f"Invalid obs_dim inferred: actor_in_dim({self.actor_in_dim}) - code_dim({self.code_dim}) = {self.obs_dim}. "
                f"Please check how ActorCritic_DWAQ was instantiated."
            )
        if self.hist_dim % self.obs_dim != 0:
            raise ValueError(
                f"obs_history dim ({self.hist_dim}) is not a multiple of obs_dim ({self.obs_dim}). "
                f"Cannot safely slice obs from history. (Expected hist_dim = num_frames * obs_dim) "
                "Runner uses num_obs_hist * num_obs :contentReference[oaicite:5]{index=5}"
            )
        self.num_frames = self.hist_dim // self.obs_dim

        self.obs_from_history = obs_from_history
        self.frame_index = frame_index

        if self.obs_from_history not in ("tail", "head", "index"):
            raise ValueError("obs_from_history must be one of: 'tail', 'head', 'index'")
        if self.obs_from_history == "index":
            if frame_index is None:
                raise ValueError("frame_index must be provided when obs_from_history='index'")
            if not (0 <= frame_index < self.num_frames):
                raise ValueError(f"frame_index out of range: 0..{self.num_frames-1}, got {frame_index}")

    def _slice_obs(self, obs_history: torch.Tensor) -> torch.Tensor:
        """
        obs_history:
          - 1D: [hist_dim]
          - 2D: [B, hist_dim]
        return:
          - 1D: [obs_dim]
          - 2D: [B, obs_dim]
        """
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

        raise ValueError(f"obs_history must be 1D or 2D, got shape: {tuple(obs_history.shape)}")

    @torch.inference_mode()
    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """
        输入:  obs_history (1D [hist_dim] 或 2D [B, hist_dim])
        输出:  action_mean (1D [num_actions] 或 2D [B, num_actions])
        """
        obs = self._slice_obs(obs_history)

        h = self.encoder(obs_history)
        mu_latent = self.encode_mean_latent(h)
        mu_vel = self.encode_mean_vel(h)
        code = torch.cat([mu_vel, mu_latent], dim=-1)  # [*, code_dim]

        actor_in = torch.cat([code, obs], dim=-1)      # [*, code_dim + obs_dim] = actor_in_dim
        action_mean = self.actor(actor_in)
        return action_mean

    @torch.inference_mode()
    def infer_numpy(self, obs_history_1d: np.ndarray) -> np.ndarray:
        """
        单机器人：输入/输出都是 1D numpy。
        """
        x = torch.from_numpy(np.asarray(obs_history_1d, dtype=np.float32))
        y = self.forward(x).cpu().numpy().astype(np.float32)
        return y


def export_dwaq_as_onnx(
    actor_critic,
    path: str,
    file_name: str = "policy.onnx",
    opset_version: int = 11,
    obs_from_history: str = "tail",
    frame_index: Optional[int] = None,
) -> str:
    """
    导出 ONNX（单输入单输出，均为 1D）：
      input : obs_history  [hist_dim]
      output: action_mean  [num_actions]

    默认从 history 末尾切出 obs（最后一帧），与 runner 的 hist 形状设计一致 :contentReference[oaicite:6]{index=6}
    """
    import torch.onnx

    onnx_dir = os.path.join(path, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, file_name)

    model = PolicyDWAQ(actor_critic, obs_from_history=obs_from_history, frame_index=frame_index).to("cpu").eval()

    dummy_input = torch.randn(model.hist_dim, dtype=torch.float32)  # 1D

    # --- sanity check: run a forward with dummy input and print shapes ---
    try:
        with torch.no_grad():
            out = model(dummy_input)
        print("[export_dwaq_as_onnx] forward OK, out.shape =", getattr(out, "shape", None))
    except Exception as e:
        print("[export_dwaq_as_onnx] Forward failed before ONNX export:", e)
        try:
            print("  hist_dim =", model.hist_dim)
            print("  obs_dim  =", model.obs_dim)
            print("  code_dim =", model.code_dim)
            print("  actor_in_dim =", model.actor_in_dim)
            first_lin = getattr(model.actor, 0, None)
            if hasattr(first_lin, "in_features"):
                print("  actor[0].in_features =", first_lin.in_features, "out_features =", first_lin.out_features)
        except Exception:
            pass
        raise

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["obs_history"],
        output_names=["action"],
    )

    print(f"[export_dwaq_as_onnx] Exported: {onnx_path}")
    print(f"  hist_dim={model.hist_dim}, obs_dim={model.obs_dim}, code_dim={model.code_dim}, num_actions={model.actor[-1].out_features}")
    print(f"  obs_from_history={obs_from_history}, frame_index={frame_index}, num_frames={model.num_frames}")
    return onnx_path