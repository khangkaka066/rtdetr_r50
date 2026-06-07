from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KalmanBoxFilter:
    """Constant-velocity Kalman filter over [cx, cy, w, h]."""

    def __init__(self, bbox, process_noise=1.0, measurement_noise=10.0):
        self.x = np.zeros(8, dtype=np.float32)
        self.x[:4] = np.asarray(bbox, dtype=np.float32)
        self.p = np.eye(8, dtype=np.float32) * 10.0
        self.q_scale = float(process_noise)
        self.r = np.eye(4, dtype=np.float32) * float(measurement_noise)

    def predict(self, dt=1.0):
        dt = max(float(dt), 1e-3)
        f = np.eye(8, dtype=np.float32)
        f[0, 4] = dt
        f[1, 5] = dt
        f[2, 6] = dt
        f[3, 7] = dt
        q = np.eye(8, dtype=np.float32) * self.q_scale
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        return self.x[:4].copy(), self.p[:4, :4].copy()

    def update(self, bbox):
        z = np.asarray(bbox, dtype=np.float32)
        h = np.zeros((4, 8), dtype=np.float32)
        h[:4, :4] = np.eye(4, dtype=np.float32)
        y = z - h @ self.x
        s = h @ self.p @ h.T + self.r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.p = (np.eye(8, dtype=np.float32) - k @ h) @ self.p
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        return self.x[:4].copy()

    @property
    def velocity(self):
        return self.x[4:8].copy()


class LSTMResidualPredictor(nn.Module):
    """Fallback LSTM residual branch.

    The heads are zero-initialized, so an untrained motion module does not
    damage Kalman predictions. Train or load this module to make residuals
    active.
    """

    def __init__(self, input_dim=8, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.residual = nn.Linear(hidden_dim, 4)
        self.log_var = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.log_var.weight)
        nn.init.constant_(self.log_var.bias, -2.0)

    def initial_state(self, device):
        h = torch.zeros(1, 1, self.hidden_dim, device=device)
        c = torch.zeros(1, 1, self.hidden_dim, device=device)
        return h, c

    def forward(self, history, state):
        out, state = self.rnn(history, state)
        last = out[:, -1]
        return self.residual(last), self.log_var(last), state


class OfficialXLSTMResidualPredictor(nn.Module):
    """NX-AI xLSTM residual branch over per-track motion history.

    Uses the official `xlstm` package when neural motion is explicitly enabled.
    The module is stateless at inference: the track history deque is the memory
    source, and the last xLSTM token becomes the residual prediction feature.
    """

    def __init__(self, input_dim=12, hidden_dim=64, context_length=16, num_blocks=2):
        super().__init__()
        try:
            from xlstm import (
                FeedForwardConfig,
                mLSTMBlockConfig,
                mLSTMLayerConfig,
                xLSTMBlockStack,
                xLSTMBlockStackConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "Official xLSTM backend requires `pip install xlstm dacite omegaconf`."
            ) from exc

        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.xlstm = xLSTMBlockStack(
            xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4,
                        qkv_proj_blocksize=4,
                        num_heads=4,
                    ),
                    feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
                ),
                context_length=context_length,
                num_blocks=num_blocks,
                embedding_dim=hidden_dim,
            )
        )
        self.residual = nn.Linear(hidden_dim, 4)
        self.log_var = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.log_var.weight)
        nn.init.constant_(self.log_var.bias, -2.0)

    def initial_state(self, device):
        return None

    def forward(self, history, state):
        history = self._pad_or_trim(history)
        x = self.input_proj(history)
        out = self.xlstm(x)
        last = out[:, -1]
        return self.residual(last), self.log_var(last), {"hidden": last}

    def _pad_or_trim(self, history):
        if history.shape[1] > self.context_length:
            return history[:, -self.context_length :]
        if history.shape[1] == self.context_length:
            return history
        pad_len = self.context_length - history.shape[1]
        pad = history[:, :1].repeat(1, pad_len, 1)
        return torch.cat([pad, history], dim=1)


class LNNResidualPredictor(nn.Module):
    """Small liquid-neural-network-style residual cell."""

    def __init__(self, input_dim=12, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.h_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.tau = nn.Parameter(torch.ones(hidden_dim))
        self.residual = nn.Linear(hidden_dim, 4)
        self.log_var = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.log_var.weight)
        nn.init.constant_(self.log_var.bias, -2.0)

    def initial_state(self, device):
        return torch.zeros(1, self.hidden_dim, device=device)

    def forward(self, x, state, dt):
        tau = F.softplus(self.tau).unsqueeze(0) + 1e-3
        target = torch.tanh(self.in_proj(x) + self.h_proj(state))
        dt = torch.as_tensor(dt, dtype=state.dtype, device=state.device).reshape(1, 1)
        state = state + dt * (target - state) / tau
        return self.residual(state), self.log_var(state), state


class FusionGate(nn.Module):
    def __init__(self, xlstm_dim=64, lnn_dim=64, extra_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(xlstm_dim + lnn_dim + extra_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, xlstm_state, lnn_state, extra):
        if isinstance(xlstm_state, dict):
            h_x = xlstm_state["hidden"]
        else:
            h_x = xlstm_state[0][-1]
        data = torch.cat([h_x, lnn_state, extra], dim=-1)
        return torch.sigmoid(self.net(data))


class HybridResidualMotion(nn.Module):
    def __init__(self, history_dim=12, lnn_input_dim=16, hidden_dim=64, motion_backend="xlstm"):
        super().__init__()
        if motion_backend == "xlstm":
            self.xlstm = OfficialXLSTMResidualPredictor(history_dim, hidden_dim)
        elif motion_backend == "lstm":
            self.xlstm = LSTMResidualPredictor(history_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported motion backend: {motion_backend}")
        self.lnn = LNNResidualPredictor(lnn_input_dim, hidden_dim)
        self.gate = FusionGate(hidden_dim, hidden_dim, extra_dim=4)

    def initial_state(self, device):
        return {
            "xlstm": self.xlstm.initial_state(device),
            "lnn": self.lnn.initial_state(device),
        }

    def predict(self, history, lnn_input, state, dt, missing_count, alpha0=1.0, beta=0.12):
        residual_x, log_var_x, xlstm_state = self.xlstm(history, state["xlstm"])
        residual_l, log_var_l, lnn_state = self.lnn(lnn_input, state["lnn"], dt)
        extra = torch.tensor(
            [[float(dt), float(missing_count), float(F.softplus(log_var_x).mean()), float(F.softplus(log_var_l).mean())]],
            dtype=history.dtype,
            device=history.device,
        )
        gate = self.gate(xlstm_state, lnn_state, extra)
        residual = gate * residual_l + (1.0 - gate) * residual_x
        uncertainty = gate * F.softplus(log_var_l) + (1.0 - gate) * F.softplus(log_var_x)
        alpha = float(alpha0) * float(np.exp(-float(beta) * float(missing_count)))
        return (
            residual.squeeze(0).detach().cpu().numpy() * alpha,
            uncertainty.squeeze(0).detach().cpu().numpy(),
            {"xlstm": xlstm_state, "lnn": lnn_state},
        )

    def update_state(self, history, lnn_input, state, dt):
        _, _, xlstm_state = self.xlstm(history, state["xlstm"])
        _, _, lnn_state = self.lnn(lnn_input, state["lnn"], dt)
        return {"xlstm": xlstm_state, "lnn": lnn_state}


@dataclass
class MotionState:
    kalman: KalmanBoxFilter
    neural: dict
    history: deque
