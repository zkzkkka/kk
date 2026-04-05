"""
反馈与训练层 (Feedback & Training Layer)
实现 DQN 强化学习主训练循环，包括经验回放池(Replay Buffer)、
Epsilon-Greedy(ε-贪心) 策略，以及可持续训练所需的 checkpoint 持久化。

如果把强化学习过程简化成一句话，就是：
“观察当前状态 -> 选一个动作 -> 看结果好不好 -> 再根据结果更新自己”

这个文件并不负责直接观测交换机或下发流表，它只负责“学习器”本身：
- 保存历史经验
- 根据当前状态选择动作
- 使用历史经验回放来更新神经网络
- 在控制器重启后恢复训练进度
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    # When imported as a module (e.g., from control_plane), prefer relative import.
    from .rl_model import RoutingDQN  # type: ignore
except Exception:
    # Fallback for running this file directly.
    from rl_model import RoutingDQN


def _env_int(name: str, default: int) -> int:
    """从环境变量读取整数配置；解析失败时回退到默认值。"""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """从环境变量读取浮点配置；解析失败时回退到默认值。"""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """从环境变量读取布尔配置，支持 0/false/no/off 表示关闭。"""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class AgentConfig:
    """
    DQN 智能体的训练超参数与 checkpoint 配置。

    这些参数决定了智能体“学得快不快、稳不稳、敢不敢探索”。
    常见字段含义：
    - memory_size: 经验池最多保留多少条历史样本
    - gamma: 未来奖励折扣系数，越大表示越看重长期收益
    - epsilon_*: 探索率相关参数，控制随机试错的力度
    - target_update_interval: 目标网络多久同步一次
    - batch_size/min_replay_size: 每次训练抽样多少条，以及最少积累多少条才开始训练
    """

    memory_size: int = 20000
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.998
    learning_rate: float = 0.001
    target_update_interval: int = 20
    batch_size: int = 8
    min_replay_size: int = 8
    grad_clip_norm: float = 5.0
    checkpoint_path: Optional[str] = None
    persist_replay_buffer: bool = True

    @classmethod
    def from_env(cls, checkpoint_path: Optional[str] = None) -> "AgentConfig":
        """
        允许通过环境变量覆盖默认训练配置，便于实验调参。

        这里把“代码里的默认值”与“运行时命令行/脚本传入的环境变量”整合起来，
        这样同一份代码可以方便地跑多组参数实验，而不需要每次手改源码。
        """
        env_checkpoint_path = os.environ.get("RL_CHECKPOINT_PATH") or checkpoint_path
        batch_size = max(1, _env_int("RL_BATCH_SIZE", cls.batch_size))
        min_replay_size = max(1, _env_int("RL_MIN_REPLAY_SIZE", cls.min_replay_size))
        min_replay_size = min(min_replay_size, batch_size)
        return cls(
            memory_size=max(1, _env_int("RL_MEMORY_SIZE", cls.memory_size)),
            gamma=_env_float("RL_GAMMA", cls.gamma),
            epsilon_start=_env_float("RL_EPSILON_START", cls.epsilon_start),
            epsilon_min=_env_float("RL_EPSILON_MIN", cls.epsilon_min),
            epsilon_decay=_env_float("RL_EPSILON_DECAY", cls.epsilon_decay),
            learning_rate=_env_float("RL_LEARNING_RATE", cls.learning_rate),
            target_update_interval=max(
                1,
                _env_int("RL_TARGET_UPDATE_INTERVAL", cls.target_update_interval),
            ),
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            grad_clip_norm=max(0.0, _env_float("RL_GRAD_CLIP_NORM", cls.grad_clip_norm)),
            checkpoint_path=env_checkpoint_path,
            persist_replay_buffer=_env_bool("RL_PERSIST_REPLAY_BUFFER", cls.persist_replay_buffer),
        )


class Agent:
    """
    DQN 智能体：负责选动作、记录经验、回放训练与持久化。

    可以把它理解成一个带记忆的决策器：
    1. `act()` 根据当前状态给出动作
    2. `remember()` 把一次交互过程存入经验池
    3. `replay()` 从经验池随机抽样训练网络
    4. `save_checkpoint()/load_checkpoint()` 负责断点续训
    """

    def __init__(self, state_size, action_size, config: Optional[AgentConfig] = None):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or AgentConfig()
        # 经验回放池：保存最近若干条交互样本，训练时随机抽样。
        self.memory = deque(maxlen=self.config.memory_size)

        # 下面这些成员都是强化学习里经常出现的“训练状态”。
        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon_start
        self.epsilon_min = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay
        self.learning_rate = self.config.learning_rate
        self.target_update_interval = self.config.target_update_interval
        self.train_steps = 0

        # 设定计算设备 (为简化流程，此处优先使用 CPU。如果具有 GPU 能力请按需调整。)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 策略网络负责实时学习；目标网络提供相对稳定的训练目标。
        self.model = RoutingDQN(state_size, action_size).to(self.device)
        self.target_model = RoutingDQN(state_size, action_size).to(self.device)
        self.update_target_model()

        # Adam 负责根据梯度更新网络参数；SmoothL1Loss/Huber loss 对异常值更稳。
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss 在在线训练场景下更稳

    def _is_valid_state(self, state):
        """校验状态向量长度，避免把异常样本写入经验池或送入网络。"""
        return state is not None and len(state) == self.state_size

    def update_target_model(self):
        """
        同步目标网络的权重。

        DQN 会维护两个网络：
        - model: 当前正在学习的网络
        - target_model: 用来生成训练目标的“参考网络”

        如果训练目标也跟着每一步一起剧烈变化，训练会很不稳定。
        因此这里采用“隔一段时间再整体同步一次”的做法。
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        将五元组信息存储入经验回放池中。

        一条经验通常写成：
        (state, action, reward, next_state, done)

        它表示：
        - state: 执行动作前的状态
        - action: 当时选择的动作
        - reward: 这个动作带来的即时反馈
        - next_state: 执行动作后的下一状态
        - done: 这一轮是否结束
        """
        if not self._is_valid_state(state) or not self._is_valid_state(next_state):
            return False
        self.memory.append((state, action, reward, next_state, done))
        return True

    @property
    def replay_size(self):
        """返回经验池当前样本数。"""
        return len(self.memory)

    def act(self, state, greedy=False, epsilon_override=None):
        """
        Epsilon-greedy 动作选择策略：
        以 epsilon 概率随机探索新路径；否则利用神经网络选择当前认知中最优的路径权重。

        可以把它理解成“有时候故意试新路，有时候用自己当前最有把握的判断”：
        - 随机探索：避免智能体一开始就被局部最优困住
        - 贪心利用：利用神经网络已经学到的知识来追求更高奖励

        `greedy=True` 时会强制关闭探索，常用于纯评估阶段。
        """
        if not self._is_valid_state(state):
            raise ValueError(
                f"state size mismatch: expected {self.state_size}, got {len(state) if state is not None else 'None'}"
            )
        epsilon = self.epsilon if epsilon_override is None else float(epsilon_override)
        if greedy:
            epsilon = 0.0

        # 探索阶段直接随机选一个动作，帮助智能体跳出当前局部最优。
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)

        # 神经网络期望输入形状是 [batch_size, state_size]，
        # 单条状态需要先补出一个 batch 维度。
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
        # 网络输出的是“每个动作的 Q 值”，而不是动作本身；
        # 因此这里通过 argmax 取出价值最高的动作编号。
        return np.argmax(action_values.cpu().data.numpy())

    def replay(self, batch_size=None):
        """
        经验回放机制：随机抽取一个批次的样本训练神经网络。

        为什么不直接用“刚刚发生的那一步”训练？
        - 连续时刻的数据高度相关，直接训练容易不稳定
        - 随机回放可以打散样本顺序，让训练更像常规监督学习
        - 同一条历史经验可以被多次利用，提高样本效率
        """
        # 先过滤掉状态维度不合法的样本，避免 batch 张量拼接时报错。
        valid_memory = [
            transition for transition in self.memory
            if self._is_valid_state(transition[0]) and self._is_valid_state(transition[3])
        ]

        # 样本太少时不训练，是为了避免网络在非常偶然的几条经验上过拟合。
        min_replay_size = max(1, self.config.min_replay_size)
        if len(valid_memory) < min_replay_size:
            return None

        batch_size = self.config.batch_size if batch_size is None else int(batch_size)
        batch_size = max(1, min(batch_size, len(valid_memory)))

        # 随机采样一个批次的数据集，使训练样本尽量“去时序相关”。
        minibatch = random.sample(valid_memory, batch_size)

        # 把随机抽样得到的 Python 列表统一堆叠成张量。
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(self.device)

        # Q(s, a):
        # 模型会对“这个 batch 中每个状态的所有动作”都给出估值，
        # gather(1, actions) 的作用是只取出“当时真正执行过的那个动作”的 Q 值。
        q_values = self.model(states).gather(1, actions).squeeze(1)

        # Target Q(s, a):
        # DQN 的核心更新目标是
        #   reward + gamma * max_a' Q_target(next_state, a')
        # 直觉上就是：当前回报 + 下一步开始能拿到的最好未来收益。
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            # done=1 时后续回报应被截断，因此用 (1 - dones) 屏蔽未来项。
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # loss 越小，表示“模型当前估计的 Q 值”越接近“基于 Bellman 方程构造的目标 Q 值”。
        loss = self.criterion(q_values, target_q_values)

        # 标准的 PyTorch 训练步骤：清梯度 -> 反向传播 -> 更新参数。
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip_norm > 0:
            # 梯度裁剪用于避免极端样本把参数更新拉得过大，在线训练里比较常见。
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        self.train_steps += 1

        # 每隔若干步把策略网络同步到目标网络，稳定训练目标。
        if self.train_steps % self.target_update_interval == 0:
            self.update_target_model()

        # 随着训练进行，逐步减少随机探索，慢慢从“多试错”转向“多利用”。
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def checkpoint_payload(self):
        """
        整理可序列化的训练状态，用于保存 checkpoint。

        这里保存的不只是模型参数，还包括优化器、epsilon、训练步数，
        因为这些都会影响“继续训练时的行为”。
        """
        payload = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
            "config": asdict(self.config),
        }
        if self.config.persist_replay_buffer:
            payload["memory"] = list(self.memory)
        return payload

    def _load_model_state_dict_compat(self, state_dict):
        """
        加载模型参数，并兼容输入维度扩展后的 checkpoint。

        典型场景：
        旧版本状态向量只有 N 维，新版本加了几个新特征变成 N+k 维。
        这时第一层线性层 `fc1.weight` 的输入维度会变大，无法直接加载旧权重。

        这里的策略是：
        - 旧特征对应的权重尽量原样保留
        - 新增特征对应的权重先补 0
        这样旧模型仍然能作为新模型的初始化起点。
        """
        current_state = self.model.state_dict()
        adapted_state = {}

        for name, current_tensor in current_state.items():
            source_tensor = state_dict.get(name)
            if source_tensor is None:
                adapted_state[name] = current_tensor
                continue

            # 形状完全一致时，直接复用旧参数即可。
            if tuple(source_tensor.shape) == tuple(current_tensor.shape):
                adapted_state[name] = source_tensor
                continue

            # 仅兼容首层输入扩展：旧 checkpoint 没有的新特征列使用 0 初始化，
            # 这样历史模型仍可作为当前版本的起点继续评估或训练。
            if (
                name == "fc1.weight"
                and source_tensor.ndim == 2
                and current_tensor.ndim == 2
                and source_tensor.shape[0] == current_tensor.shape[0]
                and source_tensor.shape[1] < current_tensor.shape[1]
            ):
                padded = torch.zeros_like(current_tensor)
                padded[:, : source_tensor.shape[1]] = source_tensor
                adapted_state[name] = padded
                continue

            raise ValueError(
                f"checkpoint tensor mismatch for {name}: expected {tuple(current_tensor.shape)}, got {tuple(source_tensor.shape)}"
            )

        self.model.load_state_dict(adapted_state)
        self.update_target_model()

    def save_checkpoint(self, path=None):
        """
        保存完整训练状态，便于下次控制器重启后续训。

        对在线路由实验来说，checkpoint 很重要：
        否则控制器重启一次，智能体就会回到“完全随机探索”的早期阶段。
        """
        checkpoint_path = path or self.config.checkpoint_path
        if not checkpoint_path:
            raise ValueError("checkpoint path is not configured")

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.checkpoint_payload(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, path=None, map_location=None):
        """
        恢复完整训练状态，包括模型、优化器、epsilon 和经验池。

        返回值：
        - True: 成功加载 checkpoint
        - False: 路径不存在或未配置 checkpoint

        注意：
        - 如果 state_size 变大，会走兼容分支，只恢复尽可能兼容的模型参数
        - 这种情况下通常不恢复旧优化器和旧经验池，以免把不兼容的历史状态继续带入
        """
        checkpoint_path = path or self.config.checkpoint_path
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return False

        payload = torch.load(
            checkpoint_path,
            map_location=map_location or self.device,
        )
        checkpoint_state_size = payload.get("state_size")
        state_size_mismatch = checkpoint_state_size != self.state_size
        if checkpoint_state_size is None:
            raise ValueError("checkpoint does not contain state_size")
        if checkpoint_state_size > self.state_size:
            raise ValueError(
                f"checkpoint state_size mismatch: expected {self.state_size}, got {checkpoint_state_size}"
            )
        if payload.get("action_size") != self.action_size:
            raise ValueError(
                f"checkpoint action_size mismatch: expected {self.action_size}, got {payload.get('action_size')}"
            )

        # 如果状态维度变了，只恢复能兼容的模型参数，不恢复优化器和旧经验池。
        if state_size_mismatch:
            self._load_model_state_dict_compat(payload["model_state_dict"])
        else:
            self.model.load_state_dict(payload["model_state_dict"])
            self.target_model.load_state_dict(payload["target_model_state_dict"])
        if not state_size_mismatch:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.epsilon = float(payload.get("epsilon", self.config.epsilon_start))
        self.train_steps = int(payload.get("train_steps", 0))

        memory = [] if state_size_mismatch else payload.get("memory", [])
        self.memory = deque(memory, maxlen=self.config.memory_size)
        return True
