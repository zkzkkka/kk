"""
智能决策层 (Intelligent Decision Layer - RL 模型)
使用 PyTorch 定义神经网络的架构。
我们在这里使用深度 Q 网络 (DQN) 来根据特定的网络状态
预测行动 (Action) 的 Q 值。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RoutingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(RoutingDQN, self).__init__()

        # 网络结构采用最基础的多层感知机：
        # 状态向量 -> 64 维隐藏层 -> 64 维隐藏层 -> 每个动作一个 Q 值
        # 这里输出的不是“已经选好的动作”，而是每个动作对应的 Q 值估计。
        # 后续由 Agent.act() 再通过 argmax 把 Q 值转换成最终动作编号。
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        """
        前向传播逻辑(Forward pass)
        x: 环境状态向量 (例如所有链路上经过归一化处理后的带宽利用率)
        returns: 返回每个可选动作的 Q 值。
        """
        # 先通过两层 ReLU 提取非线性特征，再输出每个动作的价值估计。
        # 最后一层不做 softmax，因为 Q 值本身是实数回报估计，不是概率分布。
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 在更进阶的版本中，如果动作空间(分配的具体链路权重)需要是连续的一组浮点值，
# 我们往往会使用 DDPG (深度确定性策略梯度，Deep Deterministic Policy Gradient) 算法。
# 为简便起见，在这个 DQN 的初步实现中，离散代表动作空间。
# 每个离散的行动(Action)表示选择一组“已经预定义好”的链路权值组合。
