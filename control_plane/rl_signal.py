from __future__ import annotations

"""路由动作空间、奖励函数与静态链路特征定义。"""

from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass(frozen=True)
class RoutingProfile:
    """
    一个离散动作对应的一组链路加权偏好参数。

    这里的动作不是“直接指定完整路径”，而是指定一套“链路打分偏好”：
    - util_alpha 越大，越倾向绕开高利用率链路
    - capacity_alpha 越大，越倾向选择高容量链路
    最终这些偏好会被转换成链路权重，再交给最短路算法计算路径。
    """

    profile_id: int
    util_alpha: float
    capacity_alpha: float
    name: str


@dataclass(frozen=True)
class RewardConfig:
    """
    奖励函数中的各项权重，便于通过环境变量做实验调参。

    奖励函数并不是唯一标准答案，而是“告诉智能体什么是好路由”的设计选择。
    这些权重越大，对应指标在总奖励中的影响就越强。
    """

    max_util_gain_weight: float = 0.7
    mean_util_gain_weight: float = 0.25
    util_delta_gain_weight: float = 0.1
    overload_penalty_weight: float = 1.2
    overload_threshold: float = 0.8
    hop_penalty_weight: float = 0.05
    churn_penalty_weight: float = 0.05
    drop_rate_penalty_weight: float = 0.2
    path_delay_penalty_weight: float = 0.05


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_reward_config() -> RewardConfig:
    """读取一次环境变量并缓存，避免训练过程中重复解析。"""
    return RewardConfig(
        max_util_gain_weight=_env_float(
            "RL_REWARD_MAX_UTIL_GAIN_WEIGHT",
            RewardConfig.max_util_gain_weight,
        ),
        mean_util_gain_weight=_env_float(
            "RL_REWARD_MEAN_UTIL_GAIN_WEIGHT",
            RewardConfig.mean_util_gain_weight,
        ),
        util_delta_gain_weight=_env_float(
            "RL_REWARD_UTIL_DELTA_GAIN_WEIGHT",
            RewardConfig.util_delta_gain_weight,
        ),
        overload_penalty_weight=_env_float(
            "RL_REWARD_OVERLOAD_PENALTY_WEIGHT",
            RewardConfig.overload_penalty_weight,
        ),
        overload_threshold=_env_float(
            "RL_REWARD_OVERLOAD_THRESHOLD",
            RewardConfig.overload_threshold,
        ),
        hop_penalty_weight=_env_float(
            "RL_REWARD_HOP_PENALTY_WEIGHT",
            RewardConfig.hop_penalty_weight,
        ),
        churn_penalty_weight=_env_float(
            "RL_REWARD_CHURN_PENALTY_WEIGHT",
            RewardConfig.churn_penalty_weight,
        ),
        drop_rate_penalty_weight=_env_float(
            "RL_REWARD_DROP_RATE_PENALTY_WEIGHT",
            RewardConfig.drop_rate_penalty_weight,
        ),
        path_delay_penalty_weight=_env_float(
            "RL_REWARD_PATH_DELAY_PENALTY_WEIGHT",
            RewardConfig.path_delay_penalty_weight,
        ),
    )


# 动作空间通过 util_alpha/capacity_alpha 的笛卡尔积得到。
UTIL_ALPHA_LEVELS = (0.0, 1.0, 3.0)
CAPACITY_ALPHA_LEVELS = (0.0, 0.5, 1.0)


def build_routing_profiles():
    """
    生成全部离散路由动作，供 DQN 在固定动作空间中选择。

    DQN 更适合离散动作空间，所以这里不让智能体直接输出连续权重，
    而是先人工列出若干组常见偏好参数，再让智能体从中选择。
    """
    profiles = []
    profile_id = 0
    for util_alpha in UTIL_ALPHA_LEVELS:
        for capacity_alpha in CAPACITY_ALPHA_LEVELS:
            profiles.append(
                RoutingProfile(
                    profile_id=profile_id,
                    util_alpha=float(util_alpha),
                    capacity_alpha=float(capacity_alpha),
                    name=f"util{util_alpha:g}_cap{capacity_alpha:g}",
                )
            )
            profile_id += 1
    return tuple(profiles)


ROUTING_PROFILES = build_routing_profiles()


def build_static_link_features(directed_links):
    """
    把链路带宽和时延做归一化，作为状态中的静态特征。

    这些特征本身不会在实验过程中频繁变化，但它们能告诉智能体：
    某条链路天生是“更粗”“更快”还是“更慢”。
    """
    max_bw = max((payload["bw_mbps"] for payload in directed_links.values()), default=1.0)
    max_delay = max((payload["delay_ms"] for payload in directed_links.values()), default=1.0)
    features = {}
    for edge, payload in directed_links.items():
        features[edge] = {
            "bw_mbps": float(payload["bw_mbps"]),
            "delay_ms": float(payload["delay_ms"]),
            "capacity_norm": float(payload["bw_mbps"]) / max_bw if max_bw else 0.0,
            "delay_norm": float(payload["delay_ms"]) / max_delay if max_delay else 0.0,
        }
    return features


def compute_profile_weight(profile, util_norm, capacity_norm, delay_norm):
    """
    根据 profile 偏好把一条链路映射成最短路算法使用的权重。

    注意这里是“代价”而不是“分数”：
    - 代价越低，最短路算法越愿意走这条链路
    - 代价越高，最短路算法越倾向绕开它
    """
    return float(
        delay_norm
        + (profile.util_alpha * util_norm)
        + (profile.capacity_alpha * (1.0 - capacity_norm))
    )


def compute_path_churn(previous_paths, next_paths):
    """
    计算两轮路径映射之间发生变化的主机对比例。

    路径 churn 越高，说明控制器越频繁地改路。
    适度改路可以缓解拥塞，但过度改路会带来抖动，因此奖励里通常会惩罚它。
    """
    keys = sorted(set(previous_paths.keys()) | set(next_paths.keys()))
    if not keys:
        return 0.0

    changed = sum(1 for key in keys if previous_paths.get(key) != next_paths.get(key))
    return float(changed) / float(len(keys))


def compute_reward(
    previous_metrics,
    current_metrics,
    hop_cost,
    path_churn,
    reward_config: RewardConfig | None = None,
):
    """
    计算一步动作后的即时奖励。

    奖励设计思路：
    - 鼓励降低最大/平均利用率和波动
    - 惩罚过载、路径过长、频繁切换、丢包与时延
    """
    # previous_metrics 描述“执行旧动作之前/上一轮”的网络表现，
    # current_metrics 描述“执行旧动作之后/这一轮观测到”的网络表现。
    # 因此 reward 实际上是在评价“上一轮动作”好不好。
    reward_config = get_reward_config() if reward_config is None else reward_config
    previous_max_util = float(previous_metrics.get("max_utilization", 0.0))
    current_max_util = float(current_metrics.get("max_utilization", 0.0))
    previous_mean_util = float(previous_metrics.get("mean_utilization", 0.0))
    current_mean_util = float(current_metrics.get("mean_utilization", 0.0))
    previous_util_delta = float(previous_metrics.get("util_delta", 0.0))
    current_util_delta = float(current_metrics.get("util_delta", 0.0))
    current_mean_drop_rate = float(current_metrics.get("mean_drop_rate", 0.0))
    current_path_delay_norm = float(current_metrics.get("path_delay_norm", 0.0))

    # 先把阈值裁剪到 [0, 1]，再计算超载量和平方惩罚。
    overload_threshold = min(max(reward_config.overload_threshold, 0.0), 1.0)
    overload_excess = max(0.0, current_max_util - overload_threshold)
    overload_penalty = overload_excess * overload_excess

    # 奖励的正项表示“比之前更好”，负项表示“需要付出的代价”。
    # 最终返回的是一个标量，供 DQN 把它当作即时回报。
    return float(
        reward_config.max_util_gain_weight * (previous_max_util - current_max_util)
        + reward_config.mean_util_gain_weight * (previous_mean_util - current_mean_util)
        + reward_config.util_delta_gain_weight * (previous_util_delta - current_util_delta)
        - reward_config.overload_penalty_weight * overload_penalty
        - reward_config.hop_penalty_weight * hop_cost
        - reward_config.churn_penalty_weight * path_churn
        - reward_config.drop_rate_penalty_weight * current_mean_drop_rate
        - reward_config.path_delay_penalty_weight * current_path_delay_norm
    )
