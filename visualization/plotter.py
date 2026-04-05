"""
性能评估与可视化层 (Performance Evaluation and Visualization Layer)
读取控制器输出的真实实验日志，并绘制学习曲线与基线对比图。
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
METRICS_DIR = os.path.join(PROJECT_ROOT, "metrics")
METRIC_LABELS = {
    "reward": "Reward",
    "max_utilization": "Max Link Utilization",
    "mean_utilization": "Mean Link Utilization",
    "max_drop_rate": "Max Drop Rate",
    "mean_drop_rate": "Mean Drop Rate",
    "avg_hops": "Average Hop Count",
    "path_churn": "Path Churn",
    "path_delay_ms": "Average Path Delay (ms)",
    "path_delay_norm": "Average Path Delay (Normalized)",
    "observed_path_delay_ms": "Observed Path Delay (ms)",
    "observed_path_delay_norm": "Observed Path Delay (Normalized)",
    "util_delta": "Mean Absolute Utilization Delta",
    "mean_rate_bps": "Mean Link Rate (bps)",
    "max_rate_bps": "Max Link Rate (bps)",
}


def _load_metrics(log_path, include_inactive=False):
    """
    读取 jsonl 指标日志，并筛出路由决策样本。

    默认只保留 active 阶段，是因为 warmup/teardown 往往包含：
    - 主机尚未完全发现
    - 链路还没稳定
    - 拓扑正在拆除
    这些样本会干扰对真正路由性能的判断。
    """
    records = []
    with open(log_path, "r", encoding="utf-8") as metrics_file:
        for line in metrics_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("step_type") != "routing_decision":
                continue
            if not include_inactive:
                if record.get("experiment_phase") != "active":
                    continue
                if record.get("links", 0) <= 0:
                    continue
            records.append(record)
    return records


def _latest_log(mode):
    """返回指定模式下时间戳最新的一份日志。"""
    pattern = os.path.join(METRICS_DIR, f"{mode}_metrics_*.jsonl")
    candidates = sorted(glob.glob(pattern))
    return candidates[-1] if candidates else None


def _extract_series(records, metric):
    """从记录数组中抽取某一个指标的时间序列。"""
    return [float(record[metric]) for record in records if record.get(metric) is not None]


def _moving_average(values, window_size=5):
    """对序列做简单滑动平均，便于观察整体趋势。"""
    if not values:
        return []
    if len(values) < window_size:
        return values
    kernel = np.ones(window_size) / window_size
    return np.convolve(values, kernel, mode="valid").tolist()


def plot_learning_curve(records, filename="learning_curve.png"):
    """
    绘制 RL 训练时 reward 随时间的变化曲线。

    这张图最适合回答的问题是：
    “智能体随着训练推进，是否总体上越来越能拿到更高奖励？”
    """
    rewards = _extract_series(records, "reward")
    if not rewards:
        raise ValueError("日志中没有可用于绘图的 reward 数据。")

    smoothed_rewards = _moving_average(rewards, window_size=5)
    plt.figure(figsize=(10, 5))
    # 原始曲线较噪，这里叠加 5 步滑动平均帮助观察学习趋势。
    plt.plot(rewards, label="Reward", alpha=0.35)
    if smoothed_rewards:
        x_axis = range(len(smoothed_rewards))
        if len(rewards) >= 5:
            x_axis = range(4, 4 + len(smoothed_rewards))
        plt.plot(x_axis, smoothed_rewards, label="Reward (5-step MA)", linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel(METRIC_LABELS["reward"])
    plt.title("RL Agent Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"学习曲线图表已成功保存至 {filename}")


def plot_performance_comparison(
    baseline_records,
    rl_records,
    metric="mean_utilization",
    filename="comparison.png",
):
    """
    绘制静态路由与 RL 路由在同一指标上的对比图。

    这张图更适合论文里的“基线对比”部分，用来说明 RL 是否优于传统静态策略。
    """
    baseline_data = _extract_series(baseline_records, metric)
    rl_data = _extract_series(rl_records, metric)
    if not baseline_data or not rl_data:
        raise ValueError("缺少可用于对比绘图的数据。")

    # 两组日志长度可能不同，对比时按最短公共长度截断。
    compare_len = min(len(baseline_data), len(rl_data))
    baseline_data = baseline_data[:compare_len]
    rl_data = rl_data[:compare_len]

    plt.figure(figsize=(10, 5))
    plt.plot(baseline_data, label="Static Routing", color="red")
    plt.plot(rl_data, label="RL Routing", color="blue")
    plt.xlabel("Time Step")
    plt.ylabel(METRIC_LABELS[metric])
    plt.title(f"Performance Comparison: {METRIC_LABELS[metric]}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"对比评估图表已成功保存至 {filename}")


def main():
    """命令行入口：支持绘制学习曲线、性能对比图，或两者一起绘制。"""
    parser = argparse.ArgumentParser(description="从真实实验日志绘制学习曲线与性能对比图。")
    parser.add_argument("--log", help="用于绘制学习曲线的 RL 日志文件路径。默认读取最新的 RL 日志。")
    parser.add_argument("--rl-log", help="用于性能对比的 RL 日志文件路径。默认读取最新的 RL 日志。")
    parser.add_argument(
        "--baseline-log",
        help="用于性能对比的静态基线路由日志文件路径。默认读取最新的 static 日志。",
    )
    parser.add_argument(
        "--metric",
        default="mean_utilization",
        choices=sorted(METRIC_LABELS.keys()),
        help="性能对比图使用的指标。",
    )
    parser.add_argument("--learning-curve-out", default="learning_curve.png")
    parser.add_argument("--comparison-out", default="comparison.png")
    parser.add_argument("--skip-learning-curve", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="保留 warmup/teardown 等非 active 阶段样本，用于调试原始日志。",
    )
    args = parser.parse_args()

    if not args.skip_learning_curve:
        learning_log = args.log or _latest_log("rl")
        if learning_log:
            plot_learning_curve(
                _load_metrics(learning_log, include_inactive=args.include_inactive),
                filename=args.learning_curve_out,
            )
        else:
            print("未找到 RL 日志，跳过学习曲线绘制。")

    if not args.skip_comparison:
        rl_log = args.rl_log or args.log or _latest_log("rl")
        baseline_log = args.baseline_log or _latest_log("static")
        if rl_log and baseline_log:
            plot_performance_comparison(
                _load_metrics(baseline_log, include_inactive=args.include_inactive),
                _load_metrics(rl_log, include_inactive=args.include_inactive),
                metric=args.metric,
                filename=args.comparison_out,
            )
        else:
            print("未同时找到 RL 日志和 static 日志，跳过性能对比图绘制。")


if __name__ == "__main__":
    main()
