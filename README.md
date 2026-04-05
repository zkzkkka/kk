# 基于强化学习的 SDN 路由实验系统

本项目是一个毕业设计实验项目，围绕 `Ryu + Mininet + PyTorch` 搭建强化学习驱动的 SDN 路由实验环境，用于对比 `RL` 与 `static` 路由策略在多流竞争、链路波动等场景下的表现。

## 项目结构

```text
毕设/
├── agent/                  # 强化学习模型与训练逻辑
├── control_plane/          # Ryu 控制器、监控与路径计算
├── topology/               # Mininet 拓扑与实验场景
├── visualization/          # 结果可视化脚本
├── checkpoints/            # 默认模型权重
├── metrics/                # 实验日志与统计结果
├── tests/                  # 单元测试
├── 运行步骤.md
└── 项目架构与模块功能.md
```

## 主要功能

- 使用 `Mininet` 构造 `legacy`、`multiflow`、`complexflow` 三类实验拓扑。
- 使用 `Ryu` 控制器完成拓扑发现、链路监控、路径计算与流表下发。
- 使用 DQN 智能体根据网络状态动态选择路由权重策略。
- 支持 `train` 与 `eval` 两种模式，并保存/恢复 checkpoint。
- 支持生成训练曲线和 RL/static 对比图。

## 运行环境

- Ubuntu / Linux
- Python 3
- Mininet
- Open vSwitch
- Ryu
- PyTorch

详细运行方式请查看：

- [运行步骤.md](./运行步骤.md)
- [项目架构与模块功能.md](./项目架构与模块功能.md)

## 快速开始

1. 进入项目目录并激活虚拟环境

```bash
cd /home/ubuntu/毕设
source .venv/bin/activate
```

2. 启动控制器

```bash
TOPOLOGY_SCENARIO=multiflow ryu-manager control_plane/ryu_main.py control_plane/network_monitor.py
```

3. 启动拓扑

```bash
sudo python3 topology/mininet_topo.py --scenario multiflow
```

4. 运行测试

```bash
pingall
h2 iperf -s &
h1 iperf -c 10.0.0.2 -t 20
```

