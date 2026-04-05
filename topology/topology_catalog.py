from __future__ import annotations

"""
集中维护实验场景、主机规格与链路规格。

这个文件相当于项目里的“实验真值库”：
- Mininet 用它创建网络
- 控制器用它理解场景里的链路和主机
- 测试也用它验证拓扑规模是否符合预期
"""

from dataclasses import dataclass


DEFAULT_SCENARIO_NAME = "multiflow"


@dataclass(frozen=True)
class HostSpec:
    """描述一个主机在拓扑中的接入位置与地址信息。"""

    name: str
    switch: str
    mac: str
    ip: str
    access_bw_mbps: float = 1000.0


@dataclass(frozen=True)
class SwitchLinkSpec:
    """描述一条交换机之间链路的带宽与时延。"""

    src: str
    dst: str
    bw_mbps: float
    delay_ms: float


@dataclass(frozen=True)
class FlowSpec:
    """
    描述自动演示时启动的一条业务流。

    业务流配置不仅决定“谁和谁通信”，还决定：
    - 从什么时候开始发
    - 持续多久
    - 用哪个端口区分不同 iperf 会话
    """

    src: str
    dst: str
    port: int
    start_after_s: int
    duration_s: int
    label: str


# 基础 6 交换机骨干拓扑，作为 legacy 与 multiflow 的公共底座。
BASE_SWITCH_LINKS = (
    SwitchLinkSpec("s1", "s2", 20.0, 2.0),
    SwitchLinkSpec("s1", "s3", 10.0, 5.0),
    SwitchLinkSpec("s2", "s3", 15.0, 3.0),
    SwitchLinkSpec("s2", "s4", 20.0, 2.0),
    SwitchLinkSpec("s3", "s4", 10.0, 5.0),
    SwitchLinkSpec("s3", "s5", 15.0, 4.0),
    SwitchLinkSpec("s4", "s5", 20.0, 2.0),
    SwitchLinkSpec("s4", "s6", 25.0, 1.0),
    SwitchLinkSpec("s5", "s6", 15.0, 5.0),
)

# complexflow 在基础拓扑上增加更多旁路链路，便于观察更复杂的重路由行为。
COMPLEX_SWITCH_LINKS = BASE_SWITCH_LINKS + (
    SwitchLinkSpec("s1", "s4", 12.0, 6.0),
    SwitchLinkSpec("s2", "s5", 12.0, 4.0),
    SwitchLinkSpec("s3", "s6", 9.0, 6.0),
)

# 每个场景同时定义“拓扑结构”和“实验业务流配置”，这样 Mininet 与控制器可共用同一份真值。
# 这样做的好处是：论文里描述的实验一旦改动，只需要改这里一处。
SCENARIOS = {
    "legacy": {
        "switch_links": BASE_SWITCH_LINKS,
        "hosts": (
            HostSpec("h1", "s1", "00:00:00:00:00:01", "10.0.0.1/24"),
            HostSpec("h2", "s6", "00:00:00:00:00:02", "10.0.0.2/24"),
        ),
        "flows": (
            FlowSpec("h1", "h2", 5001, 0, 35, "h1_to_h2_long"),
        ),
        "ping_pairs": (("h1", "h2"),),
        "demo_link": ("s4", "s6"),
        "demo_total_duration": 35,
    },
    "multiflow": {
        "switch_links": BASE_SWITCH_LINKS,
        "hosts": (
            HostSpec("h1", "s1", "00:00:00:00:00:01", "10.0.0.1/24"),
            HostSpec("h2", "s6", "00:00:00:00:00:02", "10.0.0.2/24"),
            HostSpec("h3", "s2", "00:00:00:00:00:03", "10.0.0.3/24"),
            HostSpec("h4", "s5", "00:00:00:00:00:04", "10.0.0.4/24"),
        ),
        "flows": (
            FlowSpec("h1", "h2", 5001, 0, 45, "h1_to_h2_long"),
            FlowSpec("h3", "h4", 5002, 0, 45, "h3_to_h4_long"),
            FlowSpec("h1", "h4", 5003, 8, 16, "h1_to_h4_burst"),
            FlowSpec("h3", "h2", 5004, 16, 16, "h3_to_h2_burst"),
        ),
        "ping_pairs": (
            ("h1", "h2"),
            ("h3", "h4"),
        ),
        "demo_link": ("s4", "s6"),
        "demo_total_duration": 45,
    },
    "complexflow": {
        "switch_links": COMPLEX_SWITCH_LINKS,
        "hosts": (
            HostSpec("h1", "s1", "00:00:00:00:00:01", "10.0.0.1/24"),
            HostSpec("h2", "s6", "00:00:00:00:00:02", "10.0.0.2/24"),
            HostSpec("h3", "s2", "00:00:00:00:00:03", "10.0.0.3/24"),
            HostSpec("h4", "s5", "00:00:00:00:00:04", "10.0.0.4/24"),
            HostSpec("h5", "s3", "00:00:00:00:00:05", "10.0.0.5/24"),
            HostSpec("h6", "s4", "00:00:00:00:00:06", "10.0.0.6/24"),
        ),
        "flows": (
            FlowSpec("h1", "h2", 5001, 0, 55, "h1_to_h2_long"),
            FlowSpec("h3", "h4", 5002, 0, 55, "h3_to_h4_long"),
            FlowSpec("h5", "h6", 5003, 0, 55, "h5_to_h6_long"),
            FlowSpec("h1", "h4", 5004, 8, 20, "h1_to_h4_burst"),
            FlowSpec("h3", "h2", 5005, 16, 20, "h3_to_h2_burst"),
            FlowSpec("h5", "h2", 5006, 24, 18, "h5_to_h2_burst"),
        ),
        "ping_pairs": (
            ("h1", "h2"),
            ("h3", "h4"),
            ("h5", "h6"),
        ),
        "demo_link": ("s4", "s6"),
        "demo_total_duration": 55,
    },
}


def available_scenarios():
    """返回所有可选场景名，供命令行参数约束使用。"""
    return tuple(sorted(SCENARIOS.keys()))


def get_scenario(name: str):
    """按名称获取场景；如果名称非法则回退到默认场景。"""
    scenario_name = (name or DEFAULT_SCENARIO_NAME).strip().lower()
    if scenario_name not in SCENARIOS:
        scenario_name = DEFAULT_SCENARIO_NAME
    return scenario_name, SCENARIOS[scenario_name]


def switch_name_to_dpid(name: str) -> int:
    """把 Mininet 交换机名 s1/s2/... 转成 Ryu 使用的整数 dpid。"""
    return int(name.lstrip("s"))


def host_name_lookup(scenario):
    """建立 host_name -> HostSpec 的快速索引。"""
    return {host.name: host for host in scenario["hosts"]}


def host_mac_lookup(scenario):
    """建立 host_mac -> HostSpec 的快速索引。"""
    return {host.mac: host for host in scenario["hosts"]}


def build_directed_link_catalog(switch_links=None):
    """
    把无向链路规格展开为双向链路字典，便于控制器按方向查询。

    因为控制器在运行时通常按 `(src_dpid, dst_dpid)` 查询链路属性，
    所以这里会把 `s1-s2` 展开成 `(1,2)` 和 `(2,1)` 两项。
    """
    switch_links = BASE_SWITCH_LINKS if switch_links is None else switch_links
    directed_links = {}
    for link in switch_links:
        src = switch_name_to_dpid(link.src)
        dst = switch_name_to_dpid(link.dst)
        payload = {
            "bw_mbps": float(link.bw_mbps),
            "delay_ms": float(link.delay_ms),
        }
        directed_links[(src, dst)] = dict(payload)
        directed_links[(dst, src)] = dict(payload)
    return directed_links


def expected_host_pair_count(scenario):
    """
    返回该场景中理论上的有向主机对数量。

    例如 4 台主机时：
    - 无向主机对是 6 对
    - 有向主机对是 4 * 3 = 12 对

    控制器内部按“源主机 -> 目的主机”建路径，所以这里使用有向计数。
    """
    host_count = len(scenario["hosts"])
    return max(1, host_count * max(0, host_count - 1))
