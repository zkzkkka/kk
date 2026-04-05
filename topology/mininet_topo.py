"""
Mininet 拓扑脚本。

- `legacy`: 保留原始 2 主机演示方式
- `multiflow`: 新的 4 主机多流竞争场景，默认用于 RL 实验

这个脚本主要负责“把论文里的实验场景真正跑起来”：
- 启动 Mininet 网络和远程控制器连接
- 根据场景定义创建主机、交换机和链路
- 按预设时间启动业务流和探测流
- 可选模拟链路 down/up，观察控制器是否能完成重路由
"""

import argparse
import os
import sys
from time import sleep, time

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from topology.topology_catalog import (
    DEFAULT_SCENARIO_NAME,
    available_scenarios,
    get_scenario,
    host_name_lookup,
)


def _warm_up_hosts(hosts):
    """通过短 ping 触发 ARP/主机发现，减少实验初始抖动。"""
    for src_name, src in hosts.items():
        for dst_name, dst in hosts.items():
            if src_name == dst_name:
                continue
            src.cmd(f"ping -c 1 -W 1 {dst.IP()} >/dev/null 2>&1")
            sleep(0.2)


def _start_iperf_servers(hosts, flows):
    """按场景流配置启动 iperf 服务端，避免同一端口重复拉起。"""
    server_logs = []
    started = set()
    for flow in flows:
        key = (flow.dst, flow.port)
        if key in started:
            continue
        started.add(key)

        dst = hosts[flow.dst]
        log_path = f"/tmp/{flow.dst}_{flow.port}_iperf_server.log"
        if not _is_iperf_server_running(dst, flow.port):
            dst.cmd(f"rm -f {log_path}")
            dst.cmd(f"iperf -s -p {flow.port} >{log_path} 2>&1 &")
        server_logs.append(log_path)
    return server_logs


def _is_iperf_server_running(host, port):
    """检查目标主机上指定端口的 iperf 服务端是否已存在。"""
    output = host.cmd(f'pgrep -f "iperf -s -p {port}"')
    return bool(output.strip())


def _restart_iperf_servers(hosts, flows):
    """重启全部 iperf 服务端，适合进入新一轮动态实验前重置环境。"""
    stopped = set()
    for flow in flows:
        key = (flow.dst, flow.port)
        if key in stopped:
            continue
        stopped.add(key)
        hosts[flow.dst].cmd(f'pkill -f "iperf -s -p {flow.port}" >/dev/null 2>&1')
    sleep(0.3)
    return _start_iperf_servers(hosts, flows)


def _wait_for_iperf_servers(hosts, flows, timeout_s=5.0):
    """等待所有 iperf 服务端就绪，超时后返回仍未启动的列表。"""
    pending = {(flow.dst, flow.port) for flow in flows}
    deadline = time() + timeout_s
    while pending and time() < deadline:
        ready = set()
        for dst_name, port in pending:
            if _is_iperf_server_running(hosts[dst_name], port):
                ready.add((dst_name, port))
        pending -= ready
        if pending:
            sleep(0.2)
    return sorted(pending)


def _stop_iperf_servers(hosts, flows):
    """关闭本轮场景中涉及的全部 iperf 服务端。"""
    stopped = set()
    for flow in flows:
        key = (flow.dst, flow.port)
        if key in stopped:
            continue
        stopped.add(key)
        hosts[flow.dst].cmd(f'pkill -f "iperf -s -p {flow.port}" >/dev/null 2>&1')


def _start_ping_probes(hosts, ping_pairs, total_duration):
    """启动持续 ping 探测流，用于观测时延与连通性变化。"""
    ping_logs = []
    ping_count = max(12, int((total_duration / 0.5) + 2))
    for src_name, dst_name in ping_pairs:
        src = hosts[src_name]
        dst = hosts[dst_name]
        log_path = f"/tmp/{src_name}_to_{dst_name}_ping.log"
        src.cmd(f"ping -i 0.5 -c {ping_count} {dst.IP()} >{log_path} 2>&1 &")
        ping_logs.append((src_name, dst_name, log_path))
    return ping_logs


def _start_workload_flows(hosts, flows):
    """
    按场景定义启动业务流，支持带启动延时的突发流。

    `start_after_s` 允许不同流错峰启动，这样更容易制造“突然拥塞”和“竞争加剧”的实验情形。
    """
    client_logs = []
    for flow in flows:
        src = hosts[flow.src]
        dst = hosts[flow.dst]
        log_path = f"/tmp/{flow.label}_iperf_client.log"
        command = (
            "sh -c "
            f"\"sleep {flow.start_after_s}; "
            f"iperf -c {dst.IP()} -p {flow.port} -t {flow.duration_s} -i 1 "
            f">{log_path} 2>&1\" &"
        )
        src.cmd(command)
        client_logs.append((flow.src, flow.label, log_path))
    return client_logs


def _print_demo_summaries(hosts, client_logs, ping_logs):
    """打印业务流和探测流摘要，便于实验结束后快速回看。"""
    info("*** 业务流结果摘要\n")
    for src_name, label, log_path in client_logs:
        summary = hosts[src_name].cmd(f"tail -n 8 {log_path}")
        info(f"*** {label}\n")
        if summary:
            info(summary if summary.endswith("\n") else summary + "\n")

    info("*** 探测流结果摘要\n")
    for src_name, dst_name, log_path in ping_logs:
        summary = hosts[src_name].cmd(f"tail -n 5 {log_path}")
        info(f"*** {src_name} -> {dst_name}\n")
        if summary:
            info(summary if summary.endswith("\n") else summary + "\n")


def _run_dynamic_demo(net, hosts, scenario, args):
    """
    自动执行链路 down/up 演示，供控制器观察并学习重路由行为。

    从论文实验视角看，这里就是“环境”在主动制造事件：
    业务流先跑起来，随后某条链路故障，再恢复，
    看控制器是否能把流量重新导向更合适的路径。
    """
    link_a, link_b = tuple(args.demo_link or scenario["demo_link"])
    configured_duration = max(
        flow.start_after_s + flow.duration_s for flow in scenario["flows"]
    )
    total_duration = max(
        args.demo_total_duration or scenario["demo_total_duration"],
        configured_duration,
        args.demo_down_after + args.demo_down_duration + 5,
    )
    repeat = max(1, args.demo_repeat)
    gap = max(0, args.demo_repeat_gap)

    info(
        (
            f"*** 自动动态链路实验: scenario={args.scenario}, {link_a} <-> {link_b}, "
            f"repeat={repeat}\n"
        )
    )
    # 动态实验之前先确保服务端统一处于干净、可用的状态。
    _restart_iperf_servers(hosts, scenario["flows"])
    pending_servers = _wait_for_iperf_servers(hosts, scenario["flows"])
    if pending_servers:
        raise RuntimeError(f"iperf 服务端未按时就绪: {pending_servers}")

    for round_index in range(repeat):
        round_no = round_index + 1
        info(f"*** 动态实验轮次 {round_no}/{repeat}\n")
        net.configLinkStatus(link_a, link_b, "up")
        # 连续多轮实验中保持服务端常驻，只在检测到异常退出时补拉，避免竞态导致 Connection refused。
        _start_iperf_servers(hosts, scenario["flows"])
        pending_servers = _wait_for_iperf_servers(hosts, scenario["flows"])
        if pending_servers:
            raise RuntimeError(f"第 {round_no} 轮 iperf 服务端未就绪: {pending_servers}")
        sleep(0.3)

        # 先启动探测流，再启动业务流，这样能更完整地覆盖链路故障前后的全过程。
        ping_logs = _start_ping_probes(hosts, scenario["ping_pairs"], total_duration)
        client_logs = _start_workload_flows(hosts, scenario["flows"])

        sleep(max(1, args.demo_down_after))
        info(f"*** [round {round_no}] 模拟链路故障: link {link_a} {link_b} down\n")
        net.configLinkStatus(link_a, link_b, "down")

        sleep(max(1, args.demo_down_duration))
        info(f"*** [round {round_no}] 恢复链路: link {link_a} {link_b} up\n")
        net.configLinkStatus(link_a, link_b, "up")

        remaining = max(0, total_duration - args.demo_down_after - args.demo_down_duration)
        if remaining > 0:
            sleep(remaining)
        sleep(2)

        _print_demo_summaries(hosts, client_logs, ping_logs)
        if round_no < repeat and gap > 0:
            info(f"*** 等待下一轮动态实验: {gap} 秒\n")
            sleep(gap)
    _stop_iperf_servers(hosts, scenario["flows"])


def create_topology(args):
    """
    按选定场景创建 Mininet 拓扑，并可选执行自动动态演示。

    注意这个函数做了两件事：
    - 创建和启动基础网络
    - 决定是否进入自动实验，还是交给用户手动在 CLI 里操作
    """
    scenario_name, scenario = get_scenario(args.scenario)
    host_specs = scenario["hosts"]
    host_specs_by_name = host_name_lookup(scenario)
    switch_links = scenario["switch_links"]

    net = Mininet(controller=RemoteController, switch=OVSSwitch, link=TCLink)

    info("*** 添加控制器\n")
    c0 = net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)

    info("*** 添加交换机\n")
    # 交换机命名与 topology_catalog 中的 s1~s6 保持一致，方便控制器和拓扑共用场景定义。
    switches = {
        f"s{index}": net.addSwitch(f"s{index}", protocols="OpenFlow13")
        for index in range(1, 7)
    }

    info("*** 添加主机\n")
    hosts = {}
    for host in host_specs:
        hosts[host.name] = net.addHost(host.name, mac=host.mac, ip=host.ip)

    info("*** 创建链路\n")
    for host in host_specs:
        net.addLink(hosts[host.name], switches[host.switch], bw=host.access_bw_mbps)

    for link in switch_links:
        # 交换机间链路使用带宽与时延参数，供 Mininet 施加链路约束。
        net.addLink(
            switches[link.src],
            switches[link.dst],
            bw=link.bw_mbps,
            delay=f"{link.delay_ms}ms",
        )

    info("*** 启动网络\n")
    net.build()
    c0.start()
    for switch in switches.values():
        switch.start([c0])

    info("*** 等待交换机连接控制器\n")
    net.waitConnected(timeout=10)
    info("*** 等待链路发现稳定\n")
    sleep(3)
    info("*** 预热主机发现与流表安装\n")
    _warm_up_hosts({name: hosts[name] for name in host_specs_by_name.keys()})

    try:
        if args.auto_dynamic_demo:
            _run_dynamic_demo(net, hosts, scenario, args)

        if args.exit_after_demo and args.auto_dynamic_demo:
            info("*** 自动动态链路实验已结束，按参数要求退出\n")
            return

        info("*** 运行命令行界面 (CLI)\n")
        CLI(net)
    finally:
        info("*** 停止网络\n")
        net.stop()


def parse_args():
    """解析命令行参数，支持切换场景与自动链路故障演示。"""
    parser = argparse.ArgumentParser(description="启动 Mininet 拓扑并可选执行动态链路演示。")
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO_NAME,
        choices=available_scenarios(),
        help="实验场景，默认使用 multiflow。",
    )
    parser.add_argument(
        "--auto-dynamic-demo",
        action="store_true",
        help="启动拓扑后自动执行一轮场景化动态链路实验。",
    )
    parser.add_argument(
        "--demo-link",
        nargs=2,
        metavar=("SW1", "SW2"),
        help="自动动态链路实验中要故障/恢复的链路端点。",
    )
    parser.add_argument(
        "--demo-down-after",
        type=int,
        default=12,
        help="自动动态链路实验中在多少秒后执行 link down。",
    )
    parser.add_argument(
        "--demo-down-duration",
        type=int,
        default=10,
        help="自动动态链路实验中链路保持 down 的秒数。",
    )
    parser.add_argument(
        "--demo-total-duration",
        type=int,
        help="自动动态链路实验总时长；未指定时使用场景默认值。",
    )
    parser.add_argument(
        "--demo-repeat",
        type=int,
        default=1,
        help="自动动态链路实验重复轮次，适合持续训练时累积更多 active 样本。",
    )
    parser.add_argument(
        "--demo-repeat-gap",
        type=int,
        default=3,
        help="自动动态链路实验多轮之间的间隔秒数。",
    )
    parser.add_argument(
        "--exit-after-demo",
        action="store_true",
        help="自动动态链路实验完成后直接退出，不进入 Mininet CLI。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setLogLevel("info")
    create_topology(parse_args())
