"""
Ryu 主控制器。

职责分层：
- 监听拓扑/主机事件并维护当前网络视图
- 周期性构造 RL 状态，调用 Agent 选择路由动作
- 根据动作更新链路权重、重算最短路并下发流表
- 记录指标并管理训练/评估模式与 checkpoint
"""

from collections import defaultdict
import json
import os
import sys
import time

import numpy as np
from ryu import cfg
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import ethernet, ether_types, packet
from ryu.ofproto import ofproto_v1_3
from ryu.topology import event
from ryu.topology.api import get_all_host, get_all_switch
import ryu.topology.switches  # Registers observe-links config option.

# Make the project root importable so we can import agent/ from control_plane.
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agent.train import Agent, AgentConfig
from control_plane.flow_installer import install_path_flows
from control_plane.rl_signal import (
    ROUTING_PROFILES,
    build_static_link_features,
    compute_path_churn,
    compute_profile_weight,
    compute_reward,
)
from control_plane.shortest_path import ShortestPathCalculator
from topology.topology_catalog import (
    DEFAULT_SCENARIO_NAME,
    build_directed_link_catalog,
    expected_host_pair_count,
    get_scenario,
    host_mac_lookup,
    switch_name_to_dpid,
)

# Keep link discovery enabled even if ryu-manager is started without --observe-links.
cfg.CONF.set_override("observe_links", True)
app_manager.require_app("ryu.topology.switches", api_style=True)


def _env_int(name, default, minimum=None):
    """读取整数环境变量，并可选限制最小值。"""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def _env_bool(name, default):
    """读取布尔环境变量，兼容常见关闭写法。"""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class RLRoutingController(app_manager.RyuApp):
    """融合拓扑感知、路由决策和强化学习训练逻辑的 Ryu 控制器。"""

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    ROUTE_FLOW_COOKIE = 0x524C0001
    ROUTE_FLOW_PRIORITY = 10
    ACTION_PROFILES = ROUTING_PROFILES

    def __init__(self, *args, **kwargs):
        super(RLRoutingController, self).__init__(*args, **kwargs)
        self.datapaths = {}

        # Topology adjacency: adjacency[src_dpid][dst_dpid] = src_port_no
        self.adjacency = defaultdict(dict)
        self.inter_switch_ports = defaultdict(set)

        # 场景目录同时给出拓扑、主机和流定义，控制器直接复用这一份配置。
        scenario_env = os.environ.get("TOPOLOGY_SCENARIO", DEFAULT_SCENARIO_NAME)
        self.scenario_name, self.scenario = get_scenario(scenario_env)
        self.host_spec_by_mac = host_mac_lookup(self.scenario)
        self.host_spec_by_switch = {
            switch_name_to_dpid(host.switch): host for host in self.scenario["hosts"]
        }
        self.directed_link_catalog = build_directed_link_catalog(self.scenario["switch_links"])
        self.link_static_features = build_static_link_features(self.directed_link_catalog)
        self.edge_order = tuple(sorted(self.directed_link_catalog.keys()))
        # 每条有向链路贡献 6 个动态/静态特征，末尾再拼接 2 个全局特征。
        self.state_features_per_edge = 6
        self.configured_link_count = len(self.edge_order)
        self.expected_host_count = len(self.scenario["hosts"])
        self.max_host_pair_count = expected_host_pair_count(self.scenario)
        self.max_reasonable_hops = max(
            1,
            len({node for edge in self.edge_order for node in edge}) - 1,
        )
        self.max_reasonable_path_delay_ms = max(
            1.0,
            max((payload["delay_ms"] for payload in self.directed_link_catalog.values()), default=1.0)
            * self.max_reasonable_hops,
        )

        # Host attachment: host_mac -> (dpid, port_no)
        self.hosts = {}

        # sp 是真正执行“选路计算”的最短路对象；
        # RL 智能体并不直接输出路径，而是通过改变 link_weights 间接影响最短路结果。
        self.sp = ShortestPathCalculator()
        self.routing_mode = os.environ.get("RL_ROUTING_MODE", "rl").strip().lower()
        if self.routing_mode not in {"rl", "static"}:
            self.routing_mode = "rl"
        self.agent_mode = os.environ.get("RL_AGENT_MODE", "train").strip().lower()
        if self.agent_mode not in {"train", "eval"}:
            self.agent_mode = "train"

        # 状态维度 = 每条有向链路的 6 个特征 * 链路数 + 2 个全局特征。
        self.state_size = (len(self.edge_order) * self.state_features_per_edge) + 2
        # 动作空间不是“直接选路径”，而是从一组预定义 profile 中选一个。
        self.action_size = len(self.ACTION_PROFILES)
        self.checkpoints_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        default_checkpoint_path = os.path.join(
            self.checkpoints_dir,
            f"{self.scenario_name}_rl_agent.pt",
        )
        self.agent_config = (
            AgentConfig.from_env(checkpoint_path=default_checkpoint_path)
            if self.routing_mode == "rl"
            else None
        )
        self.agent_checkpoint_path = (
            self.agent_config.checkpoint_path if self.agent_config is not None else None
        )
        # 下面这些参数控制“如何训练”和“多久插入一次评估窗口”。
        self.auto_load_checkpoint = _env_bool("RL_AUTO_LOAD_CHECKPOINT", True)
        self.checkpoint_save_interval = _env_int("RL_CHECKPOINT_SAVE_INTERVAL", 10, minimum=1)
        self.eval_interval_active_steps = _env_int(
            "RL_EVAL_INTERVAL_ACTIVE_STEPS",
            50,
            minimum=0,
        )
        self.eval_window_active_steps = _env_int(
            "RL_EVAL_WINDOW_ACTIVE_STEPS",
            10,
            minimum=0,
        )
        self.decision_hold_steps = _env_int("RL_DECISION_HOLD_STEPS", 1, minimum=1)
        self.agent = (
            Agent(self.state_size, self.action_size, config=self.agent_config)
            if self.routing_mode == "rl"
            else None
        )

        # current_action/current_profile 表示当前生效的路由偏好配置。
        self.current_action = 0
        self.current_profile = self.ACTION_PROFILES[self.current_action]

        # 这些“上一轮”变量用于在下一轮状态到来时计算奖励并形成一条 transition。
        self._last_state = None
        self._last_action = None
        self._last_state_metrics = None
        self._last_experiment_phase = "warmup"
        self._last_agent_phase = "warmup"
        self._last_path_churn = 0.0
        self._last_observed_utilization = {edge: 0.0 for edge in self.edge_order}
        self._installed_paths = {}
        self._eval_steps_remaining = 0
        self._train_active_steps_since_eval = 0
        self._hold_steps_remaining = 0
        self._last_checkpoint_save_step = 0

        self._topology_ready_once = False
        self._active_phase_seen = False

        self.link_weights = {}
        self.monitor = None

        # metrics 目录会持续记录每一步路由决策，供后续画图和论文分析。
        self.metrics_dir = os.path.join(PROJECT_ROOT, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.metrics_path = os.path.join(
            self.metrics_dir,
            f"{self.routing_mode}_metrics_{self.run_id}.jsonl",
        )
        self.latest_metrics_path = os.path.join(
            self.metrics_dir,
            f"{self.routing_mode}_metrics_latest.json",
        )
        # 强化学习循环单独跑在线程里，持续每隔一段时间做一次路由决策。
        self.rl_thread = hub.spawn(self._rl_loop)
        if self.agent is not None and self.auto_load_checkpoint:
            try:
                restored = self.agent.load_checkpoint(self.agent_checkpoint_path)
                if restored:
                    self._last_checkpoint_save_step = self.agent.train_steps
                    self.logger.info(
                        "Restored RL checkpoint: path=%s epsilon=%.4f train_steps=%s replay=%s",
                        self.agent_checkpoint_path,
                        self.agent.epsilon,
                        self.agent.train_steps,
                        self.agent.replay_size,
                    )
                elif self.agent_mode == "eval":
                    self.logger.warning(
                        "RL_AGENT_MODE=eval but checkpoint does not exist: %s",
                        self.agent_checkpoint_path,
                    )
            except Exception as exc:
                self.logger.warning("Failed to restore RL checkpoint: %s", exc)
        self.logger.info(
            (
                "Routing mode: %s, scenario: %s, metrics log: %s, "
                "agent_mode: %s, checkpoint: %s, replay(batch=%s,min=%s), eval(interval=%s,window=%s)"
            ),
            self.routing_mode,
            self.scenario_name,
            self.metrics_path,
            self.agent_mode,
            self.agent_checkpoint_path,
            self.agent_config.batch_size if self.agent_config is not None else None,
            self.agent_config.min_replay_size if self.agent_config is not None else None,
            self.eval_interval_active_steps,
            self.eval_window_active_steps,
        )

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """交换机建连后安装 table-miss 规则，把未知流量上送控制器。"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.datapaths[datapath.id] = datapath

        match = parser.OFPMatch()
        actions = [
            parser.OFPActionOutput(
                ofproto.OFPP_CONTROLLER,
                ofproto.OFPCML_NO_BUFFER,
            )
        ]
        self.add_flow(datapath, 0, match, actions, cookie=0)
        self.logger.info("已注册交换机特性 DPID: %s", datapath.id)

    @set_ev_cls(event.EventLinkAdd)
    def _link_add_handler(self, ev):
        """链路上线时更新邻接关系、同步监控容量并触发重计算。"""
        link = ev.link
        self.adjacency[link.src.dpid][link.dst.dpid] = link.src.port_no
        self.inter_switch_ports[link.src.dpid].add(link.src.port_no)
        self.inter_switch_ports[link.dst.dpid].add(link.dst.port_no)
        self._sync_monitor_capacity(link, active=True)
        self._sync_hosts_from_topology()
        self._handle_topology_change("link_add", link)

        self.logger.info(
            "Link add: %s:%s -> %s:%s",
            link.src.dpid,
            link.src.port_no,
            link.dst.dpid,
            link.dst.port_no,
        )

    @set_ev_cls(event.EventLinkDelete)
    def _link_delete_handler(self, ev):
        """链路下线时清理邻接关系，并让控制器重新评估当前路由。"""
        link = ev.link
        self.adjacency.get(link.src.dpid, {}).pop(link.dst.dpid, None)
        self._sync_monitor_capacity(link, active=False)
        self._sync_hosts_from_topology()
        self._handle_topology_change("link_delete", link)

        self.logger.info(
            "Link del: %s:%s -> %s:%s",
            link.src.dpid,
            link.src.port_no,
            link.dst.dpid,
            link.dst.port_no,
        )

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, cookie=None):
        """
        向交换机下发一条流表项。

        这里封装了 Ryu 的 FlowMod 细节，外部只需要关心：
        - 匹配什么报文
        - 命中后执行什么动作
        - 优先级是多少
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if cookie is None:
            cookie = self.ROUTE_FLOW_COOKIE

        if buffer_id is not None:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                buffer_id=buffer_id,
                priority=priority,
                match=match,
                instructions=inst,
                cookie=cookie,
            )
        else:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=priority,
                match=match,
                instructions=inst,
                cookie=cookie,
            )
        datapath.send_msg(mod)

    def delete_flows(self, datapath, match=None, cookie=None, cookie_mask=0):
        """删除满足条件的流表项；常用于链路变化后的路径重装。"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        if match is None:
            match = parser.OFPMatch()

        kwargs = {
            "datapath": datapath,
            "table_id": ofproto.OFPTT_ALL,
            "command": ofproto.OFPFC_DELETE,
            "out_port": ofproto.OFPP_ANY,
            "out_group": ofproto.OFPG_ANY,
            "match": match,
        }
        if cookie is not None:
            kwargs["cookie"] = cookie
            kwargs["cookie_mask"] = cookie_mask

        datapath.send_msg(parser.OFPFlowMod(**kwargs))

    def _ensure_monitor(self):
        """懒加载 NetworkMonitor brick，避免控制器启动顺序耦合。"""
        if self.monitor is None:
            self.monitor = app_manager.lookup_service_brick("NetworkMonitor")
            if self.monitor is not None:
                self.logger.info("已关联 NetworkMonitor brick")

    def _sync_monitor_capacity(self, link, active):
        """把场景里定义的链路容量同步给监控模块，便于计算真实利用率。"""
        self._ensure_monitor()
        if self.monitor is None:
            return

        forward = self.directed_link_catalog.get((link.src.dpid, link.dst.dpid))
        reverse = self.directed_link_catalog.get((link.dst.dpid, link.src.dpid))
        if active:
            # 链路恢复时，需要把两个方向上的业务配置容量都告诉监控模块。
            if forward is not None:
                self.monitor.set_configured_port_capacity(
                    link.src.dpid,
                    link.src.port_no,
                    forward["bw_mbps"] * 1_000_000.0,
                )
            if reverse is not None:
                self.monitor.set_configured_port_capacity(
                    link.dst.dpid,
                    link.dst.port_no,
                    reverse["bw_mbps"] * 1_000_000.0,
                )
        else:
            # 链路断开后清理容量，避免监控继续沿用旧链路信息。
            self.monitor.clear_configured_port_capacity(link.src.dpid, link.src.port_no)
            self.monitor.clear_configured_port_capacity(link.dst.dpid, link.dst.port_no)

    def _iter_links(self):
        """遍历当前仍然连通的无向交换机链路，供最短路图构建使用。"""
        seen = set()
        for src, nbrs in self.adjacency.items():
            for dst in nbrs.keys():
                edge = tuple(sorted((src, dst)))
                if edge in seen:
                    continue
                seen.add(edge)
                yield edge

    def _is_inter_switch_port(self, dpid, port_no):
        """判断端口是否属于交换机之间的互联端口。"""
        return port_no in self.inter_switch_ports.get(dpid, set())

    def _learn_host(self, mac, dpid, port_no):
        """
        从 PacketIn 或 HostAdd 事件中学习主机挂载位置。

        这里会做几层过滤：
        - 不是场景中声明的主机则忽略
        - 如果包是从交换机互联端口进来的，说明它不是主机接入口
        - 如果主机挂载交换机与场景真值不一致，也忽略
        """
        spec = self.host_spec_by_mac.get(mac)
        if spec is None:
            return
        if self._is_inter_switch_port(dpid, port_no):
            return
        if dpid != switch_name_to_dpid(spec.switch):
            return

        known_location = self.hosts.get(mac)
        if known_location != (dpid, port_no):
            self.hosts[mac] = (dpid, port_no)
            self.logger.info("Host learn: %s -> %s:%s", mac, dpid, port_no)

    def _bind_expected_hosts_to_access_ports(self, current_hosts):
        """在拓扑信息不完整时，用场景真值把主机绑定到唯一接入口上。"""
        access_ports = defaultdict(list)
        for dpid, port_no in self._get_access_ports():
            access_ports[dpid].append(port_no)

        for dpid, spec in self.host_spec_by_switch.items():
            ports = sorted(access_ports.get(dpid, []))
            if len(ports) != 1:
                continue
            current_hosts.setdefault(spec.mac, (dpid, ports[0]))
        return current_hosts

    def _sync_hosts_from_topology(self):
        """从 Ryu topology API 拉取主机列表，并与场景配置做交叉校验。"""
        try:
            discovered_hosts = get_all_host(self)
        except Exception as exc:
            self.logger.debug("Host sync skipped: %s", exc)
            return

        next_hosts = {}
        for host in discovered_hosts:
            spec = self.host_spec_by_mac.get(host.mac)
            if spec is None:
                continue
            port = getattr(host, "port", None)
            if port is None:
                continue
            if port.dpid != switch_name_to_dpid(spec.switch):
                continue
            next_hosts[host.mac] = (port.dpid, port.port_no)

        next_hosts = self._bind_expected_hosts_to_access_ports(next_hosts)

        removed_hosts = set(self.hosts.keys()) - set(next_hosts.keys())
        self.hosts = next_hosts

        for mac in sorted(removed_hosts):
            self.logger.info("Host forget: %s", mac)

    def _get_access_ports(self):
        """找出非交换机互联端口，也就是主机可能接入的 access 端口。"""
        access_ports = []
        try:
            switches = get_all_switch(self)
        except Exception as exc:
            self.logger.debug("Access-port lookup skipped: %s", exc)
            return access_ports

        for switch in switches:
            dpid = switch.dp.id
            for port in switch.ports:
                if self._is_inter_switch_port(dpid, port.port_no):
                    continue
                access_ports.append((dpid, port.port_no))
        return access_ports

    def _flood_to_access_ports(self, msg, ingress_dpid, ingress_port):
        """仅向 access 端口泛洪，避免广播包在交换机骨干链路里反复扩散。"""
        sent = 0
        for dpid, out_port in self._get_access_ports():
            if dpid == ingress_dpid and out_port == ingress_port:
                continue

            datapath = self.datapaths.get(dpid)
            if datapath is None:
                continue

            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto
            out = parser.OFPPacketOut(
                datapath=datapath,
                buffer_id=ofproto.OFP_NO_BUFFER,
                in_port=ofproto.OFPP_CONTROLLER,
                actions=[parser.OFPActionOutput(out_port)],
                data=msg.data,
            )
            datapath.send_msg(out)
            sent += 1
        return sent

    @set_ev_cls(event.EventHostAdd)
    def _host_add_handler(self, ev):
        """主机出现后学习位置，并在条件满足时补装已知主机对的路径。"""
        host = ev.host
        self._learn_host(host.mac, host.port.dpid, host.port.port_no)
        if len(self.hosts) >= 2 and self.link_weights:
            self._install_known_host_paths(clear_existing=False)

    @set_ev_cls(event.EventHostMove)
    def _host_move_handler(self, ev):
        """主机迁移后重新学习位置，并强制清理旧路径。"""
        host = ev.dst
        self._learn_host(host.mac, host.port.dpid, host.port.port_no)
        if len(self.hosts) >= 2 and self.link_weights:
            self._install_known_host_paths(clear_existing=True)

    def _host_sort_key(self, host_item):
        """按场景中的主机名排序，保证日志和状态构造顺序稳定。"""
        spec = self.host_spec_by_mac.get(host_item[0])
        return spec.name if spec is not None else host_item[0]

    def _known_host_items(self):
        """返回当前已知主机及其位置，顺序稳定。"""
        return sorted(self.hosts.items(), key=self._host_sort_key)

    def _host_pair_count(self):
        """返回当前已知主机能形成的有向主机对数量。"""
        host_count = len(self.hosts)
        return host_count * max(0, host_count - 1)

    def _collect_link_snapshot(self):
        """采集当前每条有向链路的动态指标，形成一帧快照。"""
        self._ensure_monitor()
        snapshot = {}
        for edge in self.edge_order:
            src, dst = edge
            out_port = self.adjacency.get(src, {}).get(dst)
            link_up_flag = 1.0 if out_port is not None else 0.0
            utilization = 0.0
            rate_bps = 0.0
            if self.monitor is not None and out_port is not None:
                # 只有链路当前在线且监控模块已关联时，才读取真实测量值。
                utilization = float(
                    self.monitor.bandwidth_utilization.get(src, {}).get(out_port, 0.0)
                )
                rate_bps = float(self.monitor.port_rate_bps.get(src, {}).get(out_port, 0.0))
                drop_rate = float(self.monitor.port_drop_rate.get(src, {}).get(out_port, 0.0))
            else:
                drop_rate = 0.0

            snapshot[edge] = {
                "out_port": out_port,
                "link_up_flag": link_up_flag,
                "utilization": utilization,
                "rate_bps": rate_bps,
                "drop_rate": drop_rate,
                **self.link_static_features[edge],
            }
        return snapshot

    def _determine_experiment_phase(self, active_link_count):
        """把当前实验阶段划分为 warmup / active / teardown。"""
        if active_link_count >= self.configured_link_count:
            self._topology_ready_once = True

        if (
            self._topology_ready_once
            and len(self.hosts) >= self.expected_host_count
            and active_link_count > 0
        ):
            self._active_phase_seen = True

        if self._active_phase_seen and active_link_count == 0:
            return "teardown"
        if (
            not self._topology_ready_once
            or len(self.hosts) < self.expected_host_count
            or active_link_count == 0
        ):
            return "warmup"
        return "active"

    def _build_state(self, advance_history=False):
        """
        把链路快照编码成 RL 状态向量，并产出奖励计算所需指标。

        这一步是连接“网络监控”和“强化学习”的关键桥梁：
        - 监控模块给出的是面向端口/链路的原始观测
        - 这里把它整理成长度固定、顺序稳定的向量，才能喂给神经网络

        返回值：
        - state: 送入 Agent 的状态向量
        - snapshot: 保留结构化链路信息，方便后续计算权重
        - metrics: 用于奖励、日志和评估的聚合指标
        """
        snapshot = self._collect_link_snapshot()

        state = []
        active_utils = []
        active_rates = []
        active_drop_rates = []
        abs_util_deltas = []

        for edge in self.edge_order:
            payload = snapshot[edge]
            # util_delta 反映链路利用率是在升高还是降低，
            # 这能帮助智能体区分“稳定拥塞”和“正在恶化/正在恢复”的状态。
            previous_util = self._last_observed_utilization.get(edge, payload["utilization"])
            util_delta = payload["utilization"] - previous_util
            payload["util_delta"] = util_delta

            # 每条链路拼 6 个特征：利用率、变化量、是否在线、容量、时延、丢包率。
            state.extend(
                [
                    payload["utilization"],
                    util_delta,
                    payload["link_up_flag"],
                    payload["capacity_norm"],
                    payload["delay_norm"],
                    payload["drop_rate"],
                ]
            )

            abs_util_deltas.append(abs(util_delta))
            if payload["link_up_flag"]:
                active_utils.append(payload["utilization"])
                active_rates.append(payload["rate_bps"])
                active_drop_rates.append(payload["drop_rate"])

        # 再拼接两个全局特征：上一轮动作编号和当前主机对覆盖比例。
        previous_action_norm = (
            float(self.current_action) / float(self.action_size - 1)
            if self.action_size > 1
            else 0.0
        )
        host_pair_ratio = float(self._host_pair_count()) / float(self.max_host_pair_count)
        state.extend([previous_action_norm, host_pair_ratio])

        if advance_history:
            # 只有在主循环里调用时才推进“上一轮观测”，
            # 避免辅助分析函数意外修改历史状态。
            self._last_observed_utilization = {
                edge: snapshot[edge]["utilization"] for edge in self.edge_order
            }

        # metrics 是 reward、日志输出、可视化分析共同使用的汇总指标。
        metrics = {
            "max_utilization": float(max(active_utils or [0.0])),
            "mean_utilization": float(np.mean(active_utils or [0.0])),
            "active_link_count": int(sum(1 for edge in self.edge_order if snapshot[edge]["link_up_flag"])),
            "max_rate_bps": float(max(active_rates or [0.0])),
            "mean_rate_bps": float(np.mean(active_rates or [0.0])),
            "max_drop_rate": float(max(active_drop_rates or [0.0])),
            "mean_drop_rate": float(np.mean(active_drop_rates or [0.0])),
            "util_delta": float(np.mean(abs_util_deltas or [0.0])),
            "max_util_delta": float(max(abs_util_deltas or [0.0])),
            "host_pair_count": self._host_pair_count(),
        }
        metrics["experiment_phase"] = self._determine_experiment_phase(metrics["active_link_count"])
        return state, snapshot, metrics

    def _compute_link_weights(self, snapshot, profile):
        """
        把当前动作对应的 profile 应用于每条链路，得到最短路权重。

        从系统视角看，这一步就是把“RL 选择的动作”翻译成“路由算法真正可用的输入”。
        """
        weights = {}
        for edge, payload in snapshot.items():
            if payload["link_up_flag"] <= 0.0:
                # 已经断开的链路不参与路由图构建。
                continue
            weights[edge] = compute_profile_weight(
                profile,
                payload["utilization"],
                payload["capacity_norm"],
                payload["delay_norm"],
            )
        self.link_weights = weights
        return weights

    def _record_metrics(self, payload):
        """
        把关键指标落盘到 jsonl，便于后处理和可视化。

        这里同时维护：
        - 追加式历史日志：保留完整时间序列
        - latest 文件：方便脚本或人工快速查看最近一步状态
        """
        payload = dict(payload)
        payload.setdefault("timestamp", time.time())
        payload.setdefault("routing_mode", self.routing_mode)
        payload.setdefault("scenario", self.scenario_name)

        try:
            with open(self.metrics_path, "a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            with open(self.latest_metrics_path, "w", encoding="utf-8") as latest_file:
                json.dump(payload, latest_file, ensure_ascii=False, indent=2)
        except Exception as exc:
            self.logger.warning("Failed to write metrics: %s", exc)

    def _clear_route_flows(self):
        """删除控制器托管的路由流表项，为重装路径腾出空间。"""
        cleared = 0
        for datapath in self.datapaths.values():
            self.delete_flows(
                datapath,
                cookie=self.ROUTE_FLOW_COOKIE,
                cookie_mask=0xFFFFFFFFFFFFFFFF,
            )
            cleared += 1
        if cleared:
            self.logger.info("已清理受控路由流表: datapaths=%s", cleared)

    def _update_shortest_path_graph(self):
        """使用最新链路权重重建最短路图。"""
        self.sp.update_topology(list(self._iter_links()), self.link_weights)

    def _build_host_pair_paths(self):
        """
        为所有已知主机对计算当前最短路径。

        因为这里按“有向主机对”逐个计算，所以 h1->h2 和 h2->h1 会被当成两条独立流。
        """
        path_map = {}
        host_items = self._known_host_items()
        if len(host_items) < 2:
            return path_map

        for i in range(len(host_items)):
            for j in range(len(host_items)):
                if i == j:
                    continue

                src_mac, (src_dpid, _src_port) = host_items[i]
                dst_mac, (dst_dpid, _dst_port) = host_items[j]
                path = self.sp.get_shortest_path(src_dpid, dst_dpid)
                if path:
                    path_map[(src_mac, dst_mac)] = path
        return path_map

    def _install_path_map(self, path_map):
        """把 host-pair -> path 的映射批量下发成交换机流表。"""
        for (src_mac, dst_mac), path in path_map.items():
            src_location = self.hosts.get(src_mac)
            dst_location = self.hosts.get(dst_mac)
            if src_location is None or dst_location is None:
                continue

            src_dpid = src_location[0]
            dst_port = dst_location[1]
            datapath = self.datapaths.get(src_dpid)
            if datapath is None:
                continue

            parser = datapath.ofproto_parser
            match = parser.OFPMatch(eth_src=src_mac, eth_dst=dst_mac)
            install_path_flows(
                self,
                path,
                match,
                dst_host_port=dst_port,
                priority=self.ROUTE_FLOW_PRIORITY,
            )

    def _install_known_host_paths(self, clear_existing):
        """
        重新计算并安装全部已知主机对路径，同时返回 churn。

        这是“把决策真正落到网络里”的一步：
        1. 根据新权重重算路径
        2. 和上一轮路径比较，得到 churn
        3. 必要时删除旧规则
        4. 安装新路径对应的流表
        """
        new_paths = self._build_host_pair_paths()
        path_churn = compute_path_churn(self._installed_paths, new_paths)

        if clear_existing or path_churn > 0.0 or (not new_paths and self._installed_paths):
            self._clear_route_flows()

        self._install_path_map(new_paths)
        self._installed_paths = new_paths
        return new_paths, path_churn

    def _average_path_hops(self, path_map=None):
        """计算给定路径集合的平均跳数。"""
        path_map = self._installed_paths if path_map is None else path_map
        if not path_map:
            return 0.0
        return float(np.mean([max(0, len(path) - 1) for path in path_map.values()]))

    def _hop_cost(self, path_map=None):
        """把平均跳数归一化成奖励函数可直接使用的 hop cost。"""
        avg_hops = self._average_path_hops(path_map)
        return avg_hops, float(avg_hops / self.max_reasonable_hops)

    def _path_delay(self, path_map=None):
        """按路径上链路时延求平均路径时延及其归一化值。"""
        path_map = self._installed_paths if path_map is None else path_map
        if not path_map:
            return 0.0, 0.0

        path_delays_ms = []
        for path in path_map.values():
            delay_ms = 0.0
            for src, dst in zip(path, path[1:]):
                delay_ms += float(self.directed_link_catalog.get((src, dst), {}).get("delay_ms", 0.0))
            path_delays_ms.append(delay_ms)

        average_delay_ms = float(np.mean(path_delays_ms or [0.0]))
        normalized_delay = min(1.0, average_delay_ms / self.max_reasonable_path_delay_ms)
        return average_delay_ms, normalized_delay

    def _resolve_agent_phase(self, current_phase):
        """
        决定当前应处于训练、评估还是静态模式。

        这里要区分两个概念：
        - experiment_phase: 拓扑当前是否已经准备完毕，是否处于 teardown
        - agent_phase: 智能体此刻应该训练、评估，还是完全不参与
        """
        if self.routing_mode != "rl" or self.agent is None:
            return "static"
        if current_phase != "active":
            return current_phase
        if self.agent_mode == "eval":
            return "eval"
        if self.eval_interval_active_steps > 0 and self.eval_window_active_steps > 0:
            if self._eval_steps_remaining > 0:
                return "eval"
        return "train"

    def _update_eval_schedule(self, current_phase, agent_phase):
        """
        在 train 模式下按固定窗口插入 greedy 评估阶段。

        这样做的目的是把“训练时带探索的表现”和“纯贪心策略的真实表现”分开观察，
        便于判断模型到底有没有学好。
        """
        if self.routing_mode != "rl" or self.agent is None or self.agent_mode != "train":
            return
        if current_phase != "active":
            if current_phase == "teardown":
                self._eval_steps_remaining = 0
                self._train_active_steps_since_eval = 0
            return
        if agent_phase == "eval":
            if self._eval_steps_remaining > 0:
                self._eval_steps_remaining -= 1
            if self._eval_steps_remaining == 0:
                self._train_active_steps_since_eval = 0
            return

        self._train_active_steps_since_eval += 1
        if (
            self.eval_interval_active_steps > 0
            and self.eval_window_active_steps > 0
            and self._train_active_steps_since_eval >= self.eval_interval_active_steps
        ):
            self._eval_steps_remaining = self.eval_window_active_steps

    def _save_agent_checkpoint(self, reason):
        """按给定原因保存 Agent checkpoint，并记录日志。"""
        if self.agent is None or not self.agent_checkpoint_path:
            return False
        try:
            self.agent.save_checkpoint(self.agent_checkpoint_path)
            self._last_checkpoint_save_step = self.agent.train_steps
            self.logger.info(
                "Saved RL checkpoint: reason=%s path=%s train_steps=%s replay=%s epsilon=%.4f",
                reason,
                self.agent_checkpoint_path,
                self.agent.train_steps,
                self.agent.replay_size,
                self.agent.epsilon,
            )
            return True
        except Exception as exc:
            self.logger.warning("Failed to save RL checkpoint (%s): %s", reason, exc)
            return False

    def _handle_topology_change(self, event_name, link):
        """
        处理拓扑变化后的权重更新、路径重装与指标记录。

        这个函数不直接训练 Agent，它更像一个“外部突发事件处理器”：
        链路一旦上下线，就立刻用当前 profile 重新计算路径，保证网络可用性。
        """
        state, snapshot, state_metrics = self._build_state(advance_history=False)
        self._compute_link_weights(snapshot, self.current_profile)
        self._update_shortest_path_graph()

        path_churn = 0.0
        if len(self.hosts) >= 2:
            _, path_churn = self._install_known_host_paths(clear_existing=True)
        else:
            self._installed_paths = {}

        avg_hops, _hop_cost = self._hop_cost()
        path_delay_ms, path_delay_norm = self._path_delay()
        self._record_metrics(
            {
                "step_type": "topology_change",
                "event": event_name,
                "action": self.current_action,
                "profile_id": self.current_profile.profile_id,
                "profile_name": self.current_profile.name,
                "util_alpha": self.current_profile.util_alpha,
                "capacity_alpha": self.current_profile.capacity_alpha,
                "experiment_phase": state_metrics["experiment_phase"],
                "util_delta": state_metrics["util_delta"],
                "path_churn": path_churn,
                "avg_hops": avg_hops,
                "path_delay_ms": path_delay_ms,
                "path_delay_norm": path_delay_norm,
                "links": len(self.link_weights),
                "configured_link_count": self.configured_link_count,
                "hosts": len(self.hosts),
                "host_pair_count": state_metrics["host_pair_count"],
                "max_utilization": state_metrics["max_utilization"],
                "mean_utilization": state_metrics["mean_utilization"],
                "max_rate_bps": state_metrics["max_rate_bps"],
                "mean_rate_bps": state_metrics["mean_rate_bps"],
                "max_drop_rate": state_metrics["max_drop_rate"],
                "mean_drop_rate": state_metrics["mean_drop_rate"],
                "src_dpid": link.src.dpid,
                "src_port": link.src.port_no,
                "dst_dpid": link.dst.dpid,
                "dst_port": link.dst.port_no,
                "agent_phase": self._resolve_agent_phase(state_metrics["experiment_phase"]),
                "agent_mode": self.agent_mode,
                "train_steps": self.agent.train_steps if self.agent is not None else 0,
                "replay_size": self.agent.replay_size if self.agent is not None else 0,
            }
        )
        self.logger.info(
            "Topology change handled: %s %s:%s -> %s:%s, profile=%s, phase=%s, paths=%s",
            event_name,
            link.src.dpid,
            link.src.port_no,
            link.dst.dpid,
            link.dst.port_no,
            self.current_profile.name,
            state_metrics["experiment_phase"],
            len(self._installed_paths),
        )

    def _rl_loop(self):
        """
        控制器核心循环：观测状态、计算奖励、训练并执行下一步动作。

        一个完整循环可以理解成：
        1. 观测当前网络，构造 state
        2. 如果上一步已经有动作，就先评价“上一步动作的效果”并生成 reward
        3. 在训练模式下把上一条经验写入经验池，并执行一次 replay
        4. 根据当前 state 选择下一步动作
        5. 用该动作更新链路权重与流表
        6. 保存这轮信息，供下一轮继续形成 transition
        """
        while True:
            try:
                state, snapshot, state_metrics = self._build_state(advance_history=True)
                current_phase = state_metrics["experiment_phase"]
                agent_phase = self._resolve_agent_phase(current_phase)
                observed_avg_hops, observed_hop_cost = self._hop_cost()
                observed_path_delay_ms, observed_path_delay_norm = self._path_delay()
                state_metrics["path_delay_ms"] = observed_path_delay_ms
                state_metrics["path_delay_norm"] = observed_path_delay_norm
                reward = None
                loss = None
                checkpoint_saved = False

                if self.routing_mode == "rl" and self.agent is not None:
                    # 只有“上一轮”和“当前轮”都处在 active 阶段时，
                    # 奖励才有可比性；否则 warmup/teardown 的噪声太大。
                    if (
                        self._last_state is not None
                        and self._last_action is not None
                        and self._last_state_metrics is not None
                        and self._last_experiment_phase == "active"
                        and current_phase == "active"
                    ):
                        # 当前这一步的 reward 来自“上一轮动作”导致的网络状态变化。
                        reward = compute_reward(
                            self._last_state_metrics,
                            state_metrics,
                            observed_hop_cost,
                            self._last_path_churn,
                        )
                        if self._last_agent_phase == "train" and self.agent_mode == "train":
                            # 把上一轮 transition 放入经验池，并在训练窗口内执行一次回放。
                            self.agent.remember(
                                self._last_state,
                                self._last_action,
                                reward,
                                state,
                                False,
                            )
                            if agent_phase == "train":
                                # replay 返回 None 代表样本还不够，属于正常情况。
                                loss = self.agent.replay()
                                if loss is not None:
                                    self.logger.info(
                                        (
                                            "RL step: reward=%.4f max_util=%.4f mean_util=%.4f "
                                            "mean_drop=%.4f path_delay_ms=%.2f hop_cost=%.4f "
                                            "churn=%.4f epsilon=%.4f replay=%s loss=%.6f"
                                        ),
                                        reward,
                                        state_metrics["max_utilization"],
                                        state_metrics["mean_utilization"],
                                        state_metrics["mean_drop_rate"],
                                        state_metrics["path_delay_ms"],
                                        observed_hop_cost,
                                        self._last_path_churn,
                                        self.agent.epsilon,
                                        self.agent.replay_size,
                                        loss,
                                    )
                                    if (
                                        self.agent.train_steps - self._last_checkpoint_save_step
                                        >= self.checkpoint_save_interval
                                    ):
                                        checkpoint_saved = self._save_agent_checkpoint("interval")

                    # 动作保持机制：避免控制器每 5 秒都改一次路，导致网络过度抖动。
                    if (
                        current_phase == "active"
                        and self.decision_hold_steps > 1
                        and self._hold_steps_remaining > 0
                    ):
                        # 为了减少路由抖动，同一个动作可以强制维持若干个控制周期。
                        action = self.current_action
                        self._hold_steps_remaining -= 1
                    else:
                        action = int(self.agent.act(state, greedy=(agent_phase == "eval")))
                        self._hold_steps_remaining = (
                            self.decision_hold_steps - 1 if current_phase == "active" else 0
                        )
                    # 评估阶段虽然仍调用 act()，但 epsilon 被强制视作 0。
                    epsilon = 0.0 if agent_phase == "eval" else self.agent.epsilon
                else:
                    # static 模式下固定使用动作 0，相当于传统基线路由。
                    action = 0
                    epsilon = None

                # 再做一次边界保护，确保动作编号始终落在合法动作空间内。
                action = max(0, min(action, self.action_size - 1))
                profile = self.ACTION_PROFILES[action]

                # 当前动作/路径/指标会成为“下一轮计算 reward 的上一轮基准”。
                self.current_action = action
                self.current_profile = profile

                # 把动作映射成链路权重，再更新图并安装新的最短路。
                self._compute_link_weights(snapshot, profile)
                self._update_shortest_path_graph()
                new_paths, path_churn = self._install_known_host_paths(clear_existing=False)
                installed_avg_hops, _installed_hop_cost = self._hop_cost(new_paths)
                installed_path_delay_ms, installed_path_delay_norm = self._path_delay(new_paths)

                # 保存这一步的上下文，等下一轮观测到 next_state 后再组成完整 transition。
                self._last_state = state
                self._last_action = action
                self._last_state_metrics = state_metrics
                self._last_experiment_phase = current_phase
                self._last_agent_phase = agent_phase
                self._last_path_churn = path_churn

                if (
                    self.agent is not None
                    and self.agent_mode == "train"
                    and current_phase == "teardown"
                    and self._last_checkpoint_save_step < self.agent.train_steps
                ):
                    checkpoint_saved = self._save_agent_checkpoint("teardown") or checkpoint_saved

                self._record_metrics(
                    {
                        "step_type": "routing_decision",
                        "action": action,
                        "profile_id": profile.profile_id,
                        "profile_name": profile.name,
                        "util_alpha": profile.util_alpha,
                        "capacity_alpha": profile.capacity_alpha,
                        "reward": reward,
                        "loss": loss,
                        "epsilon": epsilon,
                        "agent_phase": agent_phase,
                        "agent_mode": self.agent_mode,
                        "experiment_phase": current_phase,
                        "max_utilization": state_metrics["max_utilization"],
                        "mean_utilization": state_metrics["mean_utilization"],
                        "max_rate_bps": state_metrics["max_rate_bps"],
                        "mean_rate_bps": state_metrics["mean_rate_bps"],
                        "max_drop_rate": state_metrics["max_drop_rate"],
                        "mean_drop_rate": state_metrics["mean_drop_rate"],
                        "util_delta": state_metrics["util_delta"],
                        "max_util_delta": state_metrics["max_util_delta"],
                        "avg_hops": installed_avg_hops,
                        "observed_avg_hops": observed_avg_hops,
                        "path_delay_ms": installed_path_delay_ms,
                        "path_delay_norm": installed_path_delay_norm,
                        "observed_path_delay_ms": observed_path_delay_ms,
                        "observed_path_delay_norm": observed_path_delay_norm,
                        "previous_hop_cost": observed_hop_cost,
                        "path_churn": path_churn,
                        "links": len(self.link_weights),
                        "configured_link_count": self.configured_link_count,
                        "hosts": len(self.hosts),
                        "host_pair_count": state_metrics["host_pair_count"],
                        "active_link_count": state_metrics["active_link_count"],
                        "train_steps": self.agent.train_steps if self.agent is not None else 0,
                        "replay_size": self.agent.replay_size if self.agent is not None else 0,
                        "checkpoint_saved": checkpoint_saved,
                    }
                )

                self.logger.info(
                    (
                        "%s decision: action=%s profile=%s phase=%s agent_phase=%s "
                        "max_util=%.4f mean_util=%.4f mean_drop=%.4f "
                        "path_delay_ms=%.2f util_delta=%.4f path_churn=%.4f "
                        "links=%s hosts=%s replay=%s train_steps=%s"
                    ),
                    self.routing_mode.upper(),
                    action,
                    profile.name,
                    current_phase,
                    agent_phase,
                    state_metrics["max_utilization"],
                    state_metrics["mean_utilization"],
                    state_metrics["mean_drop_rate"],
                    installed_path_delay_ms,
                    state_metrics["util_delta"],
                    path_churn,
                    len(self.link_weights),
                    len(self.hosts),
                    self.agent.replay_size if self.agent is not None else 0,
                    self.agent.train_steps if self.agent is not None else 0,
                )
                self._update_eval_schedule(current_phase, agent_phase)
            except Exception as exc:
                self.logger.exception("RL loop error: %s", exc)

            hub.sleep(5)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """处理未知流量：学习主机位置，并为首次通信安装路径。"""
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug(
                "数据包被截断: 仅收到 %s 字节 (总共 %s 字节)",
                ev.msg.msg_len,
                ev.msg.total_len,
            )

        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match["in_port"]

        # PacketIn 是控制器第一次看到某个流时的入口，
        # 这里既要学习主机位置，也要尽快给这个流装上对应路径。
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src
        dpid = datapath.id

        if src in self.host_spec_by_mac:
            self._learn_host(src, dpid, in_port)

        if dst not in self.hosts:
            # 目的主机未知时，只向 access 端口泛洪，等待后续学习。
            flood_count = self._flood_to_access_ports(msg, dpid, in_port)
            if flood_count == 0:
                actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
                out = parser.OFPPacketOut(
                    datapath=datapath,
                    buffer_id=msg.buffer_id,
                    in_port=in_port,
                    actions=actions,
                    data=(None if msg.buffer_id != ofproto.OFP_NO_BUFFER else msg.data),
                )
                datapath.send_msg(out)
            return

        dst_dpid, dst_port = self.hosts[dst]
        if not self.adjacency:
            actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
            out = parser.OFPPacketOut(
                datapath=datapath,
                buffer_id=msg.buffer_id,
                in_port=in_port,
                actions=actions,
                data=(None if msg.buffer_id != ofproto.OFP_NO_BUFFER else msg.data),
            )
            datapath.send_msg(out)
            return

        # shortest_path 的输入是“交换机到交换机”，最后一跳到主机端口单独处理。
        path = self.sp.get_shortest_path(dpid, dst_dpid)
        if not path:
            flood_count = self._flood_to_access_ports(msg, dpid, in_port)
            if flood_count == 0:
                actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
                out = parser.OFPPacketOut(
                    datapath=datapath,
                    buffer_id=msg.buffer_id,
                    in_port=in_port,
                    actions=actions,
                    data=(None if msg.buffer_id != ofproto.OFP_NO_BUFFER else msg.data),
                )
                datapath.send_msg(out)
            return

        match = parser.OFPMatch(eth_src=src, eth_dst=dst)
        install_path_flows(
            self,
            path,
            match,
            dst_host_port=dst_port,
            priority=self.ROUTE_FLOW_PRIORITY,
        )

        if len(path) == 1:
            out_port = dst_port
        else:
            out_port = self.adjacency.get(dpid, {}).get(path[1], ofproto.OFPP_FLOOD)

        actions = [parser.OFPActionOutput(out_port)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=(None if msg.buffer_id != ofproto.OFP_NO_BUFFER else msg.data),
        )
        datapath.send_msg(out)
