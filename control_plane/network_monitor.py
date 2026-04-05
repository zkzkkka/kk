"""
网络监控层。

负责周期性向交换机拉取端口统计信息，并把原始字节计数转换成
控制器可直接消费的利用率、速率和丢包率指标。
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.lib import hub
from collections import defaultdict
import time


class NetworkMonitor(app_manager.RyuApp):
    """
    数据感知层 (Data Collection Layer):
    监控网络状态，包括网络拓扑发现和链路带宽利用率。
    这将为强化学习(RL)智能体构建状态(State)表示。
    """
    def __init__(self, *args, **kwargs):
        super(NetworkMonitor, self).__init__(*args, **kwargs)
        self.datapaths = {}
        # port_stats 保存上一轮原始计数器值，便于下一轮用“差分”计算速率。
        self.port_stats = {}
        self.port_speed = {}  # 映射 (dpid, 端口号) -> 端口速率 (bps)
        self.configured_port_capacity = {}  # 映射 (dpid, 端口号) -> 业务配置的链路容量 (bps)
        self.flow_stats = {}
        self.stats_thread = hub.spawn(self._monitor)

        # 链路信息和带宽利用率
        self.link_info = {}  # 映射 (源dpid, 源端口) -> (目的dpid, 目的端口)
        self.bandwidth_utilization = defaultdict(lambda: defaultdict(float))  # 映射 dpid -> 端口号 -> 利用率
        self.port_rate_bps = defaultdict(lambda: defaultdict(float))  # 映射 dpid -> 端口号 -> 当前速率
        self.port_drop_rate = defaultdict(lambda: defaultdict(float))  # 映射 dpid -> 端口号 -> 区间丢包率

    def set_configured_port_capacity(self, dpid, port_no, capacity_bps):
        """由控制器同步“业务上配置的链路容量”，优先于交换机自报速率。"""
        self.configured_port_capacity[(dpid, port_no)] = float(capacity_bps)

    def clear_configured_port_capacity(self, dpid, port_no):
        """链路失效后清除对应端口容量，避免继续沿用旧值。"""
        self.configured_port_capacity.pop((dpid, port_no), None)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('注册数据面(datapath): %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('注销数据面(datapath): %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        """
        定期向所有常规交换机请求各类统计信息。

        这里相当于整个系统的“采样器”：
        控制器并不会被动得到链路利用率，而是每隔固定时间主动去问交换机。
        """
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(5)  # 轮询间隔 (秒)

    def _request_stats(self, datapath):
        """向单台交换机请求端口统计和端口描述。"""
        self.logger.debug('发送统计请求: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # 请求端口统计信息 (Port Stats)
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

        # 请求端口描述信息，用于获取端口容量并把速率转换为利用率。
        desc_req = parser.OFPPortDescStatsRequest(datapath, 0)
        datapath.send_msg(desc_req)

        # 请求流表统计信息 (如果需要特定数据流的指标，可启用此项)
        # req = parser.OFPFlowStatsRequest(datapath)
        # datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, MAIN_DISPATCHER)
    def _port_desc_stats_reply_handler(self, ev):
        """处理端口描述回复，提取交换机上报的端口容量信息。"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto

        for stat in ev.msg.body:
            port_no = stat.port_no
            if port_no == ofproto.OFPP_LOCAL:
                continue

            # OpenFlow 1.3 的 curr_speed/max_speed 单位是 kbps，这里统一换算成 bps。
            speed_kbps = max(getattr(stat, 'curr_speed', 0), getattr(stat, 'max_speed', 0))
            if speed_kbps > 0:
                self.port_speed[(datapath.id, port_no)] = float(speed_kbps) * 1000.0

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        """
        处理端口统计信息包，以此计算网络带宽利用率。

        交换机上报的是累计计数器，例如“到目前为止收了多少字节”。
        所以这里不能直接把 `rx_bytes` 当作速率，而要用：
        “本轮累计值 - 上轮累计值” / 时间差
        才能得到这个轮询区间内的真实速率。
        """
        body = ev.msg.body
        datapath = ev.msg.datapath
        dpid = datapath.id

        now = time.time()

        for stat in body:
            port_no = stat.port_no
            if port_no == datapath.ofproto.OFPP_LOCAL:
                continue

            key = (dpid, port_no)

            # 只有拿到上一轮统计值时，才能计算这个轮询区间内的速率。
            if key in self.port_stats:
                prev_stat, prev_time = self.port_stats[key]
                time_diff = now - prev_time
                if time_diff > 0:
                    rx_bytes = stat.rx_bytes - prev_stat.rx_bytes
                    tx_bytes = stat.tx_bytes - prev_stat.tx_bytes
                    rx_packets = max(0, stat.rx_packets - prev_stat.rx_packets)
                    tx_packets = max(0, stat.tx_packets - prev_stat.tx_packets)
                    rx_dropped = max(0, stat.rx_dropped - prev_stat.rx_dropped)
                    tx_dropped = max(0, stat.tx_dropped - prev_stat.tx_dropped)

                    # 先把累计字节差分转换为 bps 速率。
                    rx_rate = (rx_bytes * 8) / time_diff
                    tx_rate = (tx_bytes * 8) / time_diff
                    current_rate_bps = max(rx_rate, tx_rate)
                    self.port_rate_bps[dpid][port_no] = current_rate_bps

                    # 存储收发两端方向的最大值作为端口的实际利用率。
                    # 如果已经拿到端口容量，则将其转换为 [0, 1] 区间的真实利用率。
                    port_capacity_bps = (
                        self.configured_port_capacity.get(key)
                        or self.port_speed.get(key)
                    )
                    if port_capacity_bps and port_capacity_bps > 0:
                        self.bandwidth_utilization[dpid][port_no] = min(
                            current_rate_bps / port_capacity_bps,
                            1.0,
                        )
                    else:
                        self.bandwidth_utilization[dpid][port_no] = 0.0

                    # 丢包率也按“区间统计”计算，而不是看累计总丢包数。
                    rx_observed_packets = rx_packets + rx_dropped
                    tx_observed_packets = tx_packets + tx_dropped
                    drop_candidates = []
                    if rx_observed_packets > 0:
                        drop_candidates.append(rx_dropped / float(rx_observed_packets))
                    if tx_observed_packets > 0:
                        drop_candidates.append(tx_dropped / float(tx_observed_packets))
                    self.port_drop_rate[dpid][port_no] = max(drop_candidates, default=0.0)

            # 无论是否能算出速率，都要更新“上一轮样本”，供下次轮询使用。
            self.port_stats[key] = (stat, now)

    def get_network_state(self):
        """
        把收集到的网络度量指标抽象为一个状态向量(State vector)供强化学习模型使用。
        返回代表当前网络负载情况的字典或数组。

        当前项目里控制器已经自己构造了更丰富的状态向量，这个方法更像一个
        早期或简化版接口，保留它有利于独立调试监控模块。
        """
        # 在真实的实现中，这里应聚合所有拓扑链路的利用率数据
        state = []
        for dpid in sorted(self.bandwidth_utilization.keys()):
            for port in sorted(self.bandwidth_utilization[dpid].keys()):
                state.append(self.bandwidth_utilization[dpid][port])
        return state
