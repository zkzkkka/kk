"""
转发执行层 (Forwarding Execution Layer)
负责解析最短路径列表，并生成相应的 OpenFlow 规则，由 Ryu 控制器推送到交换机。

约定：
- ryu_app 需要提供：
  - ryu_app.datapaths: {dpid: datapath}
  - ryu_app.adjacency: adjacency[src_dpid][dst_dpid] = out_port
  - ryu_app.hosts: {host_mac: (dpid, port_no)}
  - ryu_app.add_flow(datapath, priority, match, actions)
"""


def install_path_flows(ryu_app, path, match, dst_host_port=None, priority=10):
    """沿着 path(交换机 dpid 列表) 逐跳下发流表。

    - 对中间交换机：out_port = adjacency[current][next]
    - 对最后一跳交换机：如果 dst_host_port 提供，则 out_port=dst_host_port

    注意：match 由调用方构造（常见为 eth_src/eth_dst）。
    """
    if not path or len(path) < 1:
        return

    # 特殊情况：源主机和目的主机挂在同一台交换机上，路径只包含一个 dpid。
    if len(path) == 1:
        dpid = path[0]
        if dst_host_port is None:
            return
        datapath = ryu_app.datapaths.get(dpid)
        if datapath is None:
            return
        parser = datapath.ofproto_parser
        actions = [parser.OFPActionOutput(dst_host_port)]
        ryu_app.add_flow(datapath, priority, match, actions)
        return

    # 路径中每个交换机都安装一条相同 match 的转发表，只是输出端口不同。
    # 换句话说，路径是通过“逐跳指定 out_port”来落地的。
    for i, current_dpid in enumerate(path):
        next_dpid = path[i + 1] if i < len(path) - 1 else None

        datapath = ryu_app.datapaths.get(current_dpid)
        if datapath is None:
            continue

        parser = datapath.ofproto_parser

        # 真正的最后一台交换机负责把报文送到目的主机；
        # 前面的交换机则只需要知道下一跳交换机该走哪个端口。
        if i == len(path) - 1:
            if dst_host_port is None:
                continue
            out_port = dst_host_port
        else:
            out_port = get_out_port(ryu_app, current_dpid, next_dpid)

        if out_port is None:
            continue

        actions = [parser.OFPActionOutput(out_port)]
        ryu_app.add_flow(datapath, priority, match, actions)


def get_out_port(ryu_app, current_dpid, next_dpid):
    """查询从 current_dpid 转发到 next_dpid 的输出端口。"""
    return ryu_app.adjacency.get(current_dpid, {}).get(next_dpid)
