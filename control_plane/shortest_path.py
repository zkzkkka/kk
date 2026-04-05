"""
路由计算层 (Shortest Path Calculation Layer)
实现 Dijkstra 算法以计算最短路径，
其链路权重由强化学习(RL)智能体动态提供。
"""
import heapq


class ShortestPathCalculator:
    def __init__(self):
        self.topology = {}  # 图结构: {节点1: {节点2: 权重, 节点3: 权重}}

    def update_topology(self, links, weights):
        """
        更新内部图结构，支持按方向维护权重。

        这里的 links 表示“物理上哪些交换机之间是连通的”，
        而 weights 表示“当前策略下这些链路的代价是多少”。
        """
        self.topology = {}

        def _ensure_node(node):
            if node not in self.topology:
                self.topology[node] = {}

        # 先根据拓扑把双向节点关系补齐，再叠加当前时刻的方向性权重。
        # 如果某个方向没有显式权重，就尽量复用反向权重或默认值 1.0。
        for (src, dst) in links or []:
            _ensure_node(src)
            _ensure_node(dst)

            forward_weight = weights.get((src, dst))
            reverse_weight = weights.get((dst, src))

            if forward_weight is None:
                forward_weight = reverse_weight if reverse_weight is not None else 1.0
            if reverse_weight is None:
                reverse_weight = forward_weight if forward_weight is not None else 1.0

            self.topology[src][dst] = forward_weight
            self.topology[dst][src] = reverse_weight

        for (src, dst), weight in (weights or {}).items():
            _ensure_node(src)
            _ensure_node(dst)
            self.topology[src][dst] = weight

    def get_shortest_path(self, src, dst):
        """
        标准的 Dijkstra 最短路径算法。
        返回代表路径的节点列表: [源节点, 节点1, ..., 目的节点]

        这里求的是“总代价最小”的路径，而不是单纯 hop 数最少。
        因此 RL 只要改变链路权重，就能间接影响最终选路结果。
        """
        if src not in self.topology or dst not in self.topology:
            return None

        distances = {node: float('inf') for node in self.topology}
        distances[src] = 0
        previous_nodes = {node: None for node in self.topology}

        # 优先队列里保存“当前已知最短距离, 节点”。
        pq = [(0, src)]

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            if current_distance > distances[current_node]:
                continue

            if current_node == dst:
                break

            for neighbor, weight in self.topology[current_node].items():
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))

        # 根据 previous_nodes 从终点反向回溯整条路径。
        path = []
        current = dst
        while current is not None:
            path.append(current)
            current = previous_nodes[current]

        path.reverse()
        if path[0] == src:
            return path
        return None
