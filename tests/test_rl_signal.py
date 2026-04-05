"""奖励函数、动作空间与拓扑工具相关单元测试。"""

import os
import sys
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from control_plane.rl_signal import (  # noqa: E402
    RewardConfig,
    ROUTING_PROFILES,
    compute_path_churn,
    compute_reward,
)
from control_plane.shortest_path import ShortestPathCalculator  # noqa: E402
from topology.topology_catalog import (  # noqa: E402
    BASE_SWITCH_LINKS,
    build_directed_link_catalog,
    expected_host_pair_count,
    get_scenario,
)


class RlSignalTests(unittest.TestCase):
    def test_routing_profiles_cover_nine_action_profiles(self):
        """动作空间应覆盖 3x3 共 9 种 profile。"""
        self.assertEqual(len(ROUTING_PROFILES), 9)
        self.assertEqual(ROUTING_PROFILES[0].profile_id, 0)
        self.assertEqual(ROUTING_PROFILES[-1].profile_id, 8)

    def test_reward_increases_when_congestion_drops(self):
        """当拥塞显著下降时，奖励应优于拥塞恶化场景。"""
        previous_metrics = {
            "max_utilization": 0.7,
            "mean_utilization": 0.5,
            "util_delta": 0.2,
        }
        improved_metrics = {
            "max_utilization": 0.3,
            "mean_utilization": 0.2,
            "util_delta": 0.05,
        }
        worse_metrics = {
            "max_utilization": 0.8,
            "mean_utilization": 0.6,
            "util_delta": 0.3,
        }
        reward_config = RewardConfig()

        # improved_metrics 表示链路拥塞整体缓解，应拿到更好的 reward。
        improved_reward = compute_reward(
            previous_metrics,
            improved_metrics,
            0.3,
            0.1,
            reward_config=reward_config,
        )
        worse_reward = compute_reward(
            previous_metrics,
            worse_metrics,
            0.3,
            0.1,
            reward_config=reward_config,
        )

        self.assertGreater(improved_reward, worse_reward)
        self.assertGreater(improved_reward, -0.1)

    def test_reward_penalizes_overloaded_peak_links(self):
        """峰值链路过载越严重，奖励应越低。"""
        previous_metrics = {
            "max_utilization": 0.82,
            "mean_utilization": 0.3,
            "util_delta": 0.08,
        }
        safer_metrics = {
            "max_utilization": 0.78,
            "mean_utilization": 0.28,
            "util_delta": 0.04,
        }
        overloaded_metrics = {
            "max_utilization": 0.96,
            "mean_utilization": 0.28,
            "util_delta": 0.04,
        }
        reward_config = RewardConfig(
            overload_threshold=0.8,
            overload_penalty_weight=1.0,
        )

        safer_reward = compute_reward(
            previous_metrics,
            safer_metrics,
            0.2,
            0.0,
            reward_config=reward_config,
        )
        overloaded_reward = compute_reward(
            previous_metrics,
            overloaded_metrics,
            0.2,
            0.0,
            reward_config=reward_config,
        )

        self.assertGreater(safer_reward, overloaded_reward)

    def test_tuned_reward_prefers_reroute_over_staying_overloaded(self):
        """不同权重设置会改变“重路由”和“维持现状”之间的偏好。"""
        previous_metrics = {
            "max_utilization": 0.92,
            "mean_utilization": 0.30,
            "util_delta": 0.04,
        }
        rerouted_metrics = {
            "max_utilization": 0.84,
            "mean_utilization": 0.31,
            "util_delta": 0.06,
        }
        overloaded_metrics = {
            "max_utilization": 0.94,
            "mean_utilization": 0.29,
            "util_delta": 0.03,
        }
        conservative_config = RewardConfig(
            overload_threshold=0.8,
            overload_penalty_weight=0.5,
            churn_penalty_weight=0.15,
        )
        aggressive_config = RewardConfig()

        # conservative_config 更怕 churn，因此即使重路由降低了过载，也未必更优。
        conservative_rerouted = compute_reward(
            previous_metrics,
            rerouted_metrics,
            0.4,
            0.5,
            reward_config=conservative_config,
        )
        conservative_overloaded = compute_reward(
            previous_metrics,
            overloaded_metrics,
            0.1,
            0.0,
            reward_config=conservative_config,
        )
        aggressive_rerouted = compute_reward(
            previous_metrics,
            rerouted_metrics,
            0.4,
            0.5,
            reward_config=aggressive_config,
        )
        aggressive_overloaded = compute_reward(
            previous_metrics,
            overloaded_metrics,
            0.1,
            0.0,
            reward_config=aggressive_config,
        )

        self.assertLess(conservative_rerouted, conservative_overloaded)
        self.assertGreater(aggressive_rerouted, aggressive_overloaded)

    def test_reward_penalizes_drop_rate_and_path_delay(self):
        """在其他项不变时，丢包和时延恶化应拉低奖励。"""
        previous_metrics = {
            "max_utilization": 0.5,
            "mean_utilization": 0.3,
            "util_delta": 0.08,
        }
        cleaner_metrics = {
            "max_utilization": 0.5,
            "mean_utilization": 0.3,
            "util_delta": 0.08,
            "mean_drop_rate": 0.0,
            "path_delay_norm": 0.2,
        }
        degraded_metrics = {
            "max_utilization": 0.5,
            "mean_utilization": 0.3,
            "util_delta": 0.08,
            "mean_drop_rate": 0.2,
            "path_delay_norm": 0.7,
        }
        reward_config = RewardConfig(
            max_util_gain_weight=0.0,
            mean_util_gain_weight=0.0,
            util_delta_gain_weight=0.0,
            overload_penalty_weight=0.0,
            hop_penalty_weight=0.0,
            churn_penalty_weight=0.0,
            drop_rate_penalty_weight=0.3,
            path_delay_penalty_weight=0.2,
        )

        cleaner_reward = compute_reward(
            previous_metrics,
            cleaner_metrics,
            0.0,
            0.0,
            reward_config=reward_config,
        )
        degraded_reward = compute_reward(
            previous_metrics,
            degraded_metrics,
            0.0,
            0.0,
            reward_config=reward_config,
        )

        self.assertGreater(cleaner_reward, degraded_reward)

    def test_path_churn_detects_fraction_of_changed_paths(self):
        """路径变动比例应能正确反映 churn 大小。"""
        previous_paths = {
            ("a", "b"): [1, 2, 4],
            ("c", "d"): [3, 4, 6],
        }
        next_paths = {
            ("a", "b"): [1, 3, 4],
            ("c", "d"): [3, 4, 6],
        }

        self.assertAlmostEqual(compute_path_churn(previous_paths, next_paths), 0.5)
        self.assertEqual(compute_path_churn(previous_paths, previous_paths), 0.0)

    def test_directed_shortest_path_prefers_lower_forward_weight(self):
        """有向图场景下，最短路应遵循方向相关的链路权重。"""
        calculator = ShortestPathCalculator()
        links = [(1, 2), (2, 3), (1, 3)]
        weights = {
            (1, 2): 1.0,
            (2, 1): 5.0,
            (2, 3): 1.0,
            (3, 2): 1.0,
            (1, 3): 4.0,
            (3, 1): 1.0,
        }
        calculator.update_topology(links, weights)

        # 1->3 方向上，经由 2 的总代价更小，因此应选择 [1, 2, 3]。
        self.assertEqual(calculator.get_shortest_path(1, 3), [1, 2, 3])
        self.assertEqual(calculator.get_shortest_path(3, 1), [3, 1])

    def test_topology_catalog_exposes_expected_sizes(self):
        """场景目录应暴露稳定的链路数量和主机对数量。"""
        directed_links = build_directed_link_catalog(BASE_SWITCH_LINKS)
        legacy_name, legacy = get_scenario("legacy")
        multiflow_name, multiflow = get_scenario("multiflow")
        complexflow_name, complexflow = get_scenario("complexflow")

        self.assertEqual(legacy_name, "legacy")
        self.assertEqual(multiflow_name, "multiflow")
        self.assertEqual(complexflow_name, "complexflow")
        self.assertEqual(len(directed_links), 18)
        self.assertEqual(expected_host_pair_count(legacy), 2)
        self.assertEqual(expected_host_pair_count(multiflow), 12)
        self.assertEqual(expected_host_pair_count(complexflow), 30)


if __name__ == "__main__":
    unittest.main()
