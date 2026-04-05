"""Agent 训练与 checkpoint 相关单元测试。"""

import os
import sys
import tempfile
import unittest

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from agent.train import Agent, AgentConfig  # noqa: E402


def _transition(value):
    """构造一条最小可训练样本，便于重复喂给 Agent。"""
    state = [float(value), 0.1, 0.2, 0.3]
    next_state = [float(value) + 0.05, 0.2, 0.1, 0.4]
    return state, 1, 0.25, next_state, False


class AgentTests(unittest.TestCase):
    def test_replay_starts_once_min_replay_size_is_reached(self):
        """经验池样本达到最小门槛后，应能开始训练并衰减 epsilon。"""
        config = AgentConfig(
            memory_size=32,
            batch_size=8,
            min_replay_size=2,
            epsilon_start=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.9,
            persist_replay_buffer=False,
        )
        agent = Agent(state_size=4, action_size=3, config=config)

        # 连续喂两条最小样本，刚好达到 min_replay_size=2 的门槛。
        first = _transition(0.1)
        second = _transition(0.4)
        agent.remember(*first)
        agent.remember(*second)

        loss = agent.replay()

        self.assertIsNotNone(loss)
        self.assertGreater(agent.train_steps, 0)
        self.assertLess(agent.epsilon, config.epsilon_start)

    def test_checkpoint_round_trip_restores_training_state(self):
        """保存再加载 checkpoint 后，训练状态应尽量完整恢复。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "agent_checkpoint.pt")
            config = AgentConfig(
                memory_size=32,
                batch_size=4,
                min_replay_size=2,
                checkpoint_path=checkpoint_path,
                persist_replay_buffer=True,
            )
            agent = Agent(state_size=4, action_size=3, config=config)
            # 先做一小步训练，再保存，这样 checkpoint 里会同时包含模型参数和训练状态。
            agent.remember(*_transition(0.1))
            agent.remember(*_transition(0.3))
            agent.replay()
            saved_weight = next(agent.model.parameters()).detach().clone()
            saved_epsilon = agent.epsilon
            saved_train_steps = agent.train_steps
            saved_memory_size = agent.replay_size

            agent.save_checkpoint()

            restored_agent = Agent(state_size=4, action_size=3, config=config)
            restored = restored_agent.load_checkpoint()

            self.assertTrue(restored)
            self.assertEqual(restored_agent.train_steps, saved_train_steps)
            self.assertEqual(restored_agent.replay_size, saved_memory_size)
            self.assertAlmostEqual(restored_agent.epsilon, saved_epsilon)
            self.assertTrue(
                torch.allclose(next(restored_agent.model.parameters()), saved_weight)
            )

    def test_checkpoint_load_compat_when_state_size_expands(self):
        """状态维度扩展后，旧 checkpoint 仍应能兼容加载。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "agent_checkpoint.pt")
            old_config = AgentConfig(
                memory_size=32,
                batch_size=4,
                min_replay_size=2,
                checkpoint_path=checkpoint_path,
                persist_replay_buffer=True,
            )
            old_agent = Agent(state_size=4, action_size=3, config=old_config)
            old_agent.remember(*_transition(0.1))
            old_agent.remember(*_transition(0.3))
            old_agent.replay()
            old_fc1_weight = old_agent.model.fc1.weight.detach().clone()
            old_epsilon = old_agent.epsilon
            old_train_steps = old_agent.train_steps
            old_agent.save_checkpoint()

            new_config = AgentConfig(
                memory_size=32,
                batch_size=4,
                min_replay_size=2,
                checkpoint_path=checkpoint_path,
                persist_replay_buffer=True,
            )
            # 新模型多了 2 个状态维度，用来模拟“项目后期给状态向量新增特征”的场景。
            expanded_agent = Agent(state_size=6, action_size=3, config=new_config)
            restored = expanded_agent.load_checkpoint()

            self.assertTrue(restored)
            self.assertEqual(expanded_agent.train_steps, old_train_steps)
            self.assertAlmostEqual(expanded_agent.epsilon, old_epsilon)
            self.assertEqual(expanded_agent.replay_size, 0)
            self.assertTrue(
                torch.allclose(
                    expanded_agent.model.fc1.weight[:, :4],
                    old_fc1_weight,
                )
            )
            self.assertTrue(
                torch.allclose(
                    expanded_agent.model.fc1.weight[:, 4:],
                    torch.zeros_like(expanded_agent.model.fc1.weight[:, 4:]),
                )
            )


if __name__ == "__main__":
    unittest.main()
