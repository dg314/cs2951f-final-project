import numpy as np
from snake import SnakeGame, SnakeAgent, ParameterizedSnakeAgent, GreedySnakeAgent, compare_agents
from policy_network import PolicyNetwork, train_policy_network
import copy
import random

# Maximum Margin Inverse Reinforcement Learning
# (Algorithms for Decision Making 18.4)
class MaximumMarginIRL:
    def __init__(self, game: SnakeGame, expert_agent: SnakeAgent, max_iters: int, rollouts_per_iter: int, rollout_depth: int) -> None:
        self.game = game
        self.expert_agent = expert_agent

        self.max_iters = max_iters
        self.rollouts_per_iter = rollouts_per_iter
        self.rollout_depth = rollout_depth
        self.mixing_scalar = rollout_depth ** -3

    def optimize(self, game: SnakeGame, param_agent: ParameterizedSnakeAgent) -> None:
        param_agents: list[ParameterizedSnakeAgent] = []

        def policy(game: SnakeGame) -> int:
            return self.expert_agent.policy(game)

        for iter in range(self.max_iters):
            print(f"\n_______ Iteration {iter + 1} _______\n")

            observations = []
            actions = []

            for _ in range(self.rollouts_per_iter):
                self.game.reset()
                obs = self.game.observe()

                for _ in range(self.rollout_depth):
                    action = policy(game)
                    terminated, _ = game.step(action)

                    observations.append(obs)
                    actions.append(action)

                    if terminated:
                        break

                    obs = self.game.observe()
            
            observations = np.array(observations)
            actions = np.array(actions)

            train_policy_network(param_agent.policy_network, observations, actions)

            param_agents.append(copy.deepcopy(param_agent.policy_network))

            mixture_dist = np.array([(1 - self.mixing_scalar) ** i for i in range(iter + 1)])
            mixture_dist /= np.sum(mixture_dist)

            def policy(game: SnakeGame) -> int:
                if random.random() < (1 - self.mixing_scalar) ** iter:
                    return self.expert_agent.policy(game)
                else:
                    param_agent_index = np.random.choice(len(mixture_dist), p=mixture_dist)

                    return param_agents[param_agent_index].policy(game)

if __name__ == "__main__":
    game = SnakeGame()
    expert_agent = GreedySnakeAgent()

    policy_network = PolicyNetwork(input_size=game.observation_size, hidden_size=game.observation_size, output_size=game.num_actions)
    param_agent = ParameterizedSnakeAgent(policy_network)

    maximum_margin_irl = MaximumMarginIRL(game, expert_agent, max_iters=10, rollouts_per_iter=1000, rollout_depth=1000)
    maximum_margin_irl.optimize(game, param_agent)

    compare_agents(game, expert_agent, param_agent, follow_expert_prob=0.5, num_rollouts=1000, rollout_depth=1000)
