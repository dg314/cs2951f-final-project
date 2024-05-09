from abc import ABC, abstractmethod
import numpy as np
from snake import SnakeGame, SnakeAgent, AllergicSnakeAgent, RandomSnakeAgent, ZigZagSnakeAgent, ParameterizedSnakeAgent, RatedParameterizedSnakeAgent, GreedySnakeAgent, LeftGreedySnakeAgent, RightGreedySnakeAgent, compare_agents, analyze_agent
from models import PolicyNetwork, train_policy_network, Discriminator, train_gan
import copy
import random
from typing import Tuple, Callable

class ImitationLearner(ABC):
    @abstractmethod
    def __init__(self, game: SnakeGame, max_iters: int, rollouts_per_iter: int, rollout_depth: int) -> None:
        pass

    @abstractmethod
    def optimize(self, game: SnakeGame, demo_agent: SnakeAgent, param_agent: ParameterizedSnakeAgent) -> None:
        pass

def sample_trajectory(game: SnakeGame, policy: Callable[[SnakeGame], int], rollouts_per_iter: int, rollout_depth: int) -> Tuple[np.ndarray, np.ndarray]:
    observations = []
    actions = []

    for _ in range(rollouts_per_iter):
        game.reset()
        obs = game.observe()

        for _ in range(rollout_depth):
            action = policy(game)
            terminated, _ = game.step(action)

            observations.append(obs)
            actions.append(action)

            if terminated:
                break

            obs = game.observe()
    
    observations = np.array(observations)
    actions = np.array(actions)

    return observations, actions

# Stochastic Mixing Iterative Learning
# (Algorithms for Decision Making 18.3)
class SMILe(ImitationLearner):
    def __init__(self, game: SnakeGame, max_iters: int, rollouts_per_iter: int, rollout_depth: int) -> None:
        self.game = game
        self.max_iters = max_iters
        self.rollouts_per_iter = rollouts_per_iter
        self.rollout_depth = rollout_depth
        self.mixing_scalar = rollout_depth ** -3

    def optimize(self, game: SnakeGame, demo_agent: SnakeAgent, param_agent: ParameterizedSnakeAgent) -> None:
        param_agents: list[ParameterizedSnakeAgent] = []

        def policy(game: SnakeGame) -> int:
            return demo_agent.policy(game)

        for iter in range(self.max_iters):
            print(f"\n_______ Iteration {iter + 1} _______\n")

            observations, actions = sample_trajectory(game, policy=policy, rollouts_per_iter=self.rollouts_per_iter, rollout_depth=self.rollout_depth)

            train_policy_network(param_agent.policy_network, observations, actions)

            param_agents.append(copy.deepcopy(param_agent))

            mixture_dist = np.array([(1 - self.mixing_scalar) ** i for i in range(iter + 1)])
            mixture_dist /= np.sum(mixture_dist)

            def policy(game: SnakeGame) -> int:
                if random.random() < (1 - self.mixing_scalar) ** iter:
                    return demo_agent.policy(game)
                else:
                    param_agent_index = np.random.choice(len(mixture_dist), p=mixture_dist)

                    return param_agents[param_agent_index].policy(game)

# Generative Adversarial Imitation Learning
# (Algorithms for Decision Making 18.6)
class GAIL(ImitationLearner):
    def __init__(self, game: SnakeGame, max_iters: int, rollouts_per_iter: int, rollout_depth: int) -> None:
        self.game = game
        self.max_iters = max_iters
        self.rollouts_per_iter = rollouts_per_iter
        self.rollout_depth = rollout_depth

    def optimize(self, game: SnakeGame, demo_agent: SnakeAgent, param_agent: ParameterizedSnakeAgent) -> None:
        input_size = game.observation_size + game.num_actions
        discriminator = Discriminator(input_size=input_size, hidden_size=input_size)

        for iter in range(self.max_iters):
            print(f"\n_______ Iteration {iter + 1} _______\n")

            def sample_demo_trajectories() -> np.ndarray:
                demo_observations, demo_actions = sample_trajectory(game, policy=demo_agent.policy, rollouts_per_iter=self.rollouts_per_iter, rollout_depth=self.rollout_depth)

                demo_trajectories = [np.concatenate((obs, np.eye(game.num_actions)[action])) for obs, action in zip(demo_observations, demo_actions)]

                return np.array(demo_trajectories)
            
            def sample_generator_observations() -> np.ndarray:
                generator_observations, _ = sample_trajectory(game, policy=param_agent.policy, rollouts_per_iter=self.rollouts_per_iter, rollout_depth=self.rollout_depth)

                return generator_observations

            train_gan(generator=param_agent.policy_network, discriminator=discriminator, sample_demo_trajectories=sample_demo_trajectories, sample_generator_observations=sample_generator_observations)
                
def run_olympic_judges_experiment(game: SnakeGame, demo_agents: list[SnakeAgent], imitation_learner: ImitationLearner) -> None:
    print(f"\n__________ Running Olympic Judges experiment for {imitation_learner.__class__.__name__} __________\n")

    param_agents: list[ParameterizedSnakeAgent] = []

    for demo_agent in demo_agents:
        print(f"\nOptimizing {demo_agent.__class__.__name__} policy network:\n")

        policy_network = PolicyNetwork(input_size=game.observation_size, hidden_size=game.observation_size, output_size=game.num_actions)
        param_agent = ParameterizedSnakeAgent(policy_network)
        
        imitation_learner.optimize(game, demo_agent, param_agent)

        param_agents.append(param_agent)
    
        print(f"\n{demo_agent.__class__.__name__} individual trial:")
        compare_agents(game, demo_agent, learned_agent=param_agent, follow_demo_prob=0.5, num_rollouts=100, rollout_depth=100)
        analyze_agent(game, agent=demo_agent, num_rollouts=100, rollout_depth=100)

    policy_networks = [param_agent.policy_network for param_agent in param_agents]

    for i in range(2 ** len(demo_agents)):
        ratings = [1 if (i >> n) & 1 == 1 else 0.5 for n in range(len(demo_agents))]
        rated_param_agent = RatedParameterizedSnakeAgent(policy_networks, ratings)

        print(f"\nCombined trial with ratings {ratings}")
        analyze_agent(game, agent=rated_param_agent, num_rollouts=100, rollout_depth=100)

if __name__ == "__main__":
    game = SnakeGame()
    demo_agents = [
        LeftGreedySnakeAgent(),
        RightGreedySnakeAgent()
    ]

    imitation_learner = SMILe(game, max_iters=10, rollouts_per_iter=100, rollout_depth=100)

    run_olympic_judges_experiment(game, demo_agents, imitation_learner=imitation_learner)
