from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import copy
import random
import time
import colorsys
from models import PolicyNetwork, policy_network_to_action, policy_networks_to_action

ACTION_SHIFTS = [
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1)
]

class SnakeGame:
    def __init__(self, width: int = 4, height: int = 4, obs_trail_depth: int = 4) -> None:
        self.width: int = width
        self.height: int = height
        self.obs_trail_depth: int = obs_trail_depth
        self.num_actions: int = 4

        self.reset()

    def reset(self) -> None:
        self.board: np.ndarray = np.zeros((self.height, self.width))
        self.score: int = 1
        self.num_steps: int = 0
        self.actions: list[int] = []

        snake_int_pos, apple_int_pos = np.random.choice(np.arange(0, self.width * self.height), size=2, replace=False)

        self.snake_r: int = snake_int_pos // self.width
        self.snake_c: int = snake_int_pos % self.width

        self.board[self.snake_r, self.snake_c] = 1

        self.apple_r: int = apple_int_pos // self.width
        self.apple_c: int = apple_int_pos % self.width

    def is_square_valid(self, r: int, c: int) -> bool:
        return r >= 0 and r < self.height and c >= 0 and c < self.width

    # Returns terminated, ate_apple
    def step(self, action: int) -> Tuple[bool, bool]:
        self.num_steps += 1
        self.actions.append(action)

        shift_r, shift_c = ACTION_SHIFTS[action]

        self.snake_r += shift_r
        self.snake_c += shift_c
        
        if not self.is_square_valid(self.snake_r, self.snake_c):
            return True, False
        
        if self.board[self.snake_r, self.snake_c] > 0:
            return True, False
        
        ate_apple = self.snake_r == self.apple_r and self.snake_c == self.apple_c

        self.board[self.snake_r, self.snake_c] = self.score + 1

        if ate_apple:
            self.score += 1

            empty_rows, empty_cols = np.where(self.board == 0)

            if empty_rows.size == 0:
                return True, ate_apple

            random_idx = np.random.randint(0, empty_rows.size)
            self.apple_r = empty_rows[random_idx]
            self.apple_c = empty_cols[random_idx]
        else:
            self.board = np.maximum(self.board - 1, 0)

        return False, ate_apple
    
    # Compresses game state and action to binary vector observation
    def observe(self) -> np.ndarray:
        obs = np.zeros(((self.height + self.width) * 2 + self.obs_trail_depth * 4))

        obs[self.snake_r] = 1
        obs[self.height + self.snake_c] = 1
        obs[self.height + self.width + self.apple_c] = 1
        obs[self.height * 2 + self.width + self.apple_r] = 1

        head_apple_obs_size = (self.height + self.width) * 2

        for index, action in enumerate(self.actions[-1:(-1 - self.obs_trail_depth):-1]):
            obs[head_apple_obs_size + index * 4 + action] = 1
            
        return obs
    
    @property
    def observation_size(self) -> int:
        return (self.height + self.width) * 2 + self.obs_trail_depth * 4
    
    def visualize(self, frame_time: Optional[int], terminated: bool) -> None:
        if terminated:
            print("Steps: ", self.num_steps)
            print("Score: ", self.score)

        if frame_time is not None:
            print("\n" * 25)

            print("┌" + "─" * self.width + "┐")

            board_max = np.max(self.board)

            for r in range(self.height):
                line = "│"

                for c in range(self.width):
                    val = self.board[r, c]

                    if val > 0:
                        val_prop = val / board_max
                        raw_red, raw_green, raw_blue = colorsys.hsv_to_rgb(
                            0.5 * val_prop,
                            0 if val == board_max else 1,
                            0.6 + 0.4 * val_prop
                        )
                        red, green, blue = int(raw_red * 255), int(raw_green * 255), int(raw_blue * 255)

                        line += f"\033[38;2;{red};{green};{blue}m"

                        if val == board_max:
                            line += "█"
                        else:
                            line += "▒"

                        line += "\033[0m"
                    elif r == self.apple_r and c == self.apple_c:
                        line += f"\033[38;2;0;255;0m\033[0m"
                    else:
                        line += " "

                line += "│"

                print(line)

            print("└" + "─" * self.width + "┘")

            if not terminated:
                time.sleep(frame_time)

class SnakeAgent(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def policy(self, game: SnakeGame) -> int:
        pass

# Chooses action uniformly randomly among all actions that do not instantly terminate the game.
# (Chooses action 0 if all actions terminate the game)
class RandomSnakeAgent(SnakeAgent):
    def __init__(self) -> None:
        pass

    def policy(self, game: SnakeGame) -> int:
        safe_actions = []

        for action in range(game.num_actions):
            game_copy = copy.deepcopy(game)

            terminated, _ = game_copy.step(action)

            if not terminated:
                safe_actions.append(action)

        if len(safe_actions) > 0:
            return random.choice(safe_actions)
        
        return 0
    
# Chooses action that moves closest to apple among all actions that do not instantly terminate the game.
# (Chooses action 0 if all actions terminate the game)
class GreedySnakeAgent(SnakeAgent):
    def __init__(self) -> None:
        pass

    def policy(self, game: SnakeGame) -> int:
        best_action = None
        closest_apple_dist = float("inf")

        for action in range(game.num_actions):
            game_copy = copy.deepcopy(game)

            terminated, _ = game_copy.step(action)

            if not terminated:
                # Measure distance to apple in original game
                # (account for case that apple was eaten and regenerated in step)
                apple_dist = abs(game_copy.snake_r - game.apple_r) + abs(game_copy.snake_c - game.apple_c)

                if apple_dist < closest_apple_dist:
                    best_action = action
                    closest_apple_dist = apple_dist

        if best_action is not None:
            return best_action
        
        return 0

# Moves left if on right quarter of the board.
# Otherwise, chooses action that moves closest to apple among all actions that do not instantly terminate the game or move to the right quarter of the board.
# (Chooses left if all actions terminate the game)
class LeftGreedySnakeAgent(SnakeAgent):
    def __init__(self) -> None:
        pass

    def policy(self, game: SnakeGame) -> int:
        if game.snake_c >= game.width * 3 / 4:
            return 3

        best_action = None
        closest_apple_dist = float("inf")

        for action in range(game.num_actions):
            game_copy = copy.deepcopy(game)

            terminated, _ = game_copy.step(action)

            if not terminated and game_copy.snake_c < game.width * 3 / 4:
                # Measure distance to apple in original game
                # (account for case that apple was eaten and regenerated in step)
                apple_dist = abs(game_copy.snake_r - game.apple_r) + abs(game_copy.snake_c - game.apple_c)

                if apple_dist < closest_apple_dist:
                    best_action = action
                    closest_apple_dist = apple_dist

        if best_action is not None:
            return best_action
        
        return 3
    
# Moves right if on left quarter of the board.
# Otherwise, chooses action that moves closest to apple among all actions that do not instantly terminate the game or move to the left quarter of the board.
# (Chooses right if all actions terminate the game)
class RightGreedySnakeAgent(SnakeAgent):
    def __init__(self) -> None:
        pass

    def policy(self, game: SnakeGame) -> int:
        if game.snake_c < game.width / 4:
            return 1

        best_action = None
        closest_apple_dist = float("inf")

        for action in range(game.num_actions):
            game_copy = copy.deepcopy(game)

            terminated, _ = game_copy.step(action)

            if not terminated and game_copy.snake_c >= game.width / 4:
                # Measure distance to apple in original game
                # (account for case that apple was eaten and regenerated in step)
                apple_dist = abs(game_copy.snake_r - game.apple_r) + abs(game_copy.snake_c - game.apple_c)

                if apple_dist < closest_apple_dist:
                    best_action = action
                    closest_apple_dist = apple_dist

        if best_action is not None:
            return best_action
        
        return 1

# Chooses action that moves furthest from apple
class AllergicSnakeAgent(SnakeAgent):
    def __init__(self) -> None:
        pass

    def policy(self, game: SnakeGame) -> int:
        apple_dists = np.array([
            game.apple_r - game.snake_r,
            game.snake_c - game.apple_c,
            game.snake_r - game.apple_r,
            game.apple_c - game.snake_c
        ])

        return np.argmax(apple_dists)
    
# Chooses action to hit wall as fast as possible.
class WallSnakeAgent(SnakeAgent):
    def __init__(self) -> None:
        pass

    def policy(self, game: SnakeGame) -> int:
        wall_dists = np.array([
            game.snake_r + 1,
            game.width - game.snake_c,
            game.height - game.snake_r,
            game.snake_c + 1
        ])

        return np.argmin(wall_dists)
    
# Zig-zags in cycle that covers all squares (only works on even sized games)
class ZigZagSnakeAgent(SnakeAgent):
    def __init__(self) -> None:
        pass

    def policy(self, game: SnakeGame) -> int:
        if game.snake_c == 0:
            if game.snake_r == 0:
                return 1
            else:
                return 0
        elif game.snake_c == 1:
            if game.snake_r == 0:
                return 1
            elif game.snake_r < game.height - 1:
                if game.snake_r % 2 == 0:
                    return 1
                else:
                    return 2
            else:
                return 3
        elif game.snake_c < game.width - 1:
            if game.snake_r % 2 == 0:
                return 1
            else:
                return 3
        else:
            if game.snake_r % 2 == 0:
                return 2
            else:
                return 3

# Selects action according to output of policy network
class ParameterizedSnakeAgent(SnakeAgent):
    def __init__(self, policy_network: PolicyNetwork) -> None:
        self.policy_network = policy_network

    def policy(self, game: SnakeGame) -> int:
        obs = game.observe()

        return policy_network_to_action(self.policy_network, obs)
    
# Selects action according to outputs of multiple rated policy networks
class RatedParameterizedSnakeAgent(SnakeAgent):
    def __init__(self, policy_networks: list[PolicyNetwork], ratings: list[float]) -> None:
        self.policy_networks = policy_networks
        self.ratings = ratings

    def policy(self, game: SnakeGame) -> int:
        obs = game.observe()

        return policy_networks_to_action(self.policy_networks, self.ratings, obs)

def visualize_game(game: SnakeGame, agent: SnakeAgent, rollout_depth: int, frame_time: Optional[int]) -> None:
    game.reset()

    terminated = False

    for _ in range(rollout_depth):
        game.visualize(frame_time, terminated)

        action = agent.policy(game)
        terminated, _ = game.step(action)

        if terminated:
            break

    game.visualize(frame_time, terminated)

def compare_agents(game: SnakeGame, demo_agent: SnakeAgent, learned_agent: SnakeAgent, follow_demo_prob: float, num_rollouts: int, rollout_depth: int) -> None:
    num_steps = 0
    num_matched_steps = 0

    for _ in range(num_rollouts):
        game.reset()

        for _ in range(rollout_depth):
            param_agent_action = learned_agent.policy(game)
            demo_agent_action = demo_agent.policy(game)

            if param_agent_action == demo_agent_action:
                num_matched_steps += 1

            action = demo_agent_action if random.random() < follow_demo_prob else param_agent_action

            terminated, _ = game.step(action)

            num_steps += 1

            if terminated:
                break

    accuracy = num_matched_steps / num_steps

    print(f"Param agent matched {demo_agent.__class__.__name__} agent with accuracy {accuracy * 100:.2f}% ({num_matched_steps}/{num_steps})")

def analyze_agent(game: SnakeGame, agent: SnakeAgent, num_rollouts: int, rollout_depth: int) -> None:
    num_steps = 0
    total_score = 0

    for _ in range(num_rollouts):
        game.reset()

        for _ in range(rollout_depth):
            action = agent.policy(game)

            terminated, _ = game.step(action)

            num_steps += 1

            if terminated:
                break

        total_score += game.score

    average_score = total_score / num_rollouts
    average_rollout_steps = num_steps / num_rollouts

    print(f"{agent.__class__.__name__} acheived average score {average_score:.2f} and average rollout steps {average_rollout_steps:.2f}")

if __name__ == "__main__":
    game = SnakeGame(8, 8)
    agent = LeftGreedySnakeAgent()
    
    visualize_game(game, agent, rollout_depth=1000, frame_time=0.02)
