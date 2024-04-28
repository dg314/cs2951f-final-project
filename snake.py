from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import copy
import random
import time
import colorsys
import math

class SnakeGame:
    def __init__(self, width: int, height: int) -> None:
        self.width: int = width
        self.height: int = height
        self.num_actions: int = 4

        self.reset()

    def reset(self) -> None:
        self.board: np.ndarray = np.zeros((self.height, self.width))
        self.score: int = 1
        self.num_steps: int = 0

        snake_int_pos, apple_int_pos = np.random.choice(np.arange(0, self.width * self.height), size=2, replace=False)

        self.snake_r: int = snake_int_pos // self.width
        self.snake_c: int = snake_int_pos % self.width

        self.board[self.snake_r, self.snake_c] = 1

        self.apple_r: int = apple_int_pos // self.width
        self.apple_c: int = apple_int_pos % self.width

    # Returns terminated, ate_apple
    def step(self, action: int) -> Tuple[bool, bool]:
        self.num_steps += 1

        if action == 0:
            self.snake_r -= 1
        elif action == 1:
            self.snake_c += 1
        elif action == 2:
            self.snake_r += 1
        elif action == 3:
            self.snake_c -= 1
        else:
            raise Exception(f"Invalid action {action}")
        
        if self.snake_r < 0 or self.snake_r >= self.height or self.snake_c < 0 or self.snake_c >= self.width:
            return True, False
        
        if self.board[self.snake_r, self.snake_c] > 0:
            return True, False
        
        ate_apple = self.snake_r == self.apple_r and self.snake_c == self.apple_c

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

        self.board[self.snake_r, self.snake_c] = self.score

        return False, ate_apple
    
    def visualize(self, visualize_mode: str, terminated: bool) -> None:
        if visualize_mode == "text":
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
                            0.5 + 0.5 * val_prop
                        )
                        red, green, blue = int(raw_red * 255), int(raw_green * 255), int(raw_blue * 255)

                        line += f"\033[38;2;{red};{green};{blue}m"

                        if val == board_max:
                            line += "█"
                        elif val >= board_max * 2 / 3:
                            line += "▓"
                        elif val >= board_max / 3:
                            line += "▒"
                        else:
                            line += "░"

                        line += "\033[0m"
                    elif r == self.apple_r and c == self.apple_c:
                        line += f"\033[38;2;0;255;0m\033[0m"
                    else:
                        line += " "

                line += "│"

                print(line)

            print("└" + "─" * self.width + "┘")

            time.sleep(0.1)
        elif visualize_mode != "none":
            raise Exception(f"Invalid visualize_mode {visualize_mode}")

        if terminated:
            print("Steps: ", self.num_steps)
            print("Score: ", self.score)

class SnakeBot(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def select_action(self, game: SnakeGame) -> int:
        pass

# Chooses action uniformly randomly among all actions that do not instantly terminate the game.
# (Chooses action 0 if all actions terminate the game)
class RandomSnakeBot(SnakeBot):
    def __init__(self) -> None:
        pass

    def select_action(self, game: SnakeGame) -> int:
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
class GreedySnakeBot(SnakeBot):
    def __init__(self) -> None:
        pass

    def select_action(self, game: SnakeGame) -> int:
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
    
# Chooses action to hit wall as fast as possible.
class WallSnakeBot(SnakeBot):
    def __init__(self) -> None:
        pass

    def select_action(self, game: SnakeGame) -> int:
        wall_dists = np.array([
            game.snake_r + 1,
            game.width - game.snake_c,
            game.height - game.snake_r,
            game.snake_c + 1
        ])

        return np.argmin(wall_dists)

def play_game(game: SnakeGame, bot: SnakeBot, visualize_mode: str) -> None:
    game.reset()

    terminated = False

    while not terminated:
        game.visualize(visualize_mode, terminated)

        action = bot.select_action(game)
        terminated, _ = game.step(action)

    game.visualize(visualize_mode, terminated)

if __name__ == "__main__":
    game = SnakeGame(10, 10)
    bot = GreedySnakeBot()
    
    play_game(game, bot, visualize_mode="text")
