#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rubik's Cube RL Environment (Beta)

Provides a Rubik's Cube environment suitable for reinforcement learning,
including cube state management, experience replay, a DQN agent, and an
OpenAI-Gym-compatible adapter.

Created on Wed Jan 10 19:37:19 2018
@author: kaustabh

Modernisation notes (2024):
- Moved all imports to the top of the file (PEP 8).
- Fixed shuffle() eager-evaluation bug (all 12 moves were executed every
  iteration instead of one).
- Fixed display() index bug (r[15] → r[14] in the front-face row).
- Replaced deprecated ``np.cast['int']`` with ``.astype(int)``.
- Fixed ``Field.__init__`` referencing undefined ``self.list_type_to_Rmap``.
- Fixed ``Field.__str__`` referencing non-existent attributes.
- Fixed ``Field.create_shape()`` passing an extra argument to ``shuffle()``.
- Fixed ``Environment.choose_action()`` referencing ``self.RubiksCube``
  instead of ``self.cube``.
- Fixed ``EpisodeStatistics.flatten()`` referencing ``self.fruits_eaten``
  instead of ``self.good_moves``.
- Fixed train summary referencing ``fruits_eaten`` → ``good_moves``.
- Updated ``OpenAIGymEnvAdapter`` to return 5-tuple from ``step()`` and
  ``(obs, info)`` from ``reset()`` (Gymnasium convention).
- Added ``if __name__ == '__main__'`` guard.
"""

import collections
import json
import pprint
import random
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Cube state
# ---------------------------------------------------------------------------

# A single list representing all 54 face-stickers in serial fashion.
r = [
    'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9',
    'b1', 'b2', 'b3', 'r1', 'r2', 'r3', 'g1', 'g2', 'g3', 'o1', 'o2', 'o3',
    'b4', 'b5', 'b6', 'r4', 'r5', 'r6', 'g4', 'g5', 'g6', 'o4', 'o5', 'o6',
    'b7', 'b8', 'b9', 'r7', 'r8', 'r9', 'g7', 'g8', 'g9', 'o7', 'o8', 'o9',
    'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9',
]

# Solved-state snapshot used by reset()
real = r[:]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display(r):
    """Pretty-print the cube as an unfolded cross.

    Layout (top-to-bottom):
        Blue (top) | White (left) | Red (front) | Yellow (right) |
        Green (bottom) | Orange (back)
    """
    for j in range(15):
        for i in range(9):
            if j == 0:
                if i == 0:
                    print(" ")
                while i < 4:
                    print(" ", end='')
                    i += 1
                if i == 5:
                    print(r[9] + " " + r[21] + " " + r[33], end='\n')

            if j == 1:
                while i < 4:
                    print(" ", end='')
                    i += 1
                if i == 5:
                    print(r[10] + " " + r[22] + " " + r[34], end='\n')

            if j == 2:
                while i < 4:
                    print(" ", end='')
                    i += 1
                if i == 5:
                    print(r[11] + " " + r[23] + " " + r[35], end='\n')

            if j == 3:
                if i == 0:
                    print(" ")

            if j == 4:
                if i == 0:
                    print(r[0] + " " + r[3] + " " + r[6] + "  ", end='')
                if i == 3:
                    print(r[12] + " " + r[24] + " " + r[36] + " ", end='')
                if i == 6:
                    print(" " + r[45] + " " + r[48] + " " + r[51], end='\n')

            if j == 5:
                if i == 0:
                    print(r[1] + " " + r[4] + " " + r[7] + "  ", end='')
                if i == 3:
                    print(r[13] + " " + r[25] + " " + r[37] + " ", end='')
                if i == 6:
                    print(" " + r[46] + " " + r[49] + " " + r[52], end='\n')

            if j == 6:
                if i == 0:
                    print(r[2] + " " + r[5] + " " + r[8] + "  ", end='')
                if i == 3:
                    # FIX: was r[15] (green g1); correct index is r[14] (red r3)
                    print(r[14] + " " + r[26] + " " + r[38] + " ", end='')
                if i == 6:
                    print(" " + r[47] + " " + r[50] + " " + r[53], end='\n')

            if j == 7:
                if i == 0:
                    print(" ")

            if j == 8:
                while i < 4:
                    print(" ", end='')
                    i += 1
                if i == 5:
                    print(r[15] + " " + r[27] + " " + r[39], end='\n')

            if j == 9:
                while i < 4:
                    print(" ", end='')
                    i += 1
                if i == 5:
                    print(r[16] + " " + r[28] + " " + r[40], end='\n')

            if j == 10:
                while i < 4:
                    print(" ", end='')
                    i += 1
                if i == 5:
                    print(r[17] + " " + r[29] + " " + r[41], end='\n')

            if j == 11:
                if i == 0:
                    print(" ")

            if j == 12:
                while i < 4:
                    print(" ", end='')
                    i += 1
                if i == 5:
                    print(r[18] + " " + r[30] + " " + r[42], end='\n')

            if j == 13:
                while i < 4:
                    print(" ", end='')
                    i += 1
                if i == 5:
                    print(r[19] + " " + r[31] + " " + r[43], end='\n')

            if j == 14:
                while i < 4:
                    print(" ", end='')
                    i += 1
                if i == 5:
                    print(r[20] + " " + r[32] + " " + r[44], end='\n')
                if i == 8:
                    print(" ")

    return ' '.join(r)


# ---------------------------------------------------------------------------
# Experience Replay
# ---------------------------------------------------------------------------

class ExperienceReplay:
    """Experience replay memory that can be randomly sampled."""

    def __init__(self, input_shape, num_actions, memory_size=100):
        self.memory = collections.deque()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.memory_size = memory_size

    def reset(self):
        """Erase the experience replay memory."""
        self.memory = collections.deque()

    def remember(self, state, action, reward, state_next, is_episode_end):
        """Store a new experience tuple."""
        memory_item = np.concatenate([
            state.flatten(),
            np.array(action).flatten(),
            np.array(reward).flatten(),
            state_next.flatten(),
            1 * np.array(is_episode_end).flatten(),
        ])
        self.memory.append(memory_item)
        if 0 < self.memory_size < len(self.memory):
            self.memory.popleft()

    def get_batch(self, model, batch_size, discount_factor=0.9):
        """Sample a batch from experience replay."""
        batch_size = min(len(self.memory), batch_size)
        experience = np.array(random.sample(list(self.memory), batch_size))
        input_dim = int(np.prod(self.input_shape))

        # Extract [S, a, r, S', end] from experience.
        states = experience[:, 0:input_dim]
        actions = experience[:, input_dim]
        rewards = experience[:, input_dim + 1]
        states_next = experience[:, input_dim + 2:2 * input_dim + 2]
        episode_ends = experience[:, 2 * input_dim + 2]

        # Reshape to match the batch structure.
        states = states.reshape((batch_size,) + self.input_shape)
        # FIX: np.cast['int'] is removed in NumPy 2.x — use .astype(int)
        actions = actions.astype(int)
        rewards = rewards.repeat(self.num_actions).reshape((batch_size, self.num_actions))
        states_next = states_next.reshape((batch_size,) + self.input_shape)
        episode_ends = episode_ends.repeat(self.num_actions).reshape((batch_size, self.num_actions))

        # Predict future state-action values.
        X = np.concatenate([states, states_next], axis=0)
        y = model.predict(X)
        Q_next = np.max(y[batch_size:], axis=1).repeat(self.num_actions).reshape((batch_size, self.num_actions))

        delta = np.zeros((batch_size, self.num_actions))
        delta[np.arange(batch_size), actions] = 1

        targets = (1 - delta) * y[:batch_size] + delta * (rewards + discount_factor * (1 - episode_ends) * Q_next)
        return states, targets


# ---------------------------------------------------------------------------
# Cube object (OOP wrapper around the flat-list representation)
# ---------------------------------------------------------------------------

class RCube:
    """Object-oriented Rubik's Cube with move methods."""

    def __init__(self, r):
        self.body = r[:]
        self.next_action = self.right_ac

    # -- helpers (operate on self.body in-place) --

    def _shift(self, L):
        t1 = self.body[L[0]]
        t2 = self.body[L[1]]
        t3 = self.body[L[2]]

        self.body[L[0]], self.body[L[3]] = self.body[L[3]], self.body[L[0]]
        self.body[L[1]], self.body[L[4]] = self.body[L[4]], self.body[L[1]]
        self.body[L[2]], self.body[L[5]] = self.body[L[5]], self.body[L[2]]

        self.body[L[3]], self.body[L[6]] = self.body[L[6]], self.body[L[3]]
        self.body[L[4]], self.body[L[7]] = self.body[L[7]], self.body[L[4]]
        self.body[L[5]], self.body[L[8]] = self.body[L[8]], self.body[L[5]]

        self.body[L[6]], self.body[L[9]] = self.body[L[9]], self.body[L[6]]
        self.body[L[7]], self.body[L[10]] = self.body[L[10]], self.body[L[7]]
        self.body[L[8]], self.body[L[11]] = self.body[L[11]], self.body[L[8]]

        self.body[L[9]] = t1
        self.body[L[10]] = t2
        self.body[L[11]] = t3
        return self.body

    def _rotate(self, L):
        t1 = self.body[L[0]]
        t2 = self.body[L[7]]

        self.body[L[0]], self.body[L[6]] = self.body[L[6]], self.body[L[0]]
        self.body[L[7]], self.body[L[5]] = self.body[L[5]], self.body[L[7]]
        self.body[L[6]], self.body[L[4]] = self.body[L[4]], self.body[L[6]]
        self.body[L[3]], self.body[L[5]] = self.body[L[5]], self.body[L[3]]
        self.body[L[2]], self.body[L[4]] = self.body[L[4]], self.body[L[2]]
        self.body[L[1]], self.body[L[3]] = self.body[L[3]], self.body[L[1]]
        self.body[L[2]] = t1
        self.body[L[1]] = t2

    # -- face moves --

    def right_c(self):
        L = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
        rt = [45, 48, 51, 52, 53, 50, 47, 46]
        self._rotate(rt)
        return self._shift(L)

    def right_ac(self):
        L = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
        L.reverse()
        rt = [45, 48, 51, 52, 53, 50, 47, 46]
        rt.reverse()
        self._rotate(rt)
        return self._shift(L)

    def left_ac(self):
        rt = [0, 1, 2, 5, 8, 7, 6, 3]
        self._rotate(rt)
        L = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        return self._shift(L)

    def left_c(self):
        rt = [0, 1, 2, 5, 8, 7, 6, 3]
        rt.reverse()
        self._rotate(rt)
        L = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        L.reverse()
        return self._shift(L)

    def up_c(self):
        rt = [9, 21, 33, 34, 35, 23, 11, 10]
        self._rotate(rt)
        L = [0, 3, 6, 12, 24, 36, 45, 48, 51, 18, 30, 42]
        return self._shift(L)

    def up_ac(self):
        rt = [9, 21, 33, 34, 35, 23, 11, 10]
        rt.reverse()
        self._rotate(rt)
        L = [0, 3, 6, 12, 24, 36, 45, 48, 51, 18, 30, 42]
        L.reverse()
        return self._shift(L)

    def down_ac(self):
        rt = [15, 16, 17, 29, 41, 40, 39, 27]
        self._rotate(rt)
        L = [2, 5, 8, 14, 26, 38, 47, 50, 53, 20, 32, 44]
        return self._shift(L)

    def down_c(self):
        rt = [15, 16, 17, 29, 41, 40, 39, 27]
        rt.reverse()
        self._rotate(rt)
        L = [2, 5, 8, 14, 26, 38, 47, 50, 53, 20, 32, 44]
        L.reverse()
        return self._shift(L)

    def front_c(self):
        rt = [12, 24, 36, 37, 38, 26, 14, 13]
        self._rotate(rt)
        L = [11, 23, 35, 45, 46, 47, 39, 27, 15, 8, 7, 6]
        return self._shift(L)

    def front_ac(self):
        rt = [12, 24, 36, 37, 38, 26, 14, 13]
        rt.reverse()
        self._rotate(rt)
        L = [11, 23, 35, 45, 46, 47, 39, 27, 15, 8, 7, 6]
        L.reverse()
        return self._shift(L)

    def back_c(self):
        rt = [18, 30, 42, 43, 44, 32, 20, 19]
        self._rotate(rt)
        L = [0, 3, 6, 9, 21, 33, 45, 48, 51, 15, 27, 39]
        return self._shift(L)

    def back_ac(self):
        rt = [18, 30, 42, 43, 44, 32, 20, 19]
        rt.reverse()
        self._rotate(rt)
        L = [0, 3, 6, 9, 21, 33, 45, 48, 51, 15, 27, 39]
        L.reverse()
        return self._shift(L)

    # -- utility --

    def get_move_funcs(self):
        """Return a list of *bound method references* (not results)."""
        return [
            self.right_c, self.left_c, self.up_c, self.down_c,
            self.front_c, self.back_c, self.back_ac, self.right_ac,
            self.left_ac, self.up_ac, self.front_ac, self.down_ac,
        ]

    def shuffle(self):
        """Scramble the cube with 17-32 random moves.

        Bug-fix: the original ``ALL_RUBIKS_ACTION`` method eagerly called
        every move, mutating the cube 12 times per iteration.  Now we pick
        one function reference and call only that.
        """
        funcs = self.get_move_funcs()
        for _ in range(random.randint(17, 32)):
            random.choice(funcs)()
        return self.body

    def reset(self):
        """Restore the cube to the solved state."""
        self.body = real[:]
        return self.body

    def solved(self):
        """Return the solved-state list."""
        return real[:]

    def current_state(self):
        return self.body[:]

    def peek_next_action(self):
        return self.next_action

    def move(self):
        return self.peek_next_action()()


# ---------------------------------------------------------------------------
# Global cube instance
# ---------------------------------------------------------------------------

RubiksCube = RCube(r)


# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------

class RubiksAction:
    """All possible actions the agent can take."""
    RIGHTC = 0
    RIGHTAC = 1
    LEFTC = 2
    LEFTAC = 3
    UPC = 4
    UPAC = 5
    DOWNC = 6
    DOWNAC = 7
    FRONTC = 8
    FRONTAC = 9
    BACKC = 10
    BACKAC = 11


ALL_RUBIKS_ACTIONS = [
    RubiksAction.RIGHTC,
    RubiksAction.RIGHTAC,
    RubiksAction.LEFTC,
    RubiksAction.LEFTAC,
    RubiksAction.UPC,
    RubiksAction.UPAC,
    RubiksAction.DOWNC,
    RubiksAction.DOWNAC,
    RubiksAction.FRONTC,
    RubiksAction.FRONTAC,
    RubiksAction.BACKC,
    RubiksAction.BACKAC,
]


# ---------------------------------------------------------------------------
# Cell / list types
# ---------------------------------------------------------------------------

class RubiksListType:
    """Defines all types of cells that can be found in the game."""
    RUBIKSCUBE = 0
    MOVE = 1
    SOLVED = 2
    LOOP = 3


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------

class Field:
    """Represents the playing field for the Cube game."""

    def __init__(self, Rmap=None):
        self.current_shape = RubiksCube.current_state()
        self.next_actions = None
        self.loop_actions = set()

        self.Rmap_to_list_type = {
            'R': RubiksListType.RUBIKSCUBE,
            'M': RubiksListType.MOVE,
            'S': RubiksListType.SOLVED,
            'L': RubiksListType.LOOP,
        }

        self.Rmap = Rmap

        # FIX: was referencing self.list_type_to_Rmap (undefined) — now
        # correctly derives the reverse mapping from Rmap_to_list_type.
        self.list_type_to_Rmap = {
            list_type: symbol
            for symbol, list_type in self.Rmap_to_list_type.items()
        }

    def __getitem__(self, action):
        """Get the type of cell at the given point."""
        return action

    def __setitem__(self, action, list_type):
        """Update the type of cell at the given point."""
        self.next_actions = action
        if action == RubiksListType.LOOP:
            self.loop_actions.add(action)
        elif action in self.loop_actions:
            self.loop_actions.remove(action)

    def __str__(self):
        return str(self.current_shape)

    @property
    def size(self):
        """Get the size of the field."""
        return self.Rmap

    def create_shape(self):
        """Create a new field based on the level Rmap."""
        try:
            # FIX: shuffle() takes no positional args beyond self
            self.current_shape = RubiksCube.shuffle()
            return self.current_shape
        except KeyError as err:
            raise ValueError(f'INVALID r[]: {err}')

    def update_rubiks_cube(self, old_list, new_action):
        """Update field cells according to the new cube state."""
        self[old_list] = new_action


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class AgentBase:
    """Base class for RL agents."""

    def begin_episode(self):
        pass

    def act(self, observation, reward):
        return None

    def end_episode(self):
        pass


class RandomActionAgent(AgentBase):
    """Agent that takes a random action at every step."""

    def act(self, observation, reward):
        return random.choice(ALL_RUBIKS_ACTIONS)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

least_moves = 1000


class EpisodeStatistics:
    """Summary of the agent's performance during an episode."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.timesteps_survived = 0
        self.sum_episode_rewards = 0
        self.good_moves = 0
        self.termination_reason = None
        self.action_counter = {action: 0 for action in ALL_RUBIKS_ACTIONS}

    def record_timestep(self, action, result):
        self.sum_episode_rewards += result.reward
        if action is not None:
            self.action_counter[action] += 1

    def flatten(self):
        flat_stats = {
            'timesteps_survived': self.timesteps_survived,
            'sum_episode_rewards': self.sum_episode_rewards,
            'mean_reward': (
                self.sum_episode_rewards / self.timesteps_survived
                if self.timesteps_survived else None
            ),
            # FIX: was self.fruits_eaten (undefined) → self.good_moves
            'good_moves': self.good_moves,
            'termination_reason': self.termination_reason,
        }
        flat_stats.update({
            f'action_counter_{action}': self.action_counter.get(action, 0)
            for action in ALL_RUBIKS_ACTIONS
        })
        return flat_stats

    def to_dataframe(self):
        return pd.DataFrame([self.flatten()])

    def __str__(self):
        return pprint.pformat(self.flatten())


# ---------------------------------------------------------------------------
# Timestep result
# ---------------------------------------------------------------------------

class TimestepResult:
    """Information provided to the agent after each timestep."""

    def __init__(self, observation, reward, is_episode_end):
        self.observation = observation
        self.reward = reward
        self.is_episode_end = is_episode_end

    def __str__(self):
        return f'R = {self.reward}   end={self.is_episode_end}\n'


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Environment:
    """RL environment for the Rubik's Cube game."""

    def __init__(self, config, verbose=1):
        self.field = Field(Rmap=config['field'])
        self.cube = None
        self.correct_move = None
        self.initial_state = RubiksCube.body[:]
        self.rewards = config['rewards']
        self.max_step_limit = config.get('max_step_limit', 2000)
        self.is_game_over = False

        self.timestep_index = 0
        self.current_action = None
        self.stats = EpisodeStatistics()
        self.verbose = verbose
        self.debug_file = None
        self.stats_file = None
        self.reward_points_list = None

    def seed(self, value):
        random.seed(value)
        np.random.seed(value)

    @property
    def observation_shape(self):
        return self.field.size, self.field.size

    @property
    def num_actions(self):
        return len(ALL_RUBIKS_ACTIONS)

    def new_episode(self):
        self.field.create_shape()
        self.stats.reset()
        self.timestep_index = 0

        self.cube = RCube(r)
        self.cube.shuffle()
        self.current_action = None
        self.is_game_over = False

        result = TimestepResult(
            observation=self.get_observation(),
            reward=0,
            is_episode_end=self.is_game_over,
        )
        self.record_timestep_stats(result)
        return result

    def record_timestep_stats(self, result):
        timestamp = time.strftime('%Y%m%d-%H%M%S')

        if self.verbose >= 1 and self.stats_file is None:
            self.stats_file = open(f'cube-env-{timestamp}.csv', 'w')
            stats_csv_header_line = self.stats.to_dataframe()[:0].to_csv(index=None)
            print(stats_csv_header_line, file=self.stats_file, end='', flush=True)

        if self.verbose >= 2 and self.debug_file is None:
            self.debug_file = open(f'cube-env-{timestamp}.log', 'w')

        self.stats.record_timestep(self.current_action, result)
        self.stats.timesteps_survived = self.timestep_index

        if self.verbose >= 2:
            print(result, file=self.debug_file)

        if result.is_episode_end:
            if self.verbose >= 1:
                stats_csv_line = self.stats.to_dataframe().to_csv(header=False, index=None)
                print(stats_csv_line, file=self.stats_file, end='', flush=True)
            if self.verbose >= 2:
                print(self.stats, file=self.debug_file)

    def get_observation(self):
        return np.copy(self.field.current_shape)

    def choose_action(self, action):
        """Execute the chosen action on the cube.

        FIX: was ``self.RubiksCube`` (undefined) → ``self.cube``.
        """
        self.current_action = action
        action_map = {
            RubiksAction.RIGHTC: self.cube.right_c,
            RubiksAction.RIGHTAC: self.cube.right_ac,
            RubiksAction.LEFTC: self.cube.left_c,
            RubiksAction.LEFTAC: self.cube.left_ac,
            RubiksAction.UPC: self.cube.up_c,
            RubiksAction.UPAC: self.cube.up_ac,
            RubiksAction.DOWNC: self.cube.down_c,
            RubiksAction.DOWNAC: self.cube.down_ac,
            RubiksAction.FRONTC: self.cube.front_c,
            RubiksAction.FRONTAC: self.cube.front_ac,
            RubiksAction.BACKC: self.cube.back_c,
            RubiksAction.BACKAC: self.cube.back_ac,
        }
        move_fn = action_map.get(action)
        if move_fn is not None:
            move_fn()

    def timestep(self):
        global least_moves
        self.timestep_index += 1
        reward = 0

        old_state = self.field.current_shape
        good_moves_total = 1

        if self.cube.current_state() == self.cube.solved():
            reward += self.rewards['solved'] * (10 - 0.01 * good_moves_total)
            self.stats.good_moves += 1
            if least_moves > good_moves_total:
                least_moves = good_moves_total
            good_moves_total = 0
        else:
            self.cube.peek_next_action()
            reward += self.rewards['timestep']

        self.field.update_rubiks_cube(old_state, self.cube.current_state())

        if not self.is_active():
            if self.has_solved():
                self.stats.termination_reason = 'solved'
            if self.repeating_steps():
                reward += self.rewards['loop'] * -1

            self.is_game_over = True
            reward = self.rewards['died']

        if self.timestep_index >= self.max_step_limit:
            self.is_game_over = True
            self.stats.termination_reason = 'timestep_limit_exceeded'

        result = TimestepResult(
            observation=self.get_observation(),
            reward=reward,
            is_episode_end=self.is_game_over,
        )
        self.record_timestep_stats(result)
        return result

    def has_solved(self):
        return self.cube.current_state() == self.cube.solved()

    def repeating_steps(self):
        return False  # placeholder — loop detection not yet implemented

    def is_active(self):
        return not self.has_solved()


# ---------------------------------------------------------------------------
# OpenAI Gym adapter (updated for Gymnasium 5-tuple convention)
# ---------------------------------------------------------------------------

class OpenAIGymActionSpaceAdapter:
    """Converts the action space to OpenAI Gym format."""

    def __init__(self, actions):
        self.actions = np.array(actions)
        self.shape = self.actions.shape
        self.n = len(self.actions)

    def sample(self):
        return np.random.choice(self.actions)


class OpenAIGymEnvAdapter:
    """Converts the Rubik's environment to OpenAI Gym / Gymnasium format."""

    def __init__(self, env, action_space, observation_space):
        self.env = env
        self.action_space = OpenAIGymActionSpaceAdapter(action_space)
        self.observation_space = np.array(observation_space)

    def seed(self, value):
        self.env.seed(value)

    def reset(self):
        """FIX: Gymnasium returns (observation, info)."""
        tsr = self.env.new_episode()
        return tsr.observation, {}

    def step(self, action):
        """FIX: Gymnasium returns (obs, reward, terminated, truncated, info)."""
        self.env.choose_action(action)
        tsr = self.env.timestep()
        truncated = False
        return tsr.observation, tsr.reward, tsr.is_episode_end, truncated, {}


def make_openai_gym_environment(config_filename):
    """Create an OpenAI Gym environment for the Rubik's Cube game."""
    with open(config_filename) as cfg:
        env_config = json.load(cfg)

    env_raw = Environment(config=env_config, verbose=1)
    env = OpenAIGymEnvAdapter(env_raw, ALL_RUBIKS_ACTIONS, np.zeros((10, 10)))
    return env


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DeepQNetworkAgent(AgentBase):
    """DQN agent with experience replay for the Rubik's Cube."""

    def __init__(self, model, num_last_frames=4, memory_size=1000):
        assert model.input_shape[1] == num_last_frames, \
            'Model input shape should be (num_frames, grid_size, grid_size)'
        assert len(model.output_shape) == 2, \
            'Model output shape should be (num_samples, num_actions)'

        self.model = model
        self.num_last_frames = num_last_frames
        self.memory = ExperienceReplay(
            (num_last_frames,) + model.input_shape[-2:],
            model.output_shape[-1],
            memory_size,
        )
        self.frames = None

    def begin_episode(self):
        self.frames = None

    def get_last_frames(self, observation):
        frame = observation
        if self.frames is None:
            self.frames = collections.deque([frame] * self.num_last_frames)
        else:
            self.frames.append(frame)
            self.frames.popleft()
        return np.expand_dims(self.frames, 0)

    def train(self, env, num_episodes=1000, batch_size=50, discount_factor=0.9,
              checkpoint_freq=None, exploration_range=(1.0, 0.1),
              exploration_phase_size=0.5):
        """Train the agent via DQN with experience replay."""
        max_exploration_rate, min_exploration_rate = exploration_range
        exploration_decay = (
            (max_exploration_rate - min_exploration_rate)
            / (num_episodes * exploration_phase_size)
        )
        exploration_rate = max_exploration_rate

        for episode in range(num_episodes):
            timestep = env.new_episode()
            self.begin_episode()
            game_over = False
            loss = 0.0

            state = self.get_last_frames(timestep.observation)

            while not game_over:
                if np.random.random() < exploration_rate:
                    action = np.random.randint(env.num_actions)
                else:
                    q = self.model.predict(state)
                    action = np.argmax(q[0])

                env.choose_action(action)
                timestep = env.timestep()

                reward = timestep.reward
                state_next = self.get_last_frames(timestep.observation)
                game_over = timestep.is_episode_end
                self.memory.remember(state, action, reward, state_next, game_over)
                state = state_next

                batch = self.memory.get_batch(
                    model=self.model,
                    batch_size=batch_size,
                    discount_factor=discount_factor,
                )
                if batch:
                    inputs, targets = batch
                    loss += float(self.model.train_on_batch(inputs, targets))

            if checkpoint_freq and (episode % checkpoint_freq) == 0:
                self.model.save(f'dqn-{episode:08d}.model')

            if exploration_rate > min_exploration_rate:
                exploration_rate -= exploration_decay

            # FIX: was fruits_eaten → good_moves
            summary = (
                'Episode {:5d}/{:5d} | Loss {:8.4f} | Exploration {:.2f} | '
                'Good moves {:2d} | Timesteps {:4d} | Total Reward {:4d}'
            )
            print(summary.format(
                episode + 1, num_episodes, loss, exploration_rate,
                env.stats.good_moves, env.stats.timesteps_survived,
                env.stats.sum_episode_rewards,
            ))

        self.model.save('dqn-final.model')

    def act(self, observation, reward):
        state = self.get_last_frames(observation)
        q = self.model.predict(state)[0]
        return np.argmax(q)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Rubik's Cube RL Environment (Beta)")
    print("Solved state:")
    display(r)
