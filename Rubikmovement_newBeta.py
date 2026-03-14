#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 19:37:19 2018

@author: kaustabh
"""




#making a single list representing all sides in a serial fashion

r = ['w1','w2','w3','w4','w5','w6','w7','w8','w9','b1','b2','b3','r1','r2','r3','g1','g2','g3','o1','o2','o3','b4','b5','b6','r4','r5','r6','g4','g5','g6','o4','o5','o6','b7','b8','b9','r7','r8','r9','g7','g8','g9','o7','o8','o9', 'y1','y2','y3','y4','y5','y6','y7','y8','y9']

real = r[:] 


# cube is the list and we can't use random.shuffle because certain colors stay together
    







def display(r):
    for j in range(15):
          for i in range(9):
              if j == 0 :
                  if i == 0 :
                     print(" ")
                  while i<4:
                       print(" ", end='')
                       i += 1
                  if i == 5:
                     print(r[9]+" "+r[21]+" "+r[33],end='\n')
                     
              if j == 1 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[10]+" "+r[22]+" "+r[34],end='\n')
                     

              if j == 2 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[11]+" "+r[23]+" "+r[35],end='\n')
                     
              if j == 3 :
                  if i == 0:
                   print(" ")
                     
              if j == 4 :
                  if i==0 :
                     print(r[0]+" "+r[3]+" "+r[6]+"  ",end='')
                  if i==3 :
                     print(r[12]+" "+r[24]+" "+r[36]+" ",end='')
                  if i==6 :
                      print(" "+r[45]+" "+r[48]+" "+r[51],end='\n')
                      
              if j == 5 :
                  if i==0 :
                     print(r[1]+" "+r[4]+" "+r[7]+"  ",end='')
                  if i==3 :
                     print(r[13]+" "+r[25]+" "+r[37]+" ",end='')
                  if i==6 :
                      print(" "+r[46]+" "+r[49]+" "+r[52],end='\n')
            
              if j == 6 :
                  if i==0 :
                     print(r[2]+" "+r[5]+" "+r[8]+"  ",end='')
                  if i==3 :
                     print(r[15]+" "+r[26]+" "+r[38]+" ",end='')
                  if i==6 :
                      print(" "+r[47]+" "+r[50]+" "+r[53],end='\n')
                      
              if j == 7 :
                  if i==0:
                   print(" ")
                      
              if j == 8 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[15]+" "+r[27]+" "+r[39],end='\n')
                
              if j == 9 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[16]+" "+r[28]+" "+r[40],end='\n')
                     
              if j == 10 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[17]+" "+r[29]+" "+r[41],end='\n')
                     
              if j == 11 :
                  if i==0 :
                   print(" ")
            
              if j == 12 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[18]+" "+r[30]+" "+r[42],end='\n')
                     
              if j == 13 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[19]+" "+r[31]+" "+r[43],end='\n')
                     
              if j == 14 :
                  while i<4:
                     print(" ", end='')
                     i += 1
                  if i == 5:
                     print(r[20]+" "+r[32]+" "+r[44],end='\n')
                  if i == 8:
                      print(" ")
            
    return ' '.join(r)


import collections
import random

import numpy as np
import pandas as pd


class ExperienceReplay(object):
    """ Represents the experience replay memory that can be randomly sampled. """

    def __init__(self, input_shape, num_actions, memory_size=100):
        """
        Create a new instance of experience replay memory.
        
        Args:
            input_shape: the shape of the agent state.
            num_actions: the number of actions allowed in the environment.
            memory_size: memory size limit (-1 for unlimited).
        """
        self.memory = collections.deque()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.memory_size = memory_size

    def reset(self):
        """ Erase the experience replay memory. """
        self.memory = collections.deque()

    def remember(self, state, action, reward, state_next, is_episode_end):
        """
        Store a new piece of experience into the replay memory.
        
        Args:
            state: state observed at the previous step.
            action: action taken at the previous step.
            reward: reward received at the beginning of the current step.
            state_next: state observed at the current step. 
            is_episode_end: whether the episode has ended with the current step.
        """
        memory_item = np.concatenate([
            state.flatten(),
            np.array(action).flatten(),
            np.array(reward).flatten(),
            state_next.flatten(),
            1 * np.array(is_episode_end).flatten()
        ])
        self.memory.append(memory_item)
        if 0 < self.memory_size < len(self.memory):
            self.memory.popleft()

    def get_batch(self, model, batch_size, discount_factor=0.9):
        """ Sample a batch from experience replay. """

        batch_size = min(len(self.memory), batch_size)
        experience = np.array(random.sample(self.memory, batch_size))
        input_dim = np.prod(self.input_shape)

        # Extract [S, a, r, S', end] from experience.
        states = experience[:, 0:input_dim]
        actions = experience[:, input_dim]
        rewards = experience[:, input_dim + 1]
        states_next = experience[:, input_dim + 2:2 * input_dim + 2]
        episode_ends = experience[:, 2 * input_dim + 2]

        # Reshape to match the batch structure.
        states = states.reshape((batch_size, ) + self.input_shape)
        actions = np.cast['int'](actions)
        rewards = rewards.repeat(self.num_actions).reshape((batch_size, self.num_actions))
        states_next = states_next.reshape((batch_size, ) + self.input_shape)
        episode_ends = episode_ends.repeat(self.num_actions).reshape((batch_size, self.num_actions))

        # Predict future state-action values.
        X = np.concatenate([states, states_next], axis=0)
        y = model.predict(X)
        Q_next = np.max(y[batch_size:], axis=1).repeat(self.num_actions).reshape((batch_size, self.num_actions))

        delta = np.zeros((batch_size, self.num_actions))
        delta[np.arange(batch_size), actions] = 1

        targets = (1 - delta) * y[:batch_size] + delta * (rewards + discount_factor * (1 - episode_ends) * Q_next)
        return states, targets





class RCube(object):
    """ Represents the snake that has a position, can move, and change directions. """

    def __init__(self, r):
        """
        Create a new Cube.
        
        Args:
        r is the cube
        """
        ALL_RUBIKS_ACTION = []
        self.next_action = self.right_ac
        self.next_actions = ALL_RUBIKS_ACTION

        self.body = r
        
    def solved(self):
        
        return self.reset()

    def shift(self,r,L): #r is cube list ; L is 12 element array to be shifted
      
        t1 = self.body[L[0]]
        t2 = self.body[L[1]]
        t3 = self.body[L[2]]
        
        self.body[L[0]] , self.body[L[3]] = self.body[L[3]] , self.body[L[0]]
        self.body[L[1]] , self.body[L[4]] = self.body[L[4]] , self.body[L[1]]
        self.body[L[2]] , self.body[L[5]] = self.body[L[5]] , self.body[L[2]]
         
        self.body[L[3]] , self.body[L[6]] = self.body[L[6]] , self.body[L[3]]
        self.body[L[4]] , self.body[L[7]] = self.body[L[7]] , self.body[L[4]]
        self.body[L[5]] , self.body[L[8]] = self.body[L[8]] , self.body[L[5]]
        
        self.body[L[6]] , self.body[L[9]] = self.body[L[9]] , self.body[L[6]]
        self.body[L[7]] , self.body[L[10]] = self.body[L[10]] , self.body[L[7]]
        self.body[L[8]] , self.body[L[11]] = self.body[L[11]] , self.body[L[8]]
           
        self.body[L[9]] = t1
        self.body[L[10]] = t2
        self.body[L[11]] = t3
        
        return self.body

    def rotate(self,r, L): #a face will also self.bodyotate
        
        t1 = self.body[L[0]]
        t2 = self.body[L[7]]
        
        self.body[L[0]] , self.body[L[6]] = self.body[L[6]] , self.body[L[0]]
        self.body[L[7]] , self.body[L[5]] = self.body[L[5]] , self.body[L[7]]
        self.body[L[6]] , self.body[L[4]] = self.body[L[4]] , self.body[L[6]]
        self.body[L[3]] , self.body[L[5]] = self.body[L[5]] , self.body[L[3]]
        self.body[L[2]] , self.body[L[4]] = self.body[L[4]] , self.body[L[2]]
        self.body[L[1]] , self.body[L[3]] = self.body[L[3]] , self.body[L[1]]
        self.body[L[2]] = t1
        self.body[L[1]] = t2
        
        return 
        
        
    
    def right_c(self):
        
        L = [33,34,35,36,37,38,39,40,41,42,43,44]
        rt = [45,48,51,52,53,50,47,46]
        self.rotate(self.body , rt)
        return self.shift(self.body, L)
    
    def right_ac(self):
        
        L = [33,34,35,36,37,38,39,40,41,42,43,44]
        L.reverse()
        rt = [45,48,51,52,53,50,47,46]
        rt.reverse()
        self.rotate(self.body,rt)
        return self.shift(self.body, L)
    
    def left_ac(self):
        
        rt = [0,1,2,5,8,7,6,3]
        self.rotate(self.body,rt)
        L = [9,10,11,12,13,14,15,16,17,18,19,20]
        return self.shift(self.body, L)
    
    def left_c(self):
        
        rt = [0,1,2,5,8,7,6,3]
        rt.reverse()
        self.rotate(self.body,rt)
        L = [9,10,11,12,13,14,15,16,17,18,19,20]
        L.reverse()
        return self.shift(self.body, L)
    
    def up_c(self):
        
        rt = [9,21,33,34,35,23,11,10]
        self.rotate(self.body, rt)
        L = [0,3,6,12,24,36,45,48,51,18,30,42]
        return self.shift(self.body, L)
    
    def up_ac(self):
        
        rt = [9,21,33,34,35,23,11,10]
        rt.reverse()
        self.rotate(self.body, rt)
        L = [0,3,6,12,24,36,45,48,51,18,30,42]
        L.reverse()
        return self.shift(self.body, L)
    
    def down_ac(self):
        
        rt = [15,16,17,29,41,40,39,27]
        self.rotate(self.body,rt)
        L = [2,5,8,14,26,38,47,50,53,20,32,44]
        return self.shift(self.body, L)
    
    def down_c(self):
        
        rt = [15,16,17,29,41,40,39,27]
        rt.reverse()
        self.rotate(self.body,rt)
        L = [2,5,8,14,26,38,47,50,53,20,32,44]
        L.reverse()
        return self.shift(self.body, L)
    
    def front_c(self):
        
        rt = [12,24,36,37,38,26,14,13]
        self.rotate(self.body,rt)
        L = [11,23,35,45,46,47,39,27,15,8,7,6]
        return self.shift(self.body, L)
    
    def front_ac(self):
    
        rt = [12,24,36,37,38,26,14,13]
        rt.reverse()
        self.rotate(self.body,rt)
        L = [11,23,35,45,46,47,39,27,15,8,7,6]
        L.reverse()
        return self.shift(self.body, L)
    
    def back_c(self):
        
        rt= [18,30,42,43,44,32,20,19]
        self.rotate(self.body,rt)
        L = [0,3,6,9,21,33,45,48,51,15,27,39]
        return self.shift(self.body, L)
    
    def back_ac(self):
        
        rt= [18,30,42,43,44,32,20,19]
        rt.reverse()
        self.rotate(self.body,rt)
        L = [0,3,6,9,21,33,45,48,51,15,27,39]
        L.reverse()
        return self.shift(self.body, L)
    
    def ALL_RUBIKS_ACTION(self):
        
        ALL_RUBIKS_ACTION = [self.right_c(),self.left_c(),self.up_c(),self.down_c(),self.front_c(),self.back_c(),self.back_ac(),self.right_ac(),self.left_ac(),self.up_ac(),self.front_ac(),self.down_ac()]
        
        return ALL_RUBIKS_ACTION
    
    
    def shuffle(self):
        
        import random
        for i in range(random.randint(17,32)):
            random.choice(self.ALL_RUBIKS_ACTION)
        return self.body
    
    def reset(self):
        
        return real[:]

    def current_state(self):
        
        return self.body
    
    def peek_next_action(self):
        
        return self.next_action
    
    def move(self):
         
        return self.peek_next_action()
    

ALL_RUBIKS_ACTION = RCube(r).ALL_RUBIKS_ACTION()
        
    
RubiksCube = RCube(r)

class Rubiks_listType(object):
    """ Defines all types of cells that can be found in the game. """

    RUBIKSCUBE = 0
    MOVE = 1
    SOLVED = 2
    LOOP = 3


class Field(object):
    """ Represents the playing field for the Cube game. """

    def __init__(self, Rmap = None):
        """
        Create a new Snake field.
        
        Args:
            level_map: a list of strings representing the field objects (1 string per row).
        """
        self.current_shape = RubiksCube.current_state
        self.next_actions = None
        self.loop_actions = set()
        self.Rmap_to_list_type = {
            'R': Rubiks_listType.RUBIKSCUBE,
            'M': Rubiks_listType.MOVE,
            'S': Rubiks_listType.SOLVED,
            'L': Rubiks_listType.LOOP,
        }
        
        self.Rmap = Rmap
        self.list_type_to_Rmap = {
            Rubiks_listType: symbol
            for symbol, list_type in self.list_type_to_Rmap.items()
        }
        
    def __getaction__(self, action):
        """ Get the type of cell at the given point. """
        
        return action
    
    def __setitem__(self, action, Rubiks_listType):
        """ Update the type of cell at the given point. """
        self.next_actions = action

        # Do some internal bookkeeping to not rely on random selection of blank cells.
        if action == Rubiks_listType.LOOP:
            self.loop_actions.add(action)
        else:
            if action in self.loop_actions:
                self.loop_actions.remove(action)

    def __str__(self):
        return '\n'.join(
            ''.join(self._cell_type_to_level_map[cell] for cell in row)
            for row in self._cells
        )
        
    @property
    def size(self):
        """ Get the size of the field (size == width == height). """
        return self.Rmap

    def create_shape(self):
        """ Create a new field based on the level Rmap. """
        try:
            self.new = RubiksCube.shuffle(self.current_shape)
            return self.new
        except KeyError as err:
            raise ValueError('INVALID r[]')
            

    def update_rubiks_cube(self, old_list, new_action):
        """
        Update field cells according to the new snake position.
        
        Environment must be as fast as possible to speed up agent training.
        Therefore, we'll sacrifice some duplication of information between
        the snake body and the field just to execute timesteps faster.
        
        Args:
            old_head: position of the head before the move. 
            old_tail: position of the tail before the move.
            new_head: position of the head after the move.
        """
        self[old_list] = new_action

    
    
class AgentBase(object):
    """ Represents an intelligent agent for the Rubik's environment. """

    def begin_episode(self):
        """ Reset the agent for a new episode. """
        pass

    def act(self, observation, reward):
        """
        Choose the next action to take.
        Args:
            observation: observable state for the current timestep. 
            reward: reward received at the beginning of the current timestep.
        Returns:
            The index of the action to take next.
        """
        return None

    def end_episode(self):
        """ Notify the agent that the episode has ended. """
        pass


class RandomActionAgent(AgentBase):
    """ Represents a Rubik's agent that takes a random action at every step. """

    def __init__(self):
        pass

    def begin_episode(self):
        pass

    def act(self, observation, reward):
        return random.choice(ALL_RUBIKS_ACTION)

    def end_episode(self):
        pass
    
    
class RubiksAction(object):
    """ Defines all possible actions the agent can take in the environment. """

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


ALL_RUBIKS_ACTION = [
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


import pprint
import time

least_moves = 1000

class Environment(object):
    """
    Represents the RL environment for the RUBIKS game that implements the game logic,
    provides rewards for the agent and keeps track of game statistics.
    """

    def __init__(self, config, verbose=1):
        """
        Create a new rubiks RL environment.
        
        Args:
            config (dict): level configuration, typically found in JSON configs.  
            verbose (int): verbosity level:
                0 = do not write any debug information;
                1 = write a CSV file containing the statistics for every episode;
                2 = same as 1, but also write a full log file containing the state of each timestep.
        """
        
        self.field = Field(Rmap=config['field'])
        self.cube = None
        self.correct_move = None
        self.initial_state = RubiksCube.body
        self.rewards = config['rewards']
        self.max_step_limit = config.get('max_step_limit', 2000)
        self.is_game_over = False

        self.timestep_index = 0
        self.current_Env_action = None
        self.stats = EpisodeStatistics()
        self.verbose = verbose
        self.debug_file = None
        self.stats_file = None
    
        
        self.reward_points_list = None
        
        
    def seed(self, value):
        """ Initialize the random state of the environment to make results reproducible. """
        random.seed(value)
        np.random.seed(value)

    @property
    def observation_shape(self):
        """ Get the shape of the state observed at each timestep. """
        return self.field.size, self.field.size

    @property
    def num_actions(self):
        """ Get the number of actions the agent can take. """
        return len(ALL_RUBIKS_ACTION)

    def new_episode(self):
        """ Reset the environment and begin a new episode. """
        self.field.create_shape()
        self.stats.reset()
        self.timestep_index = 0

        self.cube = RubiksCube.shuffle()
        self.current_action = None
        self.is_game_over = False

        result = TimestepResult(
            observation=self.get_observation(),
            reward=0,
            is_episode_end=self.is_game_over
        )

        self.record_timestep_stats(result)
        return result

    def record_timestep_stats(self, result):
        """ Record environment statistics according to the verbosity level. """
        timestamp = time.strftime('%Y%m%d-%H%M%S')

        # Write CSV header for the stats file.
        if self.verbose >= 1 and self.stats_file is None:
            self.stats_file = open(f'snake-env-{timestamp}.csv', 'w')
            stats_csv_header_line = self.stats.to_dataframe()[:0].to_csv(index=None)
            print(stats_csv_header_line, file=self.stats_file, end='', flush=True)

        # Create a blank debug log file.
        if self.verbose >= 2 and self.debug_file is None:
            self.debug_file = open(f'cube-env-{timestamp}.log', 'w')

        self.stats.record_timestep(self.current_action, result)
        self.stats.timesteps_survived = self.timestep_index

        if self.verbose >= 2:
            print(result, file=self.debug_file)

        # Log episode stats if the appropriate verbosity level is set.
        if result.is_episode_end:
            if self.verbose >= 1:
                stats_csv_line = self.stats.to_dataframe().to_csv(header=False, index=None)
                print(stats_csv_line, file=self.stats_file, end='', flush=True)
            if self.verbose >= 2:
                print(self.stats, file=self.debug_file)

    def get_observation(self):
        """ Observe the state of the environment. """
        return np.copy(self.field.current_shape)

    def choose_action(self, action):
        """ Choose the action that will be taken at the next timestep. """

        self.current_action = action
        if   action == RubiksAction.RIGHTC:
            self.RubiksCube.right_c()
        elif action == RubiksAction.RIGHTAC:
            self.RubiksCube.right_ac()
        elif action == RubiksAction.BACKAC:
            self.RubiksCube.back_ac()
        elif action == RubiksAction.BACKC:
            self.RubiksCube.back_c()
        elif action == RubiksAction.LEFTC:
            self.RubiksCube.left_c()
        elif action == RubiksAction.LEFTAC:
            self.RubiksCube.left_ac()
        elif action == RubiksAction.UPC:
            self.RubiksCube.up_c()
        elif action == RubiksAction.UPAC:
            self.RubiksCube.up_ac()
        elif action == RubiksAction.DOWNAC:
            self.RubiksCube.down_ac()
        elif action == RubiksAction.DOWNC:
            self.RubiksCube.down_c()
        elif action == RubiksAction.FRONTC:
            self.RubiksCube.front_c()
        elif action == RubiksAction.FRONTAC:
            self.RubiksCube.front_ac()

    def timestep(self):
        """ Execute the timestep and return the new observable state. """

        self.timestep_index += 1
        reward = 0

        old_state = self.field.current_shape
        good_moves_total = 1
        

        # Are we about to eat the fruit?

        if self.cube.peek_next_action() == RubiksCube.solved():
            reward += self.rewards['solved'] * (10-0.01*good_moves_total)
            self.stats.solved += 1
            least_moves = good_moves_total
            if least_moves > good_moves_total :
                least_moves = good_moves_total
            good_moves_total = 0

        # If not, just move forward.
        else:
            self.cube.peek_next_action()
            reward += self.rewards['timestep']

        self.field.update_rubiks_cube(old_state, self.cube.current_state())#new_action

        # Hit a wall or own body?
        if not self.is_active():
            if self.has_solved():
                self.stats.termination_reason = 'solved'
            if self.repeating_steps():
                reward += self.rewards['loop'] * -1

            self.field[self.cube.body] = Rubiks_listType.RUBIKSCUBE
            self.is_game_over = True
            reward = self.rewards['died']

        # Exceeded the limit of moves?
        if self.timestep_index >= self.max_step_limit:
            self.is_game_over = True
            self.stats.termination_reason = 'timestep_limit_exceeded'

        result = TimestepResult(
            observation=self.get_observation(),
            reward=reward,
            is_episode_end=self.is_game_over
        )

        self.record_timestep_stats(result)
        return result


    def has_solved(self):
        """ True if the snake has hit a wall, False otherwise. """
        return self.field[self.cube.body] == Rubiks_listType.SOLVED

    def repeating_steps(self):
        """ True if the snake has hit its own body, False otherwise. """
        return self.field[self.cube.body] == Rubiks_listType.LOOP

    def is_active(self):
        """ True if the snake is still alive, False otherwise. """
        return not self.has_solved()


class TimestepResult(object):
    """ Represents the information provided to the agent after each timestep. """

    def __init__(self, observation, reward, is_episode_end):
        self.observation = observation
        self.reward = reward
        self.is_episode_end = is_episode_end

    def __str__(self):
        Rmap = '\n'.join([
            ''.join(str(cell) for cell in row)
            for row in self.observation
        ])
        return f'{Rmap}\nR = {self.reward}   end={self.is_episode_end}\n'


class EpisodeStatistics(object):
    """ Represents the summary of the agent's performance during the episode. """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Forget all previous statistics and prepare for a new episode. """
        self.timesteps_survived = 0
        self.sum_episode_rewards = 0
        self.good_moves = 0
        self.termination_reason = None
        self.action_counter = {
            action: 0
            for action in ALL_RUBIKS_ACTION
        }

    def record_timestep(self, action, result):
        """ Update the stats based on the current timestep results. """
        self.sum_episode_rewards += result.reward
        if action is not None:
            self.action_counter[action] += 1

    def flatten(self):
        """ Format all episode statistics as a flat object. """
        flat_stats = {
            'timesteps_survived': self.timesteps_survived,
            'sum_episode_rewards': self.sum_episode_rewards,
            'mean_reward': self.sum_episode_rewards / self.timesteps_survived if self.timesteps_survived else None,
            'fruits_eaten': self.fruits_eaten,
            'termination_reason': self.termination_reason,
        }
        flat_stats.update({
            f'action_counter_{action}': self.action_counter.get(action, 0)
            for action in ALL_RUBIKS_ACTION
        })
        return flat_stats

    def to_dataframe(self):
        """ Convert the episode statistics to a Pandas data frame. """
        return pd.DataFrame([self.flatten()])

    def __str__(self):
        return pprint.pformat(self.flatten())
    
    
class OpenAIGymEnvAdapter(object):
    """ Converts the Snake environment to OpenAI Gym environment format. """

    def __init__(self, env, action_space, observation_space):
        self.env = env
        self.action_space = OpenAIGymActionSpaceAdapter(action_space)
        self.observation_space = np.array(observation_space)

    def seed(self, value):
        self.env.seed(value)

    def reset(self):
        tsr = self.env.new_episode()
        return tsr.observation

    def step(self, action):
        self.env.choose_action(action)
        timestep_result = self.env.timestep()
        tsr = timestep_result
        return tsr.observation, tsr.reward, tsr.is_episode_end, {}
    
class OpenAIGymActionSpaceAdapter(object):
    """ Converts the Snake action space to OpenAI Gym action space format. """

    def __init__(self, actions):
        self.actions = np.array(actions)
        self.shape = self.actions.shape
        self.n = len(self.actions)

    def sample(self):
        return np.random.choice(self.actions)
    
    
import json 

def make_openai_gym_environment(config_filename):
    """
    Create an OpenAI Gym environment for the Snake game.
    
    Args:
        config_filename: JSON config for the Snake game level.
    Returns:
        An instance of OpenAI Gym environment.
    """

    with open(config_filename) as cfg:
        env_config = json.load(cfg)

    env_raw = Environment(config=env_config, verbose=1)
    env = OpenAIGymEnvAdapter(env_raw, ALL_RUBIKS_ACTION, np.zeros((10, 10)))
    return env
                     
                
                    
class DeepQNetworkAgent(AgentBase):
    """ Represents a Snake agent powered by DQN with experience replay. """

    def __init__(self, model, num_last_frames=4, memory_size=1000):
        """
        Create a new DQN-based agent.
        
        Args:
            model: a compiled DQN model.
            num_last_frames (int): the number of last frames the agent will consider.
            memory_size (int): memory size limit for experience replay (-1 for unlimited). 
        """
        assert model.input_shape[1] == num_last_frames, 'Model input shape should be (num_frames, grid_size, grid_size)'
        assert len(model.output_shape) == 2, 'Model output shape should be (num_samples, num_actions)'

        self.model = model
        self.num_last_frames = num_last_frames
        self.memory = ExperienceReplay((num_last_frames,) + model.input_shape[-2:], model.output_shape[-1], memory_size)
        self.frames = None

    def begin_episode(self):
        """ Reset the agent for a new episode. """
        self.frames = None

    def get_last_frames(self, observation):
        """
        Get the pixels of the last `num_last_frames` observations, the current frame being the last.
        
        Args:
            observation: observation at the current timestep. 
        Returns:
            Observations for the last `num_last_frames` frames.
        """
        frame = observation
        if self.frames is None:
            self.frames = collections.deque([frame] * self.num_last_frames)
        else:
            self.frames.append(frame)
            self.frames.popleft()
        return np.expand_dims(self.frames, 0)

    def train(self, env, num_episodes=1000, batch_size=50, discount_factor=0.9, checkpoint_freq=None,
              exploration_range=(1.0, 0.1), exploration_phase_size=0.5):
        """
        Train the agent to perform well in the given Rubiks environment.
        
        Args:
            env:
                an instance of Snake environment.
            num_episodes (int):
                the number of episodes to run during the training.
            batch_size (int):
                the size of the learning sample for experience replay.
            discount_factor (float):
                discount factor (gamma) for computing the value function.
            checkpoint_freq (int):
                the number of episodes after which a new model checkpoint will be created.
            exploration_range (tuple):
                a (max, min) range specifying how the exploration rate should decay over time. 
            exploration_phase_size (float):
                the percentage of the training process at which
                the exploration rate should reach its minimum.
        """

        # Calculate the constant exploration decay speed for each episode.
        max_exploration_rate, min_exploration_rate = exploration_range
        exploration_decay = ((max_exploration_rate - min_exploration_rate) / (num_episodes * exploration_phase_size))
        exploration_rate = max_exploration_rate

        for episode in range(num_episodes):
            # Reset the environment for the new episode.
            timestep = env.new_episode()
            self.begin_episode()
            game_over = False
            loss = 0.0

            # Observe the initial state.
            state = self.get_last_frames(timestep.observation)

            while not game_over:
                if np.random.random() < exploration_rate:
                    # Explore: take a random action.
                    action = np.random.randint(env.num_actions)
                else:
                    # Exploit: take the best known action for this state.
                    q = self.model.predict(state)
                    action = np.argmax(q[0])

                # Act on the environment.
                env.choose_action(action)
                timestep = env.timestep()

                # Remember a new piece of experience.
                reward = timestep.reward
                state_next = self.get_last_frames(timestep.observation)
                game_over = timestep.is_episode_end
                experience_item = [state, action, reward, state_next, game_over]
                self.memory.remember(*experience_item)
                state = state_next

                # Sample a random batch from experience.
                batch = self.memory.get_batch(
                    model=self.model,
                    batch_size=batch_size,
                    discount_factor=discount_factor
                )
                # Learn on the batch.
                if batch:
                    inputs, targets = batch
                    loss += float(self.model.train_on_batch(inputs, targets))

            if checkpoint_freq and (episode % checkpoint_freq) == 0:
                self.model.save(f'dqn-{episode:08d}.model')

            if exploration_rate > min_exploration_rate:
                exploration_rate -= exploration_decay

            summary = 'Episode {:5d}/{:5d} | Loss {:8.4f} | Exploration {:.2f} | ' + \
                      'Fruits {:2d} | Timesteps {:4d} | Total Reward {:4d}'
            print(summary.format(
                episode + 1, num_episodes, loss, exploration_rate,
                env.stats.fruits_eaten, env.stats.timesteps_survived, env.stats.sum_episode_rewards,
            ))

        self.model.save('dqn-final.model')

    def act(self, observation, reward):
        """
        Choose the next action to take.
        
        Args:
            observation: observable state for the current timestep. 
            reward: reward received at the beginning of the current timestep.
        Returns:
            The index of the action to take next.
        """
        state = self.get_last_frames(observation)
        q = self.model.predict(state)[0]
        return np.argmax(q)

