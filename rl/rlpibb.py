import os
import sys
import random
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pathlib as Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import custom modules
from evolutionary.individual import Individual


class RlPibb():
    """Policy Improvement with Black Box optimization agent to solve bipedal walker

    Args:
        params (list): List of parameters to initialize agent
    """

    def __init__(self, params):
        """Initialize agent"""

        # Initialize parameters for agent
        self.agent = Individual()
        self.agent.model.set_params(params)

        # Initialize environment
        self.env = gym.make('BipedalWalker-v3')

    def compute_reward(self, state):
        """Compute reward for agent"""

        # Get action from agent
        action = self.agent.choose_action()

        # Take action in environment
        next_state, reward, terminated, truncated, _ = self.env.step(action)
