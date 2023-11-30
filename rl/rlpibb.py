import os
import sys
import csv
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import custom modules
from evolutionary.individual import Individual
from controller.cpg_rbfn import CPG_RBFN


class RlPibb():
    """Policy Improvement with Black Box optimization agent to solve reinforcement learning legged locomotion tasks
    Uses CPG-RBFN network to control the robot and uses PIBB to optimize the weights of the network
    Implementation of the M.Thor paper

    Args:
        env_type (str): Gym environment type
        epochs (int): Number of epochs to train
        max_steps (int): Maximum number of steps per epoch
        rollout_size (int): Number of rollouts per epoch
        variance (float): Variance of the Gaussian noise for exploration of weights
        decay (float): Decay rate of the variance
    """

    def __init__(self, env_type: str, epochs: int=1000, max_steps: int=1000, rollout_size: int=10,
                 norm_constant: float=10.0, variance: float=1.0, decay:float=0.99):
        """Initialize parameters for RL-PIBB agent"""

        # Arguments
        self.env_type = env_type
        self.epochs = epochs
        self.max_steps = max_steps
        self.rollout_size = rollout_size
        self.norm_constant = norm_constant
        self.variance = variance
        self.decay = decay

        # Create environment
        self.env = gym.make(self.env_type)

        # Fixed parameters
        self.in_size = self.env.observation_space.shape[0]
        self.out_size = self.env.action_space.shape[0]

        # Model parameters
        self.rbfn_units = 20

        # Initialize model and agent
        # Centres of RBF are fixed calculated from formula
        self.model = CPG_RBFN(self.rbfn_units, self.out_size, fixed_centers=True)
        self.agent = Individual(self.model)

        # Reward history
        self.reward_history = []
        self.best_per_epoch = []
        self.mean_per_epoch = []
        self.mean_error_per_epoch = []

        # Weight update history
        self.weight_update_history = []

    def get_rollout_agents(self):
        """From current agent get the rollout agents with noise in the parameters"""

        # Create a list of agents with noise in the parameters
        rollout_agents = []
        rollout_agents_noise = []
        for i in range(self.rollout_size):
            agent = Individual(self.model)
            agent_param_noise = np.random.normal(0, self.variance, self.agent.get_params().shape)
            agent_params = self.agent.get_params() * agent_param_noise
            agent.set_params(agent_params)
            rollout_agents.append(agent)
            rollout_agents_noise.append(agent_param_noise)

        return rollout_agents, rollout_agents_noise

    def run_epoch(self):
        """Run an epoch of the RL-PIBB agent with rollout agents and return the epoch reward"""

        # Get the rollout agents
        rollout_agents, rollout_agent_noise = self.get_rollout_agents()

        # Get the rewards for the rollout agents
        rewards = []
        for agent in rollout_agents:

            # Reset the environment
            state, _ = self.env.reset()
            reward_sum = 0

            # Run each rollout agent for max_steps or until termination or truncation
            for _ in range(self.max_steps):
                # Choose an action
                action = agent.choose_action()

                # Take a step in the environment
                state, reward, terminated, truncated, _ = self.env.step(action)

                # Add reward to the sum
                reward_sum += reward

                # Break if terminated or truncated
                if terminated or truncated:
                    break

            # Reset CPG network
            agent.model.cpg_reset()

            # Add reward to the list
            rewards.append(reward_sum)

        # Get the reward statistics
        reward_best = np.max(rewards)
        reward_mean = np.mean(rewards)
        reward_mean_error = np.std(rewards) / np.sqrt(self.epochs)

        ## PIBB algorithm update
        # Get normalized rewards
        rewards_norm = np.exp(self.norm_constant * (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards)))
        rewards_norm /= np.sum(rewards_norm)

        # Scale the noise by rewards
        rollout_agent_noise = np.array(rollout_agent_noise)
        for i in range(self.rollout_size):
            rollout_agent_noise[i] *= rewards_norm[i]

        weight_update = np.sum(rollout_agent_noise, axis=0)

        return rewards, reward_best, reward_mean, reward_mean_error, weight_update

    def run(self, verbose: bool=False):
        """Run the algorithm"""

        # Run for the specified number of epochs
        for epoch in range(self.epochs):

            # Run an epoch
            rewards, reward_best, reward_mean, reward_mean_error, weight_update = self.run_epoch()

            # Record the reward statistics
            self.reward_history.append(rewards)
            self.best_per_epoch.append(reward_best)
            self.mean_per_epoch.append(reward_mean)
            self.mean_error_per_epoch.append(reward_mean_error)

            # Record the weight update history
            self.weight_update_history.append(weight_update)

            # Update the agent weights based rollout rewards using PIBB
            self.agent.set_params(self.agent.get_params() + weight_update)

            # Decay the variance
            self.variance *= self.decay

            # Print progress if verbose
            if verbose and (epoch % self.epochs // 10 == 0 or epoch == self.epochs - 1):
                print(f"Generation {epoch+1}: Best Reward {self.best_per_epoch[-1]}")

    def save(self, path: str):
        """Save reward history to csv and model of the last epoch"""

        # Create directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Save reward and weight update history to csv
        with open(os.path.join(path, "reward_history.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow("Reward History")
            writer.writerow(self.reward_history)
            writer.writerow("Best Reward per Epoch")
            writer.writerow(self.best_per_epoch)
            writer.writerow("Mean Reward per Epoch")
            writer.writerow(self.mean_per_epoch)
            writer.writerow("Mean Reward Error per Epoch")
            writer.writerow(self.mean_error_per_epoch)
            writer.writerow("Weight Update History")
            writer.writerow(self.weight_update_history)

        # Save the model
        torch.save(self.agent.model.state_dict(), os.path.join(path, "model.pt"))

    def get_plots(self, path: str, is_show: bool=False):
        """Plot the reward history statistics and save the plots"""

        # Plot reward of every rollout agent in every epoch
        plt.figure("Reward History All")
        plt.title("Reward History All")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        for i, rewards in enumerate(self.reward_history):
            plt.scatter([i] * len(rewards), rewards, s=1)
        plt.savefig(os.path.join(path, "reward_history_all.png"))

        # Plot best reward of every epoch
        plt.figure("Best Reward")
        plt.title("Best Reward per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.plot(self.best_per_epoch)
        plt.savefig(os.path.join(path, "best_reward.png"))

        # Plot mean reward of every epoch with error bars
        plt.figure("Mean Reward")
        plt.title("Mean Reward per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.plot(self.mean_per_epoch)
        plt.errorbar(np.arange(len(self.epochs)), self.mean_per_epoch, yerr=self.mean_error_per_epoch, fmt="none", ecolor="r")
        plt.savefig(os.path.join(path, "mean_reward.png"))

        # Show plots
        if is_show:
            plt.show()