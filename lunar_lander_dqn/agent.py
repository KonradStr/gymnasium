from dqn import DQN
from experience_replay import ReplayMemory

import itertools
import os
import yaml
import random
import numpy as np
import gymnasium
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime, timedelta

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

plt.switch_backend('agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Agent:
    def __init__(self, hyperparameter_set):
        with open('../hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameters_set = hyperparameter_set

        self.env_id = hyperparameters['env_id']  # env name
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.stop_on_average_reward = hyperparameters['stop_on_average_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params', {})

        self.loss_fn = nn.MSELoss()  # MSE = Mean Square Error
        self.optimizer = None  # initialized later

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.png')

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gymnasium.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)

        # e.g.: coordinates of lander or angle
        num_states = env.observation_space.shape[0]

        # possible actions:
        #   0 - do nothing
        #   1 - fire left orientation engine
        #   2 - fire main engine
        #   3 - fire right orientation engine
        num_actions = env.action_space.n

        rewards_per_episode = []

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            # creating target network and inits  with policy DQN weights
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            target_dqn.eval()  # set network to evaluation mode

            step_count = 0
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            epsilon_history = []
            best_reward = -float('inf')

            avg_reward = -float('inf')  # average reward tracker
            reward_history = deque(maxlen=100)
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))  # load trained model
            # policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))  # <- if cuda
            policy_dqn.eval()  # set network to evaluation mode

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            truncated = False
            episode_reward = 0.0
            step_in_episode = 0

            while not (terminated or truncated):
                # select action
                if is_training and random.random() < epsilon:  # random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():  # best learned action (highest predicted Q-value)
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, truncated, _ = env.step(action.item())
                episode_reward += reward
                step_in_episode += 1

                # convert to tensors
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                state = new_state

                # learn from experiences
                if is_training and len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)  # optimize based on mini_batch_size
                    step_count += 1

                    # update target dqn with policy dqn based on network_sync_rate hyperparameter
                    if step_count % self.network_sync_rate == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

            # end of episode
            rewards_per_episode.append(episode_reward)

            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)  # update epsilon
                epsilon_history.append(epsilon)

                reward_history.append(episode_reward)  # update reward tracking
                if len(reward_history) >= 100:
                    avg_reward = sum(reward_history) / len(reward_history)

                if episode % 50 == 0:  # logging every 50 episodes avg rewards of last 100 episodes
                    log_message = f"Episode {episode}, Reward: {episode_reward:.1f}, Avg(100): {avg_reward:.1f}, Epsilon: {epsilon:.4f}"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                # save current model if better than previous best
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:.1f}"
                    if best_reward != -float('inf'):
                        log_message += f" ({(episode_reward - best_reward) / abs(best_reward) * 100:+.1f}%)"
                    log_message += f" at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # update diagram every 60 sec
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=60):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # check if expected performance reached
                if avg_reward >= self.stop_on_average_reward:
                    log_message = f"Target performance reached! Avg reward: {avg_reward:.1f}"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    break

        if is_training:  # final diagram save
            self.save_graph(rewards_per_episode, epsilon_history)

        env.close()
        return rewards_per_episode

    def save_graph(self, rewards_per_episode, epsilon_history):
        plt.figure(figsize=(12, 5))

        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])

        plt.subplot(121)
        plt.title('Training Progress')
        plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards (last 100)')
        plt.plot(rewards_per_episode, alpha=0.3, color='blue', label='Rewards')
        plt.plot(mean_rewards, color='red', label='Mean Rewards')
        plt.legend()

        plt.subplot(122)
        plt.title('Exploration Rate')
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.plot(epsilon_history)

        plt.tight_layout()
        plt.savefig(self.GRAPH_FILE)
        plt.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)  # unpack mini_batch

        # convert to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        # compute target Q values
        with torch.no_grad():
            next_q_values = target_dqn(new_states)
            max_next_q = next_q_values.max(dim=1)[0]
            # if episode ended with 'terminations' == True than (1 - terminations) -> 0 else 1
            target_q = rewards + (1 - terminations) * self.discount_factor_g * max_next_q

        # compute current Q values
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, target_q)  # compute loss
        self.optimizer.zero_grad()  # reset gradient from previous step
        loss.backward()  # compute gradients

        # add gradient clipping to prevent exploding gradients for more stability
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1.0)
        self.optimizer.step()


if __name__ == '__main__':
    agent = Agent("lunarlander1")
    # agent.run(is_training=True, render=False)
    agent.run(is_training=False, render=True)
