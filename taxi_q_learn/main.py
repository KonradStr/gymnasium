import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def run(episodes, is_training=True, render=False):
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        try:
            f = open('rewards.pkl', 'rb')
            q = pickle.load(f)
            f.close()
        except:
            print("rewards.pkl not found")
            exit(1)


    learning_rate_a = 0.9
    discount_factor_g = 0.9

    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 0.0001  # 1/0.0001 = 10,000
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False  # when >200 actions
        total_rewards = 0

        while (not terminated and not truncated):

            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            total_rewards += reward

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                        reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate_a = 0.0001

        rewards_per_episode[i] = total_rewards

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):t + 1])
    plt.plot(sum_rewards)
    plt.savefig('rewards.png')

    if is_training:
        try:
            f = open("rewards.pkl", "wb")
            pickle.dump(q, f)
            f.close()
        except:
            print("rewards.pkl not found")
            exit(1)


if __name__ == '__main__':
    run(10, is_training=False, render=True)
    # run(15000, is_training=True, render=False)