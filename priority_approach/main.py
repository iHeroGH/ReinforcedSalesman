import random
from time import time

import matplotlib.pyplot as plt

from agent import QAgent
from environment import DeliveryEnv


def run_episode(
            env: DeliveryEnv,
            qag: QAgent,
            verbose: bool = False
        ) -> tuple[DeliveryEnv, QAgent, float]:

    city = env.reset()
    qag.reset_memory()

    episode_reward: float = 0.0

    i = 0
    while i < env.n_stops:

        # Remember which city we're at
        qag.remember_city(city)

        # Choose and take an action
        next_city = qag.act(qag.get_current_state())
        next_city, reward, done = env.step(next_city)

        # Some logging
        if verbose:
            print(next_city, reward, done)

        # Train the agent (update the Q-Table)
        qag.bellman(
            qag.get_current_state(),
            qag.get_city_action(next_city),
            reward,
            qag.get_next_state(next_city)
        )

        # Update the cache
        episode_reward += reward
        city = next_city

        # Check if we're done
        i += 1
        if done:
            break

    return env, qag, episode_reward


def run_n_episodes(
            env: DeliveryEnv,
            qag: QAgent,
            n_episodes=250_000,
            render_every=10_000
        ) -> tuple[DeliveryEnv, QAgent]:

    rewards = []

    for i in range(n_episodes):

        env, qag, reward = run_episode(env, qag)
        rewards.append(reward)

        if not i % render_every:
            print(f"Rendering Step {i}")
            # env.show()

    # Styling
    plt.style.use("dark_background")

    # Setting up
    plt.figure(figsize=(15, 3))
    plt.title("Reward Over Training")
    plt.plot(rewards)
    plt.show()

    return env, qag


env = DeliveryEnv(n_stops=50)

print("RANDOM SCHEDULE\n--------------------")
env.schedule = random.sample(list(range(env.n_stops)), env.n_stops)
print(env.format_schedule())
env.reset()

s = time()
print("\nTRAINED SCHEDULE\n--------------------")
qag = QAgent(
    env.state_space,
    env.action_space,
    env.distance_matrix,
    (env.priorities, env.max_priority, env.total_priority)
)

run_n_episodes(env, qag)
print(env.format_schedule())
print(f"Took {time() - s}")

s = time()
print("\nCONSISTENCY TEST\n--------------------")
qag = QAgent(
    env.state_space,
    env.action_space,
    env.distance_matrix,
    (env.priorities, env.max_priority, env.total_priority)
)

run_n_episodes(env, qag)
print(env.format_schedule())
print(f"Took {time() - s}")
