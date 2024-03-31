import random

from delivery import DeliveryEnvironment, DeliveryQAgent, run_n_episodes

env = DeliveryEnvironment(n_stops=50, method="distance")
agent = DeliveryQAgent(env.observation_space, env.action_space)

print("RANDOM SCHEDULE\n--------------------")
env.stops = random.sample(list(range(env.n_stops)), env.n_stops)
print(env.format_schedule())
env.reset()

run_n_episodes(env, agent)
print(env.format_schedule())
