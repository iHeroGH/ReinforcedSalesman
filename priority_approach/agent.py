from typing import TYPE_CHECKING

import numpy as np
from numpy import floating
from numpy.typing import NDArray

from environment import DISTANCE_GRANULATION, PRIORITY_GRANULATION

if TYPE_CHECKING:
    from collections import deque

    class Memory(object):

        def __init__(self, max_memory=2000):
            self.cache = deque(maxlen=max_memory)

        def save(self, args):
            self.cache.append(args)

        def empty_cache(self):
            self.__init__()


class Agent:

    def __init__(self):
        self.memory: Memory

    def remember(self, *args):
        self.memory.save(args)


class QAgent(Agent):

    def __init__(
                self,
                states: list[float],
                actions: list[float],
                distances: NDArray[floating],
                priority_information: tuple[list[int], int, int],
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.999,
                gamma=0.95,
                learning_rate=0.8
            ) -> None:
        """Initializes a QAgent"""

        # Input stuff
        self.states = states  # Evenly distributed 0 - max priority
        self.actions = actions  # Evenly distributed 0 - max prio/max distance
        self.distances = distances
        self.priorities, self.max_priority, self.total_priority = (
            priority_information
        )

        # Destruction stuff
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Learning stuff
        self.gamma = gamma
        self.learning_rate = learning_rate

        # Memory stuff
        self.reset_memory()
        self.q_table = self.build_model(len(self.states), len(self.actions))

    def build_model(self, states_size: int, actions_size: int):
        """Builds the base Q-Table"""
        return np.zeros([states_size, actions_size])

    def bellman(
                self,
                state: float,
                action: float,
                reward: float,
                next_state: float
            ) -> None:
        """Applies Bellman's Equation to retrieve a new Q value"""
        state_i = self.states.index(state)
        action_i = self.actions.index(action)
        next_state_i = self.states.index(next_state)

        self.q_table[state_i, action_i] = (
            self.q_table[state_i, action_i] +
            self.learning_rate * (
                    reward +
                    self.gamma * np.max(self.q_table[next_state_i, action_i])
                    - self.q_table[state_i, action_i]
                )
            )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state: float) -> int:
        """
        Retrieve an action in the form of a city index

        Either the best possible for this state, or a random action
        """
        if np.random.rand() > self.epsilon:  # The best action for this state
            state_i = self.states.index(state)
            possible_actions: list[tuple[float, int]] = [
                (self.get_city_action(i), i)
                for i in self.cities_to_visit
            ]

            q_values = self.q_table[
                state_i,
                [self.actions.index(action) for action, _ in possible_actions]
            ]

            best_index = int(np.argmax(q_values))
            try:
                self.actions.index(possible_actions[best_index][0])
                return possible_actions[best_index][1]
            except Exception:
                raise RuntimeError(
                    f"A best value ({self.actions[best_index]}) was found " +
                    "but a corresponding city was not."
                )

        else:  # A random action
            return np.random.choice(self.cities_to_visit)

    def get_current_state(self) -> float:
        return round(
            self.accumulated_priority / self.total_priority,
            PRIORITY_GRANULATION
        )

    def get_next_state(self, city: int) -> float:
        return round(
            (self.accumulated_priority + self.priorities[city])
            /
            self.total_priority,
            PRIORITY_GRANULATION
        )

    def get_city_action(self, city: int) -> float:
        return round(
            self.priorities[city] / self.distances[self.current_city][city],
            DISTANCE_GRANULATION
        )

    def remember_city(self, city: int) -> None:
        self.cities_to_visit.remove(city)
        self.current_city = city
        self.accumulated_priority += self.priorities[city]

    def reset_memory(self):
        self.cities_to_visit = list(range(len(self.distances)))
        self.current_city = -1
        self.accumulated_priority = 0
