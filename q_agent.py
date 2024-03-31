from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections import deque

    class Memory(object):

        def __init__(self, max_memory=2000):
            self.cache = deque(maxlen=max_memory)

        def save(self, args):
            self.cache.append(args)

        def empty_cache(self):
            self.__init__()


class Agent(object):

    def __init__(self):
        self.memory: Memory
        pass

    def expand_state_vector(self, state):
        if len(state.shape) == 1 or len(state.shape) == 3:
            return np.expand_dims(state, axis=0)
        else:
            return state

    def remember(self, *args):
        self.memory.save(args)


class QAgent(Agent):
    def __init__(
                self,
                states_size,
                actions_size,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.999,
                gamma=0.95,
                lr=0.8
            ):
        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.Q = self.build_model(states_size, actions_size)

    def build_model(self, states_size, actions_size):
        Q = np.zeros([states_size, actions_size])
        return Q

    def train(self, s, a, r, s_next):
        self.Q[s, a] = (
            self.Q[s, a] +
            self.lr * (
                    r +
                    self.gamma * np.max(self.Q[s_next, a])
                    - self.Q[s, a]
                )
            )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, s):

        q = self.Q[s, :]

        if np.random.rand() > self.epsilon:
            a = np.argmax(q)
        else:
            a = np.random.randint(self.actions_size)

        return a
