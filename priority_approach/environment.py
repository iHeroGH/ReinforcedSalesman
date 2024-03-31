import matplotlib.pyplot as plt
import numpy as np
from numpy import floating
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

PRIORITY_GRANULATION = 1  # Round to n decimal places for priority-related ops
DISTANCE_GRANULATION = 2  # Round to n decimal places for distance-related ops


class DeliveryEnv:

    def __init__(
                self,
                n_stops=10,
                box_dim=10,
                min_gen_prio=1,
                max_gen_prio=10
            ) -> None:
        """
        Initialize a delivery environment

        Parameters
        ----------
        n_stops : int, default=10
            The number of stops (cities) to create
        box_dim : int, default=5
            The maximum x or y coordinate of the city
        min_gen_prio : int, default=1
            The minimum priority that could be generated for a city
        max__gen_prio : int, default=10
            The maximum priority that could be generated for a city
        """
        print(f"Initializing env with {n_stops} stops.")

        self.n_stops = n_stops
        self.box_dim = box_dim
        self.reset()

        self.min_gen_prio = min_gen_prio
        self.max_gen_prio = max_gen_prio

        self._generate_stops()
        self._generate_priorities()
        self.distance_matrix: NDArray[floating] = self._calculate_distances()

        prio_step = 10 ** -PRIORITY_GRANULATION
        dist_step = 10 ** -DISTANCE_GRANULATION

        # Normalized priority from 0 - 1 with a granularly static step
        self.state_space: list[float] = np.round(
            np.arange(
                0,
                1 + prio_step,
                prio_step
            ),
            decimals=PRIORITY_GRANULATION
        ).tolist()

        # Normalized priority from 0 - max prio / max distance
        # with a granularly static step
        self.action_space: list[float] = np.round(
            np.arange(
                0,
                (self.max_gen_prio / dist_step) + dist_step,
                dist_step
            ),
            decimals=DISTANCE_GRANULATION
        ).tolist()

    def _generate_stops(self) -> None:
        """
        Generates two lists, a list of all the x coordinates, and a list
        of all the y coordinates
        """
        all_coords = np.random.rand(self.n_stops, 2) * self.box_dim
        self.all_x = all_coords[:, 0]
        self.all_y = all_coords[:, 1]

    def _generate_priorities(self) -> None:
        self.priorities = [
            np.random.randint(self.min_gen_prio, self.max_gen_prio + 1)
            for _ in range(self.n_stops)
        ]
        self.max_priority = max(self.priorities)
        self.total_priority = sum(self.priorities)

    def _calculate_distances(self) -> NDArray[floating]:
        """
        Returns the distance between each of the two points in all_x and all_y
        """
        all_coords = np.column_stack([self.all_x, self.all_y])
        return cdist(all_coords, all_coords)

    def get_reward(self, city: int, next_city: int) -> float:
        return 1/self.distance_matrix[city, next_city]

    def reset(self) -> int:
        """
        Reset the environment's schedule memory and returns a random starting
        city
        """
        self.schedule: list[int] = []
        self.schedule.append(city := np.random.randint(self.n_stops))
        return city

    def step(self, destination: int) -> tuple[int, float, bool]:
        """
        Perform a single step of learning

        Parameters
        ----------
        destination : int
            The index of the city to visit

        Returns
        -------
        int
            The index of the next city
        float
            The reward achieved by visiting this city
        bool
            Whether or not we are now done
        """

        current_city = self.schedule[-1]
        next_city = destination

        reward = self.get_reward(current_city, next_city)

        self.schedule.append(next_city)
        done = len(self.schedule) == self.n_stops

        return next_city, reward, done

    def format_schedule(self) -> str:
        if len(self.schedule) <= 1:
            raise ValueError("No valid path to format. Must contain 2 items")

        output = ""
        cost = 0
        priority = 0
        prev = 0
        for next in range(1, len(self.schedule)):
            prev_city = int(self.schedule[prev])
            next_city = int(self.schedule[next])

            cost += (curr_cost := self.distance_matrix[prev_city, next_city])
            priority += self.priorities[next_city]
            cost_str = f"--({curr_cost})>>"

            if not prev:
                output += f"{prev_city} {cost_str} {next_city}"
            else:
                output += f" {cost_str} {next_city}"

            prev = next

        return output + f"\nCost: {cost}\nPriority: {priority}"

    def show(self):

        # Styling
        plt.style.use("dark_background")

        # Setting up
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.set_title("Stops")

        # Plotting
        ax.scatter(self.all_x, self.all_y, c="red", s=50)
        plt.xticks([])
        plt.yticks([])

        # Saving/Displaying
        ...
        plt.show()
