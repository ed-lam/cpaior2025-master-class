from common import XY
from dataclasses import dataclass
from typing import List, Tuple
import os


@dataclass
class Agent:
    """Stores start and goal coordinates of an agent.

    Attributes:
        start: Start coordinates.
        goal: Goal coordinates.
    """

    start: XY
    goal: XY


@dataclass
class Instance:
    """Holds a MAPF instance, comprising of the dimensions, grid and list of agents.

    Attributes:
        width: Map width in cells (number of columns)
        height: Map height in cells (number of rows)
        grid: List of strings; each string is a row of characters
        agents: List of Agent objects
    """

    width: int
    height: int
    grid: List[str]  # each row string of '.' (free) and '@' (blocked)
    agents: List[Agent]

    def is_free(self, xy: XY) -> bool:
        """Checks if a location is traversable.

        Args:
            xy: The location.
        """

        return (
            0 <= xy.x < self.width
            and 0 <= xy.y < self.height
            and self.grid[xy.y][xy.x] == "."
        )


def read_map(map_path: str) -> Tuple[int, int, List[str]]:
    """Read a MovingAI .map file.

    Args:
        map_path: File path of the map file.
    """

    # Read all lines.
    with open(map_path, "r") as f:
        lines = [ln.strip("\n") for ln in f.readlines()]

    # Read map size.
    assert lines[0].lower().startswith("type"), "Expect MovingAI header: type octile"
    height = int(lines[1].split()[-1])
    width = int(lines[2].split()[-1])

    # Read map.
    assert lines[3].lower().startswith("map"), "Expect 'map' header before grid"
    grid = lines[4 : 4 + height]

    # Check
    assert len(grid) == height, f"Map has {len(grid)} rows but expected {height}"
    for i, row in enumerate(grid):
        assert (
            len(row) == width
        ), f"Map row {i} has width {len(row)} but expected {width}"

    return width, height, grid


def load_instance(scen_path: str, num_agents: int) -> Instance:
    """Read a MovingAI .scen file.

    Args:
        scen_path: File path of the scenario file.
        num_agents: Number of agents to read.
    """

    # Read the file and skip the first (header) line.
    with open(scen_path, "r") as f:
        lines = [line.strip("\n") for line in f.readlines()]

    # Each subsequent non-empty line describes one agent.
    map_path = None
    agents: List[Agent] = []
    for line in lines[1:]:

        # Skip if not valid.
        if not line:
            continue

        # Read th map file.
        parts = line.split()
        assert map_path is None or map_path == parts[1]
        map_path = parts[1]

        # Read the agent data.
        start_x, start_y, goal_x, goal_y = map(
            int, [parts[4], parts[5], parts[6], parts[7]]
        )
        agents.append(Agent(XY(start_x, start_y), XY(goal_x, goal_y)))
        if num_agents and len(agents) >= num_agents:
            break

    # Resolve directory of the map file.
    scen_dir = os.path.dirname(os.path.abspath(scen_path))
    map_path = os.path.join(scen_dir, map_path)

    # Read map.
    width, height, grid = read_map(map_path)

    # Create the instance object.
    instance = Instance(width, height, grid, agents)
    return instance
