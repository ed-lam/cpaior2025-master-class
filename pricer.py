from collections import defaultdict
from common import Agent, is_ge, Path, Time, Vertex, XY
from dataclasses import dataclass
from numpy import inf
from typing import Dict, Optional, Tuple
import heapq
import pyscipopt as scip


class PriorityQueue:
    """Tiny wrapper around `heapq` to act as a priority queue.

    This wrapper stores a counter to guarantee a total ordering of pushed
    items (so that two items with equal priority do not require comparing
    the items themselves).
    """

    def __init__(self):
        """Create an empty priority queue.

        Attributes:
            heap: Internal heap list storing tuples (priority, counter, item).
            counter: Monotonic counter used to break ties between equal priorities.
        """

        self.heap = []
        self.counter = 0

    def push(self, priority, item):
        """Push ``item`` with given ``priority`` into the queue.

        Args:
            priority: Numeric priority (lower is popped earlier).
            item: The payload to store.
        """

        self.counter += 1
        heapq.heappush(self.heap, (priority, self.counter, item))

    def pop(self):
        """Pop and return the item with the smallest priority."""

        return heapq.heappop(self.heap)[-1]

    def __len__(self) -> int:
        """Return the number of items currently in the queue."""

        return len(self.heap)


@dataclass(frozen=True)
class State:
    """Search state used by A*.

    Attributes:
        f: Estimated total cost (g + h).
        g: Cost from start to this state.
        h: Heuristic estimate from this state to a goal.
        xy: Location of this state.
        t:  Timestep of this state.
        parent: Optional parent State used to reconstruct the path.
    """

    f: float
    g: float
    h: int
    xy: XY
    t: Time
    parent: Optional["State"] = None

    def __str__(self) -> str:
        return f"f={self.f:.3f} g={self.g:.3f} h={self.h} xy={self.xy} t={self.t}"


class AStarPricer(scip.Pricer):
    """A* algorithm for solving the pricing problem."""

    def __init__(self, instance, master=None, debug: bool = False):
        """Initialise the pricer and register it with the solver.

        Args:
            instance: The MAPF instance.
            master: The master problem.
            debug: If ``True``, enable verbose logging.
        """

        super().__init__()

        # Store the instance.
        self.instance = instance

        # Store other parts of the solver.
        self.master = master

        # Attach the pricer.
        if master:
            self.master.model.includePricer(self, "astar", "A*", delay=True)

        # Store verbosity.
        self.debug = debug

        # Initialise the pricer with zero costs.
        self.cost_offset = 0.0
        self.vertex_penalties = defaultdict(lambda: 0.0)

    def pricerinit(self):
        """Initialise the pricer."""

        # Transform the constraints in the master problem.
        # TODO: transform_constraints() should really be called from the problem class but we don't
        # have this.
        self.master.transform_constraints()

    def log(self, *args, **kwargs):
        """Conditional logging controlled by ``self.debug``."""

        if self.debug:
            print(*args, **kwargs)

    def neighbours(self, current_xy: XY):
        """Yield neighbour coordinates (wait, north, south, east, west) that are free.

        Args:
            current_xy: Current coordinate.
        """

        # Wait.
        next_xy = XY(current_xy.x, current_xy.y)
        if self.instance.is_free(next_xy):
            yield next_xy

        # Move east.
        next_xy = XY(current_xy.x + 1, current_xy.y)
        if self.instance.is_free(next_xy):
            yield next_xy

        # Move west.
        next_xy = XY(current_xy.x - 1, current_xy.y)
        if self.instance.is_free(next_xy):
            yield next_xy

        # Move south.
        next_xy = XY(current_xy.x, current_xy.y + 1)
        if self.instance.is_free(next_xy):
            yield next_xy

        # Move north.
        next_xy = XY(current_xy.x, current_xy.y - 1)
        if self.instance.is_free(next_xy):
            yield next_xy

    @staticmethod
    def manhattan(xy1: XY, xy2: XY) -> int:
        """Return the Manhattan distance between two coordinates.

        Args:
            xy1: First coordinate.
            xy2: Second coordinate.
        """

        return abs(xy1.x - xy2.x) + abs(xy1.y - xy2.y)

    def generate_start(self, mode: str, start: XY, goal: XY):
        """Create and push the initial state.

        Args:
            mode: ``redcost`` or ``farkas`` for f-value and key calculations.
            start: Start coordinate.
            goal: Goal coordinate.
        """

        # Compute the distance to the goal.
        h = self.manhattan(start, goal)

        # Compute the costs.
        g = 0.0
        t = 0
        if mode == "redcost":
            f = g + h
            key = (f, h)
        else:
            f = g
            key = (g, t + h, h)

        # Push into the open set.
        state = State(f, g, h, start, t)
        self.push(key, state)

    def edge_cost(self, current: Vertex, next: Vertex):
        """Compute the cost of an edge.

        Args:
            current: Origin of the edge.
            next: Destination of the edge.
        """

        # [TASK1] Every edge has cost 1.0. Return this edge cost.
        raise Exception("Not yet implemented")

    def vertex_penalty(self, next: Vertex):
        """Retrieve the penalty for visiting a vertex.

        Args:
            next: The vertex visited.
        """

        # [TASK3c] Return the penalty for contesting a vertex in use by another agent. Look up the
        # `vertex` in the dictionary `self.vertex_penalties` and return that value.
        return 0.0

    def edge_reduced_cost(self, mode: str, current: Vertex, next: Vertex):
        """Compute the reduced cost or Farkas cost of an edge.

        Args:
            mode: ``redcost`` or ``farkas`` for computing the reduced cost or Farkas cost.
            current: Origin of the edge.
            next: Destination of the edge.
        """

        if mode == "redcost":
            return self.edge_cost(current, next) + self.vertex_penalty(next)
        else:
            return self.vertex_penalty(next)

    def generate_next(self, mode: str, current: State, next_xy: XY, goal: XY):
        """Generate a successor state.

        Args:
            mode: ``redcost`` or ``farkas`` for computing the reduced cost or Farkas cost.
            current: Current State being expanded.
            next_xy: Coordinate of the successor.
            goal: Goal coordinate.
        """

        # Compute the distance to-go.
        h = self.manhattan(next_xy, goal)

        # Compute the costs.
        t = current.t + 1
        g = current.g + self.edge_reduced_cost(
            mode, Vertex(current.xy, current.t), Vertex(next_xy, t)
        )
        if mode == "redcost":
            # Choose the state with the lowest cost estimate (min f), then tie-break with states
            # closest to the goal location (min h).
            f = g + h
            key = (f, h)
        else:
            # Since actions have 0 cost in Farkas pricing, the Manhattan heuristic is not valid.
            # Just have to use h = 0 everywhere. Hence, f = g, falling back to Dijkstra's algorithm.
            # Choose the state with lowest f (but = g), then tie-break on lowest estimated arrival
            # time (t + h) and then distance to the goal location (h).
            f = g
            key = (g, t + h, h)

        # Push into the open set.
        next = State(f, g, h, next_xy, t, current)
        self.push(key, next)

    def push(self, key: Tuple, state: State):
        """Push a state into the open set if it improves on the closed cost.

        Args:
            key: Ordering key for the priority queue.
            state: State to push.
        """

        vertex = Vertex(state.xy, state.t)
        if state.g < self.closed[vertex]:
            self.open.push(key, state)
            self.closed[vertex] = state.g
            # TODO: The heap provided by Python does not support delete and decrease key. So we just
            # push the lower-cost state into the queue. The dominated state will be popped
            # eventually and ignored.
            self.log(f"        [PUSH] {state} key={key}")

    def add_path(self, mode: str, agent: Agent, goal_state: State):
        """Reconstruct a path from ``goal_state`` parent links and add it.

        Args:
            mode: ``redcost`` or ``farkas`` for debugging purposes.
            agent: Agent number for which the path was found.
            goal_state: State at the goal from which to backtrack.
        """

        # Chase the parent pointers to get the path.
        vertices = []
        current_state = goal_state
        while True:
            i = current_state.xy
            vertices.append(i)
            current_state = current_state.parent
            if not current_state:
                break
        vertices.reverse()

        # Create a path object.
        path = Path(agent, vertices)
        path_str = ",".join(
            map(lambda data: f"({data[1]},{data[0]})", enumerate(path.xys))
        )
        self.log(f"        [FOUND_PATH] agent={agent} path={path_str}")

        # Add a column.
        if self.master:
            self.master.add_var(path, mode, goal_state.g + self.cost_offset)

    def solve(
        self,
        mode: str,
        agent: Agent,
        cost_ub: float = float("inf"),
        cost_offset: float = 0.0,
        vertex_penalties: Dict[Vertex, float] = defaultdict(lambda: 0.0),
    ):
        """Solve the single-agent shortest path pricing problem using A*.

        Args:
            mode: ``redcost`` or ``farkas`` for cost calculations.
            agent: Agent index to price for.
            cost_ub: Upper bound on acceptable reduced cost. The search will stop when
                current.f + cost_offset >= cost_ub.
            cost_offset: Initial g-value/offset added to reduced costs.
            vertex_penalties: Mapping from ``Vertex`` to penalty (float) derived from duals.
        """

        # Get the agent's start and goal position.
        start = self.instance.agents[agent].start
        goal = self.instance.agents[agent].goal
        self.log(f"[{mode.upper()}] Agent {agent} start={start} goal={goal}")

        # Initialise internal state.
        self.cost_offset = cost_offset
        self.vertex_penalties = vertex_penalties
        self.open = PriorityQueue()
        self.closed = defaultdict(lambda: inf)

        # Main loop.
        self.generate_start(mode, start, goal)
        while len(self.open) > 0:

            # Get the best state from the open set.
            current = self.open.pop()
            self.log(f"    [POP] {current}")

            # Stop if no path has negative reduced cost.
            if is_ge(current.f + self.cost_offset, cost_ub):
                return

            # Stop if a path with negative reduced cost is found.
            if current.xy == goal:
                self.log(f"    [GOAL] {current}")
                self.add_path(mode, agent, current)
                return current.g + self.cost_offset

            # Expand to neighbours.
            for next_pos in self.neighbours(current.xy):
                self.generate_next(mode, current, next_pos, goal)

        # All paths have non-negative reduced cost.
        return

    def price(self, mode: str):
        """Get the dual multipliers from the master problem and call A* on each agent to generate
        columns.

        Args:
            mode: ``redcost`` or ``farkas`` for selecting which dual multiplier to query.
        """

        # Get the master problem.
        master = self.master
        model = master.model

        # Define a convenience function to get the correct dual multiplier depending on the mode
        # (Farkas or reduced cost pricing).
        get_dual_multiplier = (
            model.getDualsolLinear if mode == "redcost" else model.getDualfarkasLinear
        )

        # Impose dual multipliers from vertex conflict conflict constraints.
        global_vertex_penalties = defaultdict(lambda: 0.0)
        for vertex, constraint in master.vertex_conflict_constraints.items():
            # [TASK3b] Input the dual multiplier of the vertex conflict constraints into the A*
            # algorithm.
            # 1. Check if the constraint is active (not disabled) by calling the isActive() method.
            # 2. If it's active, call the convenience function get_dual_multiplier() to get either
            #    the reduced cost or Farkas dual multiplier.
            # 3. Add the negative dual multiplier to the value of `vertex` in the dictionary
            #    `global_vertex_penalties`.
            pass

        # Run A* for each agent.
        if mode == "redcost":
            master_lb = master.model.getLPObjVal()
        for agent in range(len(self.instance.agents)):

            # Get dual multiplier from agent constraint.
            cost_offset = 0.0
            constraint = master.agent_constraints[agent]
            # [TASK2] Input the dual multiplier of the agent constraint into the A* algorithm.
            # 1. Check if the constraint is active (not disabled) by calling the isActive() method.
            # 2. If it's active, call the convenience function get_dual_multiplier() to get either
            #    the reduced cost or Farkas dual multiplier.
            # 3. Set cost_offset to the negative dual multiplier.
            pass

            # Impose dual multipliers from branching constraints.
            # [TASK4b] Read this function. Now that you've created a constraint imposing the
            # branching decision, you need to handle its dual variable. Treat it just like the
            # standard vertex conflicts but it is only for one agent, not all agents. Therefore,
            # we use an agent-specific copy of the data structure storing the dual multipliers.
            agent_vertex_penalties = global_vertex_penalties.copy()
            for vertex, constraint in master.vertex_decisions[agent]:
                if constraint.isActive():
                    dual = get_dual_multiplier(constraint)
                    agent_vertex_penalties[vertex] -= dual

            # Solve.
            reduced_cost = self.solve(
                mode, agent, 0.0, cost_offset, agent_vertex_penalties
            )

            # Add term to lower bound calculation.
            if mode == "redcost" and reduced_cost:
                master_lb += reduced_cost

        # Return 'SUCCESS' if either at least one column was added, or it proved no columns with
        # negative reduced cost exists. # Otherwise, return 'DIDNOTRUN', including when a column was
        # not found due to time out. The current implementation does not use the time limit, so it
        # always solves to completion.
        output = {"result": scip.SCIP_RESULT.SUCCESS}
        if mode == "redcost":
            output["lowerbound"] = master_lb
        return output

    def pricerredcost(self):
        """Reduced cost pricing function when the master problem is feasible."""

        return self.price("redcost")

    def pricerfarkas(self):
        """Farkas cost pricing function when the master problem is infeasible."""

        return self.price("farkas")
