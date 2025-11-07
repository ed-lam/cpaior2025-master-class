from collections import defaultdict
from common import Agent, is_gt, Vertex
from pyscipopt import Branchrule, quicksum, SCIP_RESULT
from typing import Dict, Set, Tuple
import numpy as np


class VertexBranching(Branchrule):
    """Down-only branching on a vertex."""

    def __init__(self, instance, master, debug: bool = False):
        """Initialise the branching rule and register it with the solver.

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

        # Store verbosity.
        self.debug = debug

        # Attach the branching rule.
        self.master.model.includeBranchrule(
            self,
            "vertex_branch",
            "Vertex branching",
            priority=30000000,
            maxdepth=-1,
            maxbounddist=1.0,
        )

    def log(self, *args, **kwargs):
        """Conditional logging controlled by ``self.debug``."""

        if self.debug:
            print(*args, **kwargs)

    def compute_usage(self, solution=None) -> Dict[Vertex, Set[Agent]]:
        """Project the current solution over paths onto vertices.

        Scans all path variables in the master problem and accumulates the amount of each vertex
        used by each agent. Only vertices used by at least two agents are returned.

        Args:
            solution: Optional SCIP solution handle to evaluate variables.
                      If ``None``, the current LP solution is used.
        """

        usage: Dict[Vertex, Set[Agent]] = defaultdict(lambda: set())
        for _, paths_vars in enumerate(self.master.paths):
            for path, var in paths_vars.items():
                val = self.master.model.getSolVal(solution, var)
                if is_gt(val, 0.0):
                    for vertex in path.vertices():
                        usage[vertex].add(path.agent)
        usage = {vertex: agents for vertex, agents in usage.items() if len(agents) >= 2}
        return usage

    def decide(self, usage: Dict[Vertex, Set[Agent]]) -> Tuple[Vertex, Agent, Agent]:
        """Randomly choose a vertex and two distinct agents for branching.

        Args:
            usage: Mapping from Vertex to the agents using it.
        """

        if len(usage) == 0:
            return

        usage = [(vertex, agents) for vertex, agents in usage.items()]
        rng = np.random.default_rng(123)
        vertex, agents = rng.choice(usage)
        a1, a2 = rng.choice(list(agents), size=2, replace=False)
        return vertex, a1, a2

    def create_child(self, vertex: Vertex, agent: Agent):
        """Create a child node that forbids an agent from using a vertex.

        Args:
            vertex: The vertex to forbid.
            agent: The agent id whose paths should be blocked in the child node.
        """

        # [TASK4a] Read this function. It takes a decision, consisting of a vertex and an agent,
        # creates a child node and prevents the agent from using that vertex in the node and the
        # subtree below. The branching decision is realized by adding a local constraint (method 1
        # from the slides).

        # Create a child node.
        model = self.master.model
        child = model.createChild(1.0, model.getCurrentNode().getLowerbound())

        # Get the variables involved in the cut.
        variables = []
        for path, var in self.master.paths[agent].items():
            if path.uses_vertex(vertex):
                variables.append(var)

        # Create a local cut.
        name = f"vertex_decision({agent},{vertex})"
        constraint = model.createConsFromExpr(
            quicksum(variables) <= 0.0,
            local=True,
            separate=False,
            modifiable=True,
            enforce=True,
            check=False,
            name=name,
        )
        model.addConsNode(child, constraint)

        # Store the decision.
        self.master.vertex_decisions[agent].append((vertex, constraint))

    def branchexeclp(self, allowaddcons):
        """SCIP callback to perform branching at an LP node."""

        # Print solution.
        # self.master.print_sol()

        # Compute usage of each vertex.
        usage = self.compute_usage()

        # Choose a vertex and two agents using that vertex.
        decision = self.decide(usage)
        assert (
            decision is not None
        ), "Failed to find a decision but the node is fractional"
        vertex, a1, a2 = decision

        # Create two children nodes.
        self.log(f"[BRANCH] vertex {vertex} between agents {a1} and {a2}")
        self.create_child(vertex, a1)
        self.create_child(vertex, a2)

        # Indicate that children nodes are created.
        return {"result": SCIP_RESULT.BRANCHED}

    def branchexecps(self, allowaddcons):
        """SCIP callback to perform branching on a pseudosolution. This sometimes occurs immediately
        before timeout but has no real benefit.
        """

        # Stop.
        print("WARNING: attempting to branch on pseudosolution")
