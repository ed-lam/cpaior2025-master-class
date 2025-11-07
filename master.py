from common import is_eq, is_gt, Path
from instance import Instance
import pyscipopt as scip


class Master:
    """
    Master problem
    """

    def __init__(self, instance: Instance, time_limit: float = None):
        """Create the master problem and create a SCIP model.

        Args:
            instance: The MAPF instance.
            time_limit: Optional time limit in seconds.
        """

        # Store the instance.
        self.instance = instance

        # Create a SCIP model.
        self.model = scip.Model("Minimal BCP-MAPF")

        # Set number of threads.
        self.model.setParam("lp/threads", 1)
        self.model.setParam("parallel/maxnthreads", 1)

        # Set time limit.
        if time_limit is not None:
            self.model.setRealParam("limits/time", time_limit)

        # Turn off presolving. Nothing to presolve because the model is empty (no columns).
        self.model.setPresolve(scip.SCIP_PARAMSETTING.OFF)

        # Turn off cutting planes. Cannot use cuts (e.g., Gomory) because they induce
        # dual variables that our pricer does not know how to handle.
        self.model.setSeparating(scip.SCIP_PARAMSETTING.OFF)

        # Turn off propagation. Pricer needs the exact structure we give it.
        self.model.setIntParam("propagating/rootredcost/freq", -1)

        # Make it more verbose.
        self.model.setIntParam("display/freq", 1)

        # Tell SCIP that the objective always takes integer values to enable it to cutoff nodes by
        # rounding the lower bound up.
        self.model.setObjIntegral()

        # Add constraint requiring every agent to use one path.
        self.agent_constraints = [
            self.model.addCons(
                scip.quicksum([]) == 1.0,
                separate=False,
                modifiable=True,
                name=f"agent_constraint[{a}]",
            )
            for a in range(len(instance.agents))
        ]

        # Prepare somewhere to store vertex conflict constraints.
        self.vertex_conflict_constraints = dict()

        # Prepare somewhere to store vertex branching decisions for each agent.
        self.vertex_decisions = {a: [] for a in range(len(instance.agents))}

        # Prepare somewhere to store the paths of each agent.
        self.paths = [dict() for _ in range(len(instance.agents))]

    def transform_constraints(self):
        """Retrieve transformed constraint objects from SCIP."""

        # Overwrite the constraints from the the transformed problem. SCIP does a transformation
        # before solving but this changes nothing when the model is empty (common in column
        # generation).
        self.agent_constraints = [
            self.model.getTransformedCons(cons) for cons in self.agent_constraints
        ]

    def add_var(self, path: Path, mode: str, reduced_cost: float):
        """Add a path variable and include it in all relevant constraints.

        Args:
            path: The ``Path`` object to add as a column/variable.
            mode: ``redcost`` or ``farkas`` determining which duals to query.
            reduced_cost: Reduced cost value computed by the pricer for debugging purposes.
        """

        # Define a convenience function to get the correct dual multiplier depending on the mode
        # (Farkas or reduced cost pricing). This is only used for debugging here.
        get_dual = (
            self.model.getDualsolLinear
            if mode == "redcost"
            else self.model.getDualfarkasLinear
        )

        # Get the path cost.
        path_len = len(path.xys)
        path_cost = path_len - 1

        # Create the variable.
        vertices_str = ",".join(map(lambda i: f"({i.x},{i.y})", path.xys))
        name = f"path[{path.agent},{vertices_str}]"
        var = self.model.addVar(
            vtype="I", lb=0, ub=None, obj=path_cost, name=name, pricedVar=True
        )

        # Initialise checking value for debugging purposes.
        check_reduced_cost = path_cost if mode == "redcost" else 0.0

        # Add the variable to the agent constraint.
        constraint = self.agent_constraints[path.agent]
        self.model.addConsCoeff(constraint, var, 1.0)
        check_reduced_cost -= get_dual(constraint)

        # Add the variable to vertex conflict constraints.
        # [TASK3d] Read over this code. It adds an entry/coefficient to the column of new paths if
        # the path uses a vertex of a vertex conflict constraint.
        for vertex, constraint in self.vertex_conflict_constraints.items():
            if path.uses_vertex(vertex):
                self.model.addConsCoeff(constraint, var, 1.0)
                check_reduced_cost -= get_dual(constraint)

        # Add the variable to vertex decision constraints.
        for vertex, constraint in self.vertex_decisions[path.agent]:
            if path.uses_vertex(vertex):
                self.model.addConsCoeff(constraint, var, 1.0)
                check_reduced_cost -= get_dual(constraint)

        # Check that the computed reduced cost matches the actual reduced cost.
        assert is_eq(check_reduced_cost, reduced_cost), (
            f"Your computed reduced cost {reduced_cost} does not match the actual reduced cost "
            f"{check_reduced_cost}"
        )

        # Check if the path already exists.
        if path in self.paths[path.agent]:
            existing_var = self.paths[path.agent][path]
            print(
                f"Duplicate path {vertices_str}, original in LP: {existing_var.isInLP()}"
            )

        # Store the path.
        self.paths[path.agent][path] = var

    def time_remaining(self) -> float:
        """Get the time remaining in seconds."""

        time_limit = self.model.getParam("limits/time")
        elapsed_time = self.model.getSolvingTime()
        time_remaining = time_limit - elapsed_time
        return time_remaining

    def print_sol(self, solution=None):
        """Print a solution.

        Args:
            solution: Optional SCIP solution for evaluating the variable values.
                      If ``None``, the current LP solution is used.
        """

        # Print the objective value.
        obj = self.model.getSolObjVal(solution)
        print(f"Total cost {obj:.0f}")

        # Print the paths.
        for agent, paths_vars in enumerate(self.paths):
            for path, var in paths_vars.items():
                val = self.model.getSolVal(solution, var)
                if is_gt(val, 0.0):
                    path_str = ", ".join(
                        map(lambda data: f"({data[1]},{data[0]})", enumerate(path.xys))
                    )
                    print(f"Agent {agent:3d}, val {val:6.2f}, path {path_str}")

    def write(self):
        """Write the model to file."""

        self.model.writeProblem("master.lp", trans=True)
