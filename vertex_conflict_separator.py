from collections import defaultdict
from common import is_gt, Vertex
from pyscipopt import Conshdlr, quicksum, SCIP_RESULT
from typing import Dict


class VertexConflictConstraintHandler(Conshdlr):
    """Constraint handler for creating a single constraint that detects and separates vertex
    conflicts.
    """

    def __init__(self, instance, master, debug: bool = False):
        """Initialise the constraint handler for vertex conflict constraints.

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

        # Attach the separator.
        self.master.model.includeConshdlr(
            self, "vertex_conflict", "Vertex conflict", chckpriority=10, enfopriority=10
        )

        # Create a constraint. In SCIP, the term "constraint" is used to mean something between a
        # global constraint in constraint programming and a separator in integer programming. The
        # terminology doesn't quite align with a separator, which is what we need.
        constraint = self.master.model.createCons(self, "vertex_conflict")
        self.master.model.addPyCons(constraint)

    def log(self, *args, **kwargs):
        """Conditional logging controlled by ``self.debug``."""

        if self.debug:
            print(*args, **kwargs)

    def compute_usage(self, solution=None) -> Dict[Vertex, float]:
        """Project the current solution onto vertices.

        Scans all path variables in the master problem and accumulates the amount of each vertex
        used.

        Args:
            solution: Optional SCIP solution for evaluating the variable values.
                      If ``None``, the current LP solution is used.
        """
        # Calculate the value of each vertex from the value of the paths.
        # This calculation is called "projection".
        usage: Dict[Vertex, float] = defaultdict(lambda: 0.0)
        for _, paths_vars in enumerate(self.master.paths):
            for path, var in paths_vars.items():
                val = self.master.model.getSolVal(solution, var)
                if is_gt(val, 0.0):
                    for vertex in path.vertices():
                        usage[vertex] += val
        return usage

    def conscheck(
        self,
        constraints,
        solution,
        check_integrality,
        check_lp_rows,
        print_reason,
        completely,
        **results,
    ):
        """Check whether a given solution is feasible with respect to vertex conflicts.

        Args:
            solution: Optional SCIP solution for evaluating the variable values.
                      If ``None``, the current LP solution is used.
        """

        # Print solution.
        # self.master.print_sol(solution)

        # The solution is infeasible if any has proportion > 1.
        usage = self.compute_usage(solution)
        for vertex, val in usage.items():
            if is_gt(val, 1.0):
                return {"result": SCIP_RESULT.INFEASIBLE}
        return {"result": SCIP_RESULT.FEASIBLE}

    def consenfolp(self, constraints, n_useful_conss, sol_infeasible):
        """Separate violated vertex conflict constraints."""

        # Print solution.
        # self.master.print_sol()

        # Create constraints preventing vertex conflicts.
        added = False
        usage = self.compute_usage()
        for vertex, val in usage.items():
            if is_gt(val, 1.0):

                # Get the variables involved in the cut.
                variables = []
                # [TASK3a] The check above asserts that `vertex` is used with proportion > 1. Hence
                # we know there is a vertex conflict at `vertex`. Create a constraint to prevent
                # this vertex conflict in the future.
                # 1. Check if `path` uses `vertex` by executing `path.uses_vertex(vertex)`.
                # 2. If so, append the variable `var` to the list `variables` of variables appearing
                #    in the constraint.
                for agent, agent_path_vars in enumerate(self.master.paths):
                    for path, var in agent_path_vars.items():
                        pass

                # Create a cut. The check for emptiness is not required in real code. It's only
                # there because the code above is waiting for the user to implement.
                if len(variables) > 0:
                    name = f"vertex_conflict{vertex}"
                    constraint = self.master.model.addCons(
                        quicksum(variables) <= 1.0,
                        local=False,
                        separate=False,
                        modifiable=True,
                        enforce=True,
                        check=True,
                        name=name,
                    )
                    self.master.vertex_conflict_constraints[vertex] = constraint
                    added = True

        if added:
            return {"result": SCIP_RESULT.CONSADDED}
        else:
            return {"result": SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        pass
