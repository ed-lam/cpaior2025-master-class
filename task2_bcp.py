from common import is_gt
from instance import load_instance
from master import Master
from pricer import AStarPricer
from vertex_branching import VertexBranching
from vertex_conflict_separator import VertexConflictConstraintHandler
import argparse
class BranchAndCutAndPriceSolver:
    """
    Branch-and-cut-and-price solver for the simplified multi-agent pathfinding problem where agents
    disappear upon reaching their goals.
    """

    def __init__(self, instance, time_limit=None):
        """
        Create the components of the solver.

        Args:
            instance: The MAPF instance.
            time_limit: Optional time limit in seconds for solving.
        """

        self.master = Master(instance, time_limit)
        self.vertex_branching = VertexBranching(instance, self.master)
        self.pricer = AStarPricer(instance, self.master)
        self.vertex_conflict_handler = VertexConflictConstraintHandler(
            instance, self.master
        )

    def solve(self):
        """Run the solver."""

        # Solve.
        self.master.model.optimize()

        # Print solving statistics.
        # self.master.model.printStatistics()

        # Print solution.
        if self.master.model.getNSols() > 0:
            print("")
            obj = self.master.model.getObjVal()
            print(f"Total cost: {obj:.0f}")

            for agent, paths_vars in enumerate(self.master.paths):
                for path, var in paths_vars.items():
                    val = self.master.model.getVal(var)
                    if is_gt(val, 0.0):
                        path_str = ", ".join(
                            map(
                                lambda data: f"({data[1]},{data[0]})",
                                enumerate(path.xys),
                            )
                        )
                        print(f"Agent {agent}: {path_str}")
            print("")


def main():

    # Set up command line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("scen", help="MovingAI .scen file")
    ap.add_argument("--num-agents", nargs="?", const=10, type=int)
    ap.add_argument("--time-limit", help="Time limit in seconds")
    args = ap.parse_args()

    # Read the instance.
    num_agents = int(args.num_agents) if args.num_agents else None
    instance = load_instance(args.scen, num_agents)

    # Solve.
    time_limit = float(args.time_limit) if args.time_limit else None
    bcp = BranchAndCutAndPriceSolver(instance, time_limit)
    bcp.solve()


if __name__ == "__main__":
    main()
