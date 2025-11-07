from instance import load_instance
from pricer import AStarPricer
import argparse


def main():

    # Set up command line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("scen", help="MovingAI .scen file")
    args = ap.parse_args()

    # Read the instance.
    instance = load_instance(args.scen, 1)

    # Run A* to find a path for agent 0.
    astar = AStarPricer(instance=instance, debug=True)
    astar.solve(mode="redcost", agent=0)


if __name__ == "__main__":
    main()
