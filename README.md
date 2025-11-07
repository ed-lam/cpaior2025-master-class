# Minimal Branch-and-Cut-and-Price for Simplified Multi-Agent Path Finding

This repository contains a minimal working example of a branch-and-cut-and-price algorithm for solving a variant of the multi-agent path finding (MAPF) problem. In MAPF, there is a gridworld, where some cells are obstacles. There is a set of agents, and every agent is given a start location and a goal location. All start locations are unique and all goal locations are unique, but an agent's goal location could be identical to its start location. The objective is to find paths for every agent from its start to its goal without colliding into other agents and minimising the sum of arrival times. This code considers a simplified variant with only vertex conflicts, meaning that two or more agents cannot occupy the same position at the same time. In particular, the code ignores edge conflicts for simplicity. Additionally, in this variant, agents disappear upon reaching the goal. They are considered to have exited the environment. This avoids complex mechanisms for handling an agent waiting at its goal location indefinitely, as in the common variant.

The solver is inspired by version 1 of BCP-MAPF. The code is written in Python and uses PySCIPOpt as the integer programming solver. This code is meant for educational purposes and should not be used for benchmarking purposes.

The accompanying slides can be found in [slides.pdf](slides.pdf).

## Install

You need a recent version of Python and the PySCIPOpt package. To install PySCIPOpt, run
```bash
pip install pyscipopt
```

## Run the solution
```bash
python3 solution/task2_bcp.py --num-agents 55 instances/empty-16-16-random-9.scen
```

You should see a log of the progress.
