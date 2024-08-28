# Route Solver

This folder contains strategies that can be used to define the route solver. 

route_solver.py contains the abstract class that new solvers must inherit from.

___

## Strategies

### Brute Force Solver
Generates every permutation of possible routes and iterates through them to find the optimal route.

Guaranteed to find optimal route.

Takes factorial time to solve, not recommended for large routes.

### D-Wave Solver
Formulates the Asymmetrical Travelling Salesman Problem (ATSP) into a Quadratic Unconstrained Binary Optimisation (QUBO).
D-Wave converts the QUBO into a Binary Quadratic Model so the quantum annealer provide a solution to the ATSP.

The QUBO for the TSP formulation will return a dictionary of 2-tuple binary variables (BV) with corresponding energy levels, also referred to as QUBO edges.
The formulation is defined by 4 sets of requirements:
 - Cost Objective
    - Encodes asymmetrical distances as QUBO edges between sequential nodes.
    - Penalty directly proportional to the cost of a route.
    - Incentives picking lower cost routes.
 - Selection Incentive
    - Encourages variable selection.
    - Reward for making any variable selection.
    - Incentives picking enough nodes for valid solution.
 - Time Constraints
    - Encodes horizontal one-hot violations.
    - Penalty for being in multiple places at same time.
    - Alongside position constraints, discourages too many selections.
 - Position Constraints
    - Encodes vertical one-hot violations.
    - Penalty for revisiting nodes.
    - Alongside time constraints, discourages too many selections.

Lower overall energy levels should represent more cost-efficient solutions. As a result, the D-Wave annealer aims to find the sample of minimum energy.

The QUBO formulation is sent to the D-Wave machine which returns a response.
The response contains a record of samples taken, their energy level and their number of occurences.
Due to the limitations of QUBO being unable to model hard constraints, the response can contain both valid and invalid solutions, so we must filter out all invalid solutions.

We cherry-pick the solution with the minimum energy. 
Typically the best solution found by D-Wave corresponds to the sample that occurred the most, however, this may not always be the case.

The solver will then return the best route and it's cost.

Aims to find the approximate optimal route.
