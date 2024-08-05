# Route Solver

This folder contains strategies that can be used to define the route solver. 

route_solver.py contains the abstract class
route_solver_context.py controls which route solving strategy is being used. Utilises strategy pattern

This idea is that you can easily change between route solving strategies without much trouble.
___

## Strategies

### Brute Force Solver
Generates every permutation of possible routes and iterates through them to find the optimal route.

Takes factorial time to solve, not recommended for large routes
