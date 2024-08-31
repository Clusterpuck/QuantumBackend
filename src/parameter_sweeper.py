import ast

"""
Optimal parameters are unknown, we must incorporate a method to sweep for optimal D-Wave parameters. Add everything to a separate folder. Its goal must be able to provide sufficient data to allow parameter sets to be analysed.

Input = sets of D-Wave hyperparams, these will have to be added to the solver factory.
Input = Exclusion list, exclude a particular combination, rather have this earlier than later
Output = Output file storing data for analysis

Structure

- Preprocessing + relevant data to begin processing
- Loop over the combination of parameters
  - Call generate routes function with D-Wave solver of a configuration
  - Write results to file

Will need some metric for "optimality". Could compare costs of BFS and Quantum?
2 parameters, can do simple graph?
Heatmap?
Contour plot?
"""
# python .\parameter_sweeper.py "newFile"
# Plan
import itertools
import os
import sys
from distance_factory import DistanceFactory
from pydantic_models import RouteInput, Order, OrderInput
from solver_factory import SolverFactory
from vehicle_clusterer_factory import VehicleClustererFactory

vehicle_clusterer_factory = VehicleClustererFactory()
distance_factory = DistanceFactory()
solver_factory = SolverFactory()
# Args should be
    # File path for payload
    # File path for paramerters
    # File path for output

# Call with args, output will be in data folder
def wrapper():
    # Read payload, must be quantum
    # Read tuning_parameters, get combinations of params
    # Read solver_params
    # Write to file, Check if it can write before we start
    # Make the clusterers, distance finder via factories
    # Setup brute force so we have a baseline
    # for each combination of params
        # Recreate the solver with new params
        # solve_vrp, extend and override so I can get cost
            # test with brute force, compare with hardcoded custom route before we go quantum
        # Append analysis data to file
    # Save matplotlib heatmap to data folder
    # Save a contour plot if have time
    # Save something to judge individual hyperparams
    f = open(os.path.join("data", sys.argv[1]), "x")
    f.write("Now the file has more content")
    f.close()


# Read payload
def read_payload(filepath : str) -> list[Order]:
    # Read payload file, get orders, max solve size
    # Hardcode the other params for setup
    # Use orders_to_cartesian before returning
    pass

# Read parameters
def get_tuning_parameters(file_path : str) -> list:
    # Plan
    # Read file to get cost-constraint ratio and chain
    result = {}
    with open(os.path.join("data", file_path), 'r') as file:
        for line in file:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                # Convert the value part from string to list using ast.literal_eval
                value = ast.literal_eval(value.strip())
                result[key] = value
    print(type(result.get("cost_constraint_ratio")))
    result = get_combinations(result.get("cost_constraint_ratio"), result.get("chain_strength"))
    print(result)
    return result

# TODO Can just combine this with above
def get_combinations(list1, list2):
    list3 = list(itertools.product(list1, list2))
    #print(list3)
    return list3

# Read D-Wave settings
def get_solver_parameters(file_path : str) -> dict:
    parameters = {}
    file = open(os.path.join("data", file_path), "r") #TODO Work out this encoding that pylint is screaming about
    for line in file:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()  # Remove surrounding quotes from the key
            value = parse_value(value)    # Convert value to the appropriate type
            parameters[key] = value
    # TODO Validate Params before return
    print(parameters)
    return parameters

def parse_value(value : str) -> int | bool:
    if value.isdigit():
        value = int(value)
    elif value.lower() == 'true':
        value = True
    elif value.lower() == 'false':
        value = False
    else:
        raise TypeError("Unexpected type when parsing")
    print(value)
    print(type(value))
    return value

get_solver_parameters("solver_params")
get_tuning_parameters("tuning_params")

