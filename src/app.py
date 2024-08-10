import random
from fastapi import FastAPI, HTTPException

from pydantic_models import Fact, RouteInput
from main import partition_vehicles
from route_optimisation.distance_matrix.cartesian_distance_finder import CartesianDistanceFinder
from route_optimisation.route_solver.brute_force_solver import BruteForceSolver


app = FastAPI()

facts = ["One", "Two", "Three", "Four", "Five"]

split_threshold = 3  # Controls various city count tradeoffs in recursion
dm = CartesianDistanceFinder()
rs = BruteForceSolver()
# Later app iterations will likely include the strategy in the request itself as optionals


def get_total_facts():
    return len(facts)


# Endpoints
@app.get("/")
def default_test():
    return "Switching to FastAPI"


@app.get("/randomfact")
def random_fact():
    total_facts = get_total_facts()
    if total_facts is None:
        # Sample exception handling
        raise HTTPException(status_code=500, detail="Failed to retrieve facts")

    random_fact_id = random.randint(0, total_facts - 1)

    return {"fact": facts[random_fact_id]}


@app.post("/addfact")
def add_fact(new_fact: Fact):
    facts.append(new_fact.fact)
    return {"message": "Fact added successfully", "total_facts": get_total_facts()}


@app.post("/generate-routes")
async def generate_routes(request: RouteInput):
    # Solve VRP
    try:
        optimal_route_per_vehicle, cluster_tree = partition_vehicles(
            request.orders, request.num_vehicle, split_threshold, dm, rs
        )

        # Cluster tree is an arbitrarily recursive list, ending in list[CartesianOrder]
        # It's mainly for debugging, frankly
        print(cluster_tree)
    except ValueError as e:
        # User sent bad data
        raise HTTPException(status_code=400, detail=e)

    return optimal_route_per_vehicle
