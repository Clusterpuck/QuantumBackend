import random
from fastapi import FastAPI, HTTPException

app = FastAPI()

facts = ["One", "Two", "Three", "Four", "Five"]

def get_total_facts():
        return len(facts)

@app.get("/")
def default_test():
    return "Removed database connection"

@app.get("/randomfact")
def random_fact():
    total_facts = get_total_facts()
    if total_facts is None:
        #Sample exception handling
        raise HTTPException(status_code=500, detail="Failed to retrieve facts")

    random_fact_id = random.randint(1, total_facts-1)

    return {'fact': facts[random_fact_id]}
