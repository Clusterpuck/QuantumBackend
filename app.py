import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from apps.sub import api_router


app = FastAPI()

app.include_router(api_router)

facts = ["One", "Two", "Three", "Four", "Five"]

#Needed for defining an incoming class
class Fact(BaseModel):
    fact: str

def get_total_facts():
        return len(facts)

@app.get("/")
def default_test():
    return "Hello World"

@app.get("/randomfact")
def random_fact():
    total_facts = get_total_facts()
    if total_facts is None:
        #Sample exception handling
        raise HTTPException(status_code=500, detail="Failed to retrieve facts")

    random_fact_id = random.randint(0, total_facts-1)

    return {'fact': facts[random_fact_id]}

@app.post("/addfact")
def add_fact(new_fact: Fact):
    facts.append(new_fact.fact)
    return {"message": "Fact added successfully", "total_facts": get_total_facts()}
