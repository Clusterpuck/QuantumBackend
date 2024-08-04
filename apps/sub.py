#Get endpoint sample
from fastapi import APIRouter

api_router = APIRouter()

@api_router.get("/hello")
def hello():
    return "Hello World in subfile"