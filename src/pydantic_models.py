# Let these act as both API validation and system-wide models

from pydantic import BaseModel


class Fact(BaseModel):
    fact: str


class Order(BaseModel):
    # Synonyms include location (plus id) or customer (orders imply a customer)
    order_id: int  # Might need the alias generator to cast to snake_case
    lat: float
    long: float


class CartesianOrder(Order):  # Internal
    # Adds cartesian fields, allowing cleaner pre-computation
    x: float
    y: float
    z: float


class RouteInput(BaseModel):
    num_vehicle: str
    orders: list[Order]
