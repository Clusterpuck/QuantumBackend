# Let these act as both API validation and system-wide models

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Fact(BaseModel):
    fact: str


class OrderInput(BaseModel):
    # Synonyms include location (plus id) or customer (orders imply a customer)
    # Might need the alias generator to cast to snake_case
    order_id: int = Field(..., ge=0)
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class Order(OrderInput):  # Internal
    # Adds cartesian fields, allowing common pre-computation
    x: float
    y: float
    z: float


class ClusterConfig(BaseModel):
    # Everything related to the clusterer and its needs
    type: str

    # Clustering params depend heavily on the type, so keep all fields
    model_config = ConfigDict(extra="allow")


class SolverConfig(BaseModel):
    # Everything related to the solver and its needs
    type: str
    distance: str
    max_solve_size: int = Field(ge=1)


class RouteInput(BaseModel):
    vehicle_cluster_config: ClusterConfig
    subcluster_config: ClusterConfig
    solver_config: SolverConfig

    orders: list[OrderInput]

    @field_validator("orders")
    @classmethod
    def orders_unique(cls, v: list[OrderInput]) -> list[OrderInput]:
        seen_id = set()
        for order in v:
            if order.order_id in seen_id:
                raise ValueError("order_id must be unique")
            else:
                seen_id.add(order.order_id)
        return v
