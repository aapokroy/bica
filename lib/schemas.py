from pydantic import BaseModel, ConfigDict


class Relationship(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int
    title: str
    description: str
    feeling_centroid: list[float]
    is_active: bool


class Action(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int
    title: str
    description: str
    actor_appraisal_delta: list[float]
    recipient_appraisal_delta: list[float]
    actor_relationships: list[Relationship]
    recipient_relationships: list[Relationship]
    is_active: bool
