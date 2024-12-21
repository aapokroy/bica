from sqlalchemy.orm import Session

from lib import schemas
from lib.clients.database.models import Relationship


def create(
    db: Session,
    title: str,
    description: str,
    feeling_centroid: list[float],
) -> schemas.Relationship:
    relationship = Relationship(
        title=title,
        description=description,
        feeling_centroid=feeling_centroid,
    )
    db.add(relationship)
    db.commit()
    db.refresh(relationship)
    return schemas.Relationship(**relationship.__dict__)


def get(db: Session, id: int) -> schemas.Relationship | None:
    relationship = db.query(Relationship).filter(Relationship.id == id).first()
    if relationship:
        return schemas.Relationship(**relationship.__dict__)


def get_all(db: Session) -> list[schemas.Relationship]:
    relationships = db.query(Relationship).all()
    return [schemas.Relationship(**rel.__dict__) for rel in relationships]


def get_active(db: Session) -> list[schemas.Relationship]:
    relationships = db.query(Relationship).filter(Relationship.is_active == True).all()
    return [schemas.Relationship(**rel.__dict__) for rel in relationships]


def update(
    db: Session,
    id: int,
    title: str | None = None,
    description: str | None = None,
    feeling_centroid: list[float] | None = None,
    is_active: bool | None = None,
) -> schemas.Relationship | None:
    relationship = db.query(Relationship).filter(Relationship.id == id).first()
    if relationship:
        if title:
            relationship.title = title
        if description:
            relationship.description = description
        if feeling_centroid is not None:
            relationship.feeling_centroid = feeling_centroid
        if is_active is not None:
            relationship.is_active = is_active
        db.commit()
        db.refresh(relationship)
        return schemas.Relationship(**relationship.__dict__)


def delete(db: Session, id: int) -> schemas.Relationship:
    relationship = db.query(Relationship).filter(Relationship.id == id).first()
    if relationship:
        db.delete(relationship)
        db.commit()
        return schemas.Relationship(**relationship.__dict__)
