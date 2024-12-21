from sqlalchemy.orm import Session

from lib import schemas
from lib.clients.database.models import Action, Relationship
from lib.clients.database.utils import sqlalchemy_to_dict


def create(
    db: Session,
    title: str,
    description: str,
    actor_appraisal_delta: list[float],
    recipient_appraisal_delta: list[float],
    actor_relationships: list[int],
    recipient_relationships: list[int],
) -> schemas.Action:
    actor_rels = db.query(Relationship).filter(
        Relationship.id.in_(actor_relationships)
    ).all()
    recipient_rels = db.query(Relationship).filter(
        Relationship.id.in_(recipient_relationships)
    ).all()
    action = Action(
        title=title,
        description=description,
        actor_appraisal_delta=actor_appraisal_delta,
        recipient_appraisal_delta=recipient_appraisal_delta,
        actor_relationships=actor_rels,
        recipient_relationships=recipient_rels,
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return schemas.Action(**sqlalchemy_to_dict(action))


def get(db: Session, id: int) -> schemas.Action | None:
    action = db.query(Action).filter(Action.id == id).first()
    if action:
        return schemas.Action(**sqlalchemy_to_dict(action))


def get_all(db: Session) -> list[schemas.Action]:
    actions = db.query(Action).all()
    return [schemas.Action(**sqlalchemy_to_dict(action)) for action in actions]


def get_active(db: Session) -> list[schemas.Action]:
    actions = db.query(Action).filter(Action.is_active == True).all()
    return [schemas.Action(**sqlalchemy_to_dict(action)) for action in actions]


def update(
    db: Session,
    id: int,
    title: str | None = None,
    description: str | None = None,
    actor_appraisal_delta: list[float] | None = None,
    recipient_appraisal_delta: list[float] | None = None,
    actor_relationships: list[int] | None = None,
    recipient_relationships: list[int] | None = None,
    is_active: bool | None = None,
) -> schemas.Action | None:
    action = db.get(Action, id)
    if action:
        if title:
            action.title = title
        if description:
            action.description = description
        if actor_appraisal_delta is not None:
            action.actor_appraisal_delta = actor_appraisal_delta
        if recipient_appraisal_delta is not None:
            action.recipient_appraisal_delta = recipient_appraisal_delta
        if actor_relationships is not None:
            action.actor_relationships = db.query(Relationship).filter(
                Relationship.id.in_(actor_relationships)
            ).all()
        if recipient_relationships is not None:
            action.recipient_relationships = db.query(Relationship).filter(
                Relationship.id.in_(recipient_relationships)
            ).all()
        if is_active is not None:
            action.is_active = is_active
        db.commit()
        db.refresh(action)
        return schemas.Action(**sqlalchemy_to_dict(action))


def delete(db: Session, id: int) -> schemas.Action:
    action = db.query(Action).filter(Action.id == id).first()
    if action:
        db.delete(action)
        db.commit()
        return schemas.Action(**sqlalchemy_to_dict(action))
