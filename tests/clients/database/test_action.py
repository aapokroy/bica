from lib.clients.database import crud
from lib.schemas import Action, Relationship


RELATIONSHIP = Relationship(
    id=1,
    title='Friendship',
    description='A close relationship',
    feeling_centroid=[0.5, 0.5, 0.5],
    is_active=True,
)
ACTION = Action(
    id=1,
    title='Chat',
    description='Small talk',
    actor_appraisal_delta=[0.1, 0.2, 0.1],
    recipient_appraisal_delta=[0.1, 0.2, 0.1],
    actor_relationships=[RELATIONSHIP],
    recipient_relationships=[RELATIONSHIP],
    is_active=True,
)


def test_create(db):
    crud.relationship.create(
        db,
        RELATIONSHIP.title,
        RELATIONSHIP.description,
        RELATIONSHIP.feeling_centroid,
    )
    action = crud.action.create(
        db,
        ACTION.title,
        ACTION.description,
        ACTION.actor_appraisal_delta,
        ACTION.recipient_appraisal_delta,
        [rel.id for rel in ACTION.actor_relationships],
        [rel.id for rel in ACTION.recipient_relationships],
    )
    assert action is not None
    assert action == ACTION


def test_delete(db):
    relationship = crud.relationship.create(
        db,
        RELATIONSHIP.title,
        RELATIONSHIP.description,
        RELATIONSHIP.feeling_centroid,
    )
    action = crud.action.create(
        db,
        ACTION.title,
        ACTION.description,
        ACTION.actor_appraisal_delta,
        ACTION.recipient_appraisal_delta,
        [rel.id for rel in ACTION.actor_relationships],
        [rel.id for rel in ACTION.recipient_relationships],
    )
    crud.action.delete(db, action.id)
    action = crud.action.get(db, action.id)
    relationship = crud.relationship.get(db, relationship.id)
    assert action is None
    assert relationship is not None


def test_delete_relationship(db):
    relationship = crud.relationship.create(
        db,
        RELATIONSHIP.title,
        RELATIONSHIP.description,
        RELATIONSHIP.feeling_centroid,
    )
    action = crud.action.create(
        db,
        ACTION.title,
        ACTION.description,
        ACTION.actor_appraisal_delta,
        ACTION.recipient_appraisal_delta,
        [rel.id for rel in ACTION.actor_relationships],
        [rel.id for rel in ACTION.recipient_relationships],
    )
    assert action.recipient_relationships == [relationship]
    crud.relationship.delete(db, relationship.id)
    action = crud.action.get(db, action.id)
    relationship = crud.relationship.get(db, relationship.id)
    assert action is not None
    assert relationship is None
    assert action.recipient_relationships == []
