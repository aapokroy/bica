from lib.clients.database import crud
from lib.schemas import Relationship


RELATIONSHIP = Relationship(
    id=1,
    title='Friendship',
    description='A close relationship',
    feeling_centroid=[0.5, 0.5, 0.5],
    is_active=True,
)


def test_create(db):
    relationship = crud.relationship.create(
        db,
        RELATIONSHIP.title,
        RELATIONSHIP.description,
        RELATIONSHIP.feeling_centroid,
    )
    assert relationship is not None
    assert relationship == RELATIONSHIP


def test_get(db):
    relationship = crud.relationship.create(
        db,
        RELATIONSHIP.title,
        RELATIONSHIP.description,
        RELATIONSHIP.feeling_centroid,
    )
    retrieved = crud.relationship.get(db, relationship.id)
    assert relationship is not None
    assert retrieved is not None
    assert relationship == retrieved


def test_get_all(db):
    relationship = crud.relationship.create(
        db,
        RELATIONSHIP.title,
        RELATIONSHIP.description,
        RELATIONSHIP.feeling_centroid,
    )
    retrieved = crud.relationship.get_all(db)
    assert relationship is not None
    assert retrieved is not None
    assert [relationship] == retrieved


def test_update(db):
    relationship = crud.relationship.create(
        db,
        RELATIONSHIP.title,
        RELATIONSHIP.description,
        RELATIONSHIP.feeling_centroid,
    )
    crud.relationship.update(
        db,
        relationship.id,
        title='Best Friends',
        feeling_centroid=[1.0, 1.0, 1.0],
    )
    retrieved = crud.relationship.get(db, relationship.id)
    assert retrieved is not None
    assert retrieved.title == 'Best Friends'
    assert retrieved.feeling_centroid == [1.0, 1.0, 1.0]


def test_delete(db):
    relationship = crud.relationship.create(
        db,
        RELATIONSHIP.title,
        RELATIONSHIP.description,
        RELATIONSHIP.feeling_centroid,
    )
    crud.relationship.delete(db, relationship.id)
    retrieved = crud.relationship.get(db, relationship.id)
    assert retrieved is None
