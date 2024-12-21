from sqlalchemy import Table, Column, ForeignKey, Float, ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


ActionActorRelationship = Table(
    'action_actor_relationship',
    Base.metadata,
    Column('action_id', ForeignKey('action.id', ondelete='CASCADE'), primary_key=True),
    Column('relationship_id', ForeignKey('relationship.id', ondelete='CASCADE'), primary_key=True),
)
ActionRecipientRelationship = Table(
    'action_recipient_relationship',
    Base.metadata,
    Column('action_id', ForeignKey('action.id', ondelete='CASCADE'), primary_key=True),
    Column('relationship_id', ForeignKey('relationship.id', ondelete='CASCADE'), primary_key=True),
)


class Relationship(Base):
    __tablename__ = 'relationship'

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(unique=True, nullable=False)
    description: Mapped[str]
    feeling_centroid: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)


class Action(Base):
    __tablename__ = 'action'

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(unique=True, nullable=False)
    description: Mapped[str]
    actor_appraisal_delta: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    recipient_appraisal_delta: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    actor_relationships: Mapped[list[Relationship]] = relationship(
        secondary=ActionActorRelationship,
    )
    recipient_relationships: Mapped[list[Relationship]] = relationship(
        secondary=ActionRecipientRelationship,
    )
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
