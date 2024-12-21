from typing import Any


def sqlalchemy_to_dict(obj: Any) -> dict:
    if isinstance(obj, list):
        return [sqlalchemy_to_dict(item) for item in obj]

    if not hasattr(obj, '__table__'):
        return obj

    data = {}
    for column in obj.__table__.columns:
        data[column.name] = getattr(obj, column.name)

    for rel in obj.__mapper__.relationships:
        value = getattr(obj, rel.key)
        if value is not None:
            if rel.uselist:
                data[rel.key] = [sqlalchemy_to_dict(v) for v in value]
            else:
                data[rel.key] = sqlalchemy_to_dict(value)

    return data
