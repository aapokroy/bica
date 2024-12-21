import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from lib.clients.database.models import Base
from lib.config import cfg


DATABASE_URL = 'postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}'.format(
    host=cfg.postgres.host,
    port=cfg.postgres.port,
    user=cfg.postgres.user,
    password=cfg.postgres.password,
    db=cfg.postgres.test_db,
)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def db():
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        yield db
    Base.metadata.drop_all(bind=engine)
