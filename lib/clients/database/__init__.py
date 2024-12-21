from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from lib.config import cfg


DATABASE_URL = 'postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}'.format(
    host=cfg.postgres.host,
    port=cfg.postgres.port,
    user=cfg.postgres.user,
    password=cfg.postgres.password,
    db=cfg.postgres.db,
)


engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
