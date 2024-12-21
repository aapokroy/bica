from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresConfig(BaseModel):
    host: str = 'postgres'
    port: int = 5432
    user: str = 'postgres'
    password: str = 'postgres'
    db: str = 'db'
    test_db: str = 'test_db'


class StreamlitConfig(BaseModel):
    page_icon: str = ':material/psychology:'


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter='__',
        env_file='.env',
    )

    postgres: PostgresConfig = PostgresConfig()
    streamlit: StreamlitConfig = StreamlitConfig()


cfg = Config()
