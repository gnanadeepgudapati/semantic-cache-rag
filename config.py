# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    chunk_size: int = 200
    overlap: int = 50
    top_k: int = 3

    class Config:
        env_file = ".env"

settings = Settings()