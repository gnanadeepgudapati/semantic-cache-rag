# models.py
from pydantic import BaseModel

class IngestRequest(BaseModel):
    title: str
    text: str

class IngestResponse(BaseModel):
    message: str
    chunks_created: int

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]