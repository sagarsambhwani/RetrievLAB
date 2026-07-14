from pydantic import BaseModel


class BenchmarkCase(BaseModel):
    query: str
    relevant_chunk_ids: list[str]