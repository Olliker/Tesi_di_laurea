class ChromaCollection(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: list[float]
