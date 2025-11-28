class ForecastSearchResult(BaseModel):
    predictions: List[str] = Field(default_factory=list)
    explanations: List[Optional[str]] = Field(default_factory=list)
    count: int = 0
    fallback: bool = False
    error: Optional[str] = None
