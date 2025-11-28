class ForecastRun(SQLModel, table=True):
    __tablename__: ClassVar[str] = "forecast_runs"

    id: str = Field(sa_column=Column(VARCHAR(36), primary_key=True))
    mape: float = Field(index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    predictions: List["XGBoostPrediction"] = Relationship(back_populates="run")