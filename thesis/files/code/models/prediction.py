class XGBoostPrediction(SQLModel, table=True):
    __tablename__: ClassVar[str] = "predictions_xgboost"
    __table_args__ = (
        UniqueConstraint(
            "forecast_id", "article_id", "prediction_date", name="uq_pred_key"
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    forecast_id: str = Field(
        sa_column=Column(VARCHAR(36), ForeignKey("forecast_runs.id"))
    )
    article_id: int = Field(index=True)
    article_name: str = Field(max_length=255)
    prediction_date: datetime = Field(index=True)
    predicted_quantity: float

    # lato N:1
    run: Optional["ForecastRun"] = Relationship(back_populates="predictions")