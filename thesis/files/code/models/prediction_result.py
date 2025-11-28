class PredictionResult(BaseModel):
    future_predictions: Dict[Hashable, Any]
    MAPE: float
