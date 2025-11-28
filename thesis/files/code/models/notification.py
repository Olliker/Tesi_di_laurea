class NotificationEvent(BaseModel):
    event: str
    job_id: str
    forecast_id: str
    status: str
    error: Optional[str] = None
