forecasting = APIRouter(dependencies=[Depends(jwt_dependencies.check_token)])

class ForecastRequest(BaseModel):
    start_date: datetime = Field(..., description="Data di inizio previsione inclusa")
    periods: int = Field(default=6, gt=0, description="Numero di finestre da prevedere")

    @model_validator(mode="after")
    def _check_dates(self) -> "ForecastRequest":
        if self.periods <= 0:
            raise ValueError("periods must be greater than zero")
        return self

@forecasting.post("/forecast")
def forecast(
    background_tasks: BackgroundTasks,
    response: Response,
    payload: ForecastRequest,
):
    job_service = JobService()

    job = job_service.create_job(kind=JobKind.forecasting)

    job_id_raw = getattr(job, "id", None) or getattr(job, "job_id", None)
    job_id_uuid = UUID(str(job_id_raw))
    response.headers["Location"] = f"/jobs/{job_id_uuid}"

    forecast_id = str(uuid4())

    def task():
        try:
            job_service.update_job_status(job_id_uuid, JobStatus.running)
            ForecastingService().forecast(
                forecast_id=forecast_id,
                start_date=payload.start_date,
                periods=payload.periods,
            )
            job_service.update_job_status(job_id_uuid, JobStatus.succeeded)

            NotificationService().notify(
                NotificationEvent(
                    event="forecast_completed",
                    job_id=str(job_id_uuid),
                    forecast_id=forecast_id,
                    status="succeeded",
                )
            )
        except Exception as e:
            job_service.update_job_status(job_id_uuid, JobStatus.failed, detail=str(e))
            NotificationService().notify(
                NotificationEvent(
                    event="forecast_failed",
                    job_id=str(job_id_uuid),
                    forecast_id=forecast_id,
                    status="failed",
                    error=str(e),
                )
            )

    background_tasks.add_task(task)
    return {
        "job": Job(
            job_id=str(job_id_uuid), kind=JobKind.forecasting, status=JobStatus.queued
        ),
        "forecast_id": forecast_id,
    }
