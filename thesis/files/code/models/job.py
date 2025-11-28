class JobKind(str, Enum):
    training = "training"
    forecasting = "forecasting"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class Job(BaseModel):
    job_id: str
    status: JobStatus
    kind: JobKind
    detail: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
