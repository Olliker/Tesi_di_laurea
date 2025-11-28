from forecasting.app.models.db.forecast_db import JobRun
from forecasting.app.repositories.job_repository import (
    JobRepository,
    JobKind,
    JobStatus,
)
from forecasting.app.models.job import Job

from typing import Optional
from uuid import UUID


class JobService:
    def __init__(self, repo: JobRepository | None = None):
        self.db = repo or JobRepository()

    def create_job(self, kind: JobKind) -> JobRun:
        return self.db.create_job(kind=kind)

    def update_job_status(
        self, job_id: UUID, status: JobStatus, detail: Optional[str] = None
    ) -> None:
        self.db.update_job_status(job_id, status, detail)

    def get_job(self, job_id: UUID) -> Optional[Job]:
        return self.db.get_job(job_id)

    def list_jobs(
        self, kind: Optional[JobKind] = None, status: Optional[JobStatus] = None
    ) -> list[Job]:
        return self.db.list_jobs(kind, status)
