job = APIRouter(dependencies=[Depends(jwt_dependencies.check_token)])

@job.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: UUID):
    job = JobService().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@job.get("/jobs", response_model=list[Job])
def list_jobs(kind: Optional[JobKind] = None, status: Optional[JobStatus] = None):
    return JobService().list_jobs(kind, status)
