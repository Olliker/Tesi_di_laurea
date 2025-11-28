class JobRun(SQLModel, table=True):
    __tablename__: ClassVar[str] = "jobs"

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        sa_column=Column(VARCHAR(36), primary_key=True),
    )
    kind: JobKind = Field(
        sa_column=Column(Enum(JobKind, name="job_kind"), nullable=False)
    )

    status: JobStatus = Field(
        default=JobStatus.queued,
        sa_column=Column(Enum(JobStatus, name="job_status"), nullable=False),
    )

    detail: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    dataset_id: Optional[str] = Field(default=None, index=True, max_length=128)
    output_forecast_run_id: Optional[str] = Field(
        default=None, index=True, max_length=36
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), index=True
    )
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))