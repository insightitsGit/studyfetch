from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class PipelineProgress:
    pipeline_id: str
    status: str = "queued"
    percent: int = 0
    stage: str = ""
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status,
            "percent": self.percent,
            "stage": self.stage,
            "detail": self.detail,
        }


@dataclass
class Job:
    id: str
    document_id: str
    pipelines: list[PipelineProgress]
    status: str = "queued"
    stage: str = "Queued"
    detail: str = ""
    error: str | None = None
    results: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def overall_percent(self) -> int:
        if not self.pipelines:
            return 0
        total = sum(
            100 if p.status == "ok" else (p.percent if p.status == "running" else 0)
            for p in self.pipelines
        )
        return min(100, int(total / len(self.pipelines)))

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "document_id": self.document_id,
            "status": self.status,
            "percent": 100 if self.status == "ok" else self.overall_percent(),
            "stage": self.stage,
            "detail": self.detail,
            "error": self.error,
            "results": self.results,
            "pipelines": [p.as_dict() for p in self.pipelines],
        }


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def create(self, document_id: str, pipeline_ids: list[str]) -> Job:
        job = Job(
            id=f"job_{uuid.uuid4().hex[:12]}",
            document_id=document_id,
            pipelines=[PipelineProgress(pipeline_id=p) for p in pipeline_ids],
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, val in kwargs.items():
                setattr(job, key, val)


jobs = JobStore()


class ProgressReporter:
    """Callback a pipeline uses to publish stage completion (0–100 within that pipeline)."""

    def __init__(self, job: Job, pipeline_id: str):
        self.job = job
        self.pipeline_id = pipeline_id

    def _pipe(self) -> PipelineProgress:
        for p in self.job.pipelines:
            if p.pipeline_id == self.pipeline_id:
                return p
        raise KeyError(self.pipeline_id)

    def start(self) -> None:
        pipe = self._pipe()
        pipe.status = "running"
        pipe.percent = 1
        pipe.stage = "starting"
        self.job.status = "running"
        self.job.stage = f"{self.pipeline_id}: starting"

    def stage(self, name: str, local_percent: int, detail: str = "") -> None:
        pipe = self._pipe()
        pipe.status = "running"
        pipe.percent = max(0, min(99, int(local_percent)))
        pipe.stage = name
        pipe.detail = detail
        self.job.status = "running"
        self.job.stage = f"{self.pipeline_id}: {name}"
        self.job.detail = detail

    def finish(self, ok: bool = True, error: str | None = None) -> None:
        pipe = self._pipe()
        pipe.status = "ok" if ok else "error"
        pipe.percent = 100 if ok else pipe.percent
        pipe.stage = "done" if ok else "failed"
        pipe.detail = error or pipe.detail
        if not ok:
            self.job.status = "error"
            self.job.error = error
            self.job.stage = f"{self.pipeline_id}: failed"


def bind_progress(pipeline, reporter: ProgressReporter | None) -> Callable[..., None]:
    def report(name: str, local_percent: int, detail: str = "") -> None:
        if reporter:
            reporter.stage(name, local_percent, detail)

    pipeline.report = report
    return report
