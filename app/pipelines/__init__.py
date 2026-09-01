from app.pipelines.base import PIPELINES, ingest_pdf, resolve_pdf_path
from app.pipelines.baseline import BaselinePipeline
from app.pipelines.prism import PrismPipeline, PrismShield
from app.pipelines.relay import RelayPipeline

REGISTRY = {
    "baseline": BaselinePipeline,
    "prism": PrismPipeline,
    "relay": RelayPipeline,
}


def get_pipeline(name: str, store):
    if name not in REGISTRY:
        raise KeyError(name)
    return REGISTRY[name](store)


__all__ = [
    "PIPELINES",
    "REGISTRY",
    "ingest_pdf",
    "resolve_pdf_path",
    "get_pipeline",
    "BaselinePipeline",
    "PrismPipeline",
    "RelayPipeline",
    "PrismShield",
]
