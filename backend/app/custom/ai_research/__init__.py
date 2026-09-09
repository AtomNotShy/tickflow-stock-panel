"""Automatic AI research, candidate lifecycle and paper-portfolio extension."""
from app.extensions import (
    BACKEND_EXTENSION_API_VERSION,
    BackendExtensionRegistrar,
    ExtensionContext,
    PipelineCompletedContext,
    PostPipelineHook,
)

from .router import router
from .service import get_service, startup_service

EXTENSION_ID = "tickflow.ai-research"
EXTENSION_API_VERSION = BACKEND_EXTENSION_API_VERSION


class ResearchPipelineHook(PostPipelineHook):
    def after_pipeline(self, context: PipelineCompletedContext) -> None:
        del context
        get_service().enqueue("daily_pipeline")


def setup(registrar: BackendExtensionRegistrar) -> None:
    registrar.include_router(router)
    registrar.register_post_pipeline_hook(
        "tickflow.ai-research.daily", ResearchPipelineHook(), order=100
    )


def startup(context: ExtensionContext) -> None:
    startup_service(context)
