"""Stable, small-grained contracts for in-repository secondary development."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

BACKEND_EXTENSION_API_VERSION = 1


class RepositoryAccess(Protocol):
    """Read-oriented repository surface exposed to backend extensions."""

    def get_name_map(self, symbols: list[str] | None = None) -> dict[str, str]: ...

    def get_enriched_latest(self) -> tuple[Any, Any]: ...

    def get_daily(
        self,
        symbol: str,
        start: Any,
        end: Any,
        columns: list[str] | None = None,
    ) -> Any: ...

    def get_index_daily(
        self,
        symbol: str,
        start: Any,
        end: Any,
        columns: list[str] | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class ExtensionContext:
    api_version: int
    data_dir: Path
    repository: RepositoryAccess


@dataclass(frozen=True)
class NotificationFormatContext:
    api_version: int


@dataclass(frozen=True)
class PipelineCompletedContext:
    """Immutable notification emitted after a daily pipeline is fully persisted."""

    api_version: int
    completed_at: datetime
    result: Mapping[str, Any]


class PostPipelineHook(ABC):
    """Fast callback invoked after a successful daily pipeline.

    Implementations must enqueue long-running work and return promptly. Callback
    failures are isolated and never change the completed pipeline status.
    """

    api_version = BACKEND_EXTENSION_API_VERSION

    @abstractmethod
    def after_pipeline(self, context: PipelineCompletedContext) -> None:
        raise NotImplementedError


class NotificationFormatter(ABC):
    """Customize notification copy without changing the event schema or semantics."""

    api_version = BACKEND_EXTENSION_API_VERSION

    @abstractmethod
    def format_message(
        self,
        event: dict[str, Any],
        context: NotificationFormatContext,
    ) -> str:
        """Return notification copy. The input event must not be mutated."""
        raise NotImplementedError


class DefaultNotificationFormatter(NotificationFormatter):
    def format_message(
        self,
        event: dict[str, Any],
        context: NotificationFormatContext,
    ) -> str:
        del context
        return str(event.get("message") or "")
