"""The VehicleDataSource contract shared by recorded files, synthetic demo data
and the (future, optional) OBD-II adapter. Everything downstream - validation,
replay, analysis, history and reports - consumes NormalizedSession / Reading,
so adding a real adapter later does not require changes there."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

from core.schema import DataOrigin, NormalizedSession, TimeBasis


@dataclass(frozen=True)
class SourceCapabilities:
    origin: DataOrigin
    time_basis: TimeBasis
    signals: tuple[str, ...]          # canonical fields this source can provide
    is_live: bool = False             # True only for a physically connected adapter
    supports_seek: bool = True
    hardware_tested: bool = False     # True only after a documented real-vehicle test


@dataclass
class Reading:
    """One timestamped multi-signal sample. Missing signals are absent from ``values``."""
    t_s: float
    values: dict[str, float]
    origin: DataOrigin
    unsupported: list[str] = field(default_factory=list)


class DataSourceError(RuntimeError):
    """Raised with a user-actionable message when a source cannot be read."""


class VehicleDataSource(ABC):
    name: str = "source"

    @abstractmethod
    def capabilities(self) -> SourceCapabilities: ...

    def open(self) -> None:  # noqa: B027 - optional for batch sources
        pass

    def close(self) -> None:  # noqa: B027
        pass

    @abstractmethod
    def read_session(self) -> NormalizedSession:
        """Return the full normalized data (batch sources) or what has been
        recorded so far (live sources)."""

    def iter_readings(self) -> Iterator[Reading]:
        """Default streaming view over a batch session."""
        s = self.read_session()
        cols = s.signals
        for row in s.data.itertuples(index=False):
            d = row._asdict()
            vals = {c: float(d[c]) for c in cols if d.get(c) == d.get(c)}  # drop NaN
            yield Reading(float(d["t_s"]), vals, s.meta.origin)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
