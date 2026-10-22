"""Qt-independent replay engine.

The engine maps wall-clock time to session time: position advances by
``elapsed_wall * speed``. The caller drives it with ``tick()`` (a QTimer in
the GUI, a fake clock in tests) and receives the rows that became due since
the previous tick. Recorded inter-sample timing is used as-is; for sources
without time the session already carries a labelled synthetic clock.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd

from core.schema import TIME_COL, NormalizedSession

ALLOWED_SPEEDS = (1.0, 2.0, 4.0, 8.0, 16.0)


class ReplayState(str, Enum):
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    FINISHED = "finished"


class ReplayEngine:
    def __init__(self, session: NormalizedSession, clock: Callable[[], float] = time.monotonic):
        if session.data.empty:
            raise ValueError("Cannot replay an empty session.")
        self.session = session
        self._t = session.data[TIME_COL].to_numpy(dtype=float)
        self._t0 = float(self._t[0])
        self._clock = clock
        self.speed = 1.0
        self.state = ReplayState.STOPPED
        self._pos = self._t0           # session time of the playhead
        self._idx = 0                  # next row index not yet emitted
        self._last_wall: float | None = None

    # ---- properties ---------------------------------------------------
    @property
    def position_s(self) -> float:
        return self._pos

    @property
    def start_s(self) -> float:
        """Session time of the first sample."""
        return self._t0

    @property
    def elapsed_s(self) -> float:
        return self._pos - self._t0

    @property
    def duration_s(self) -> float:
        return float(self._t[-1] - self._t0)

    @property
    def index(self) -> int:
        """Number of rows emitted so far (rows [0, index) are 'in the past')."""
        return self._idx

    # ---- controls -----------------------------------------------------
    def play(self) -> None:
        if self.state == ReplayState.FINISHED:
            self.restart()
        self.state = ReplayState.PLAYING
        self._last_wall = self._clock()

    def pause(self) -> None:
        if self.state == ReplayState.PLAYING:
            self._advance()
            self.state = ReplayState.PAUSED

    def resume(self) -> None:
        if self.state == ReplayState.PAUSED:
            self.play()

    def stop(self) -> None:
        self.state = ReplayState.STOPPED
        self._pos, self._idx = self._t0, 0

    def restart(self) -> None:
        self.stop()

    def set_speed(self, speed: float) -> None:
        if speed not in ALLOWED_SPEEDS:
            raise ValueError(f"speed must be one of {ALLOWED_SPEEDS}")
        if self.state == ReplayState.PLAYING:
            self._advance()
        self.speed = speed

    def seek(self, elapsed_s: float) -> None:
        """Jump to ``elapsed_s`` seconds after the first sample. Rows at or
        before the new position count as emitted."""
        elapsed_s = min(max(elapsed_s, 0.0), self.duration_s)
        self._pos = self._t0 + elapsed_s
        self._idx = int(np.searchsorted(self._t, self._pos, side="right"))
        if self.state == ReplayState.FINISHED and self._idx < len(self._t):
            self.state = ReplayState.PAUSED
        self._last_wall = self._clock()

    # ---- driving ------------------------------------------------------
    def _advance(self) -> None:
        now = self._clock()
        if self._last_wall is not None:
            self._pos += (now - self._last_wall) * self.speed
        self._last_wall = now

    def tick(self) -> pd.DataFrame:
        """Advance the playhead and return newly due rows (possibly empty)."""
        if self.state != ReplayState.PLAYING:
            return self.session.data.iloc[0:0]
        self._advance()
        end = int(np.searchsorted(self._t, self._pos, side="right"))
        rows = self.session.data.iloc[self._idx:end]
        self._idx = end
        if end >= len(self._t):
            self._pos = float(self._t[-1])
            self.state = ReplayState.FINISHED
        return rows

    def history(self, max_rows: int | None = None) -> pd.DataFrame:
        """Rows already emitted, optionally only the most recent ``max_rows``."""
        start = 0 if max_rows is None else max(0, self._idx - max_rows)
        return self.session.data.iloc[start:self._idx]

    def current_row(self) -> pd.Series | None:
        return None if self._idx == 0 else self.session.data.iloc[self._idx - 1]
