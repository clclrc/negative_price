from __future__ import annotations

from math import isfinite
from time import monotonic
from typing import Callable, Sequence


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(float(seconds), 0.0)
    if seconds >= 60.0:
        total_seconds = int(round(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{seconds:.1f}s"


def format_metric(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if not isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def format_rate(count: int | float, elapsed_seconds: float) -> str:
    if elapsed_seconds <= 0:
        return "n/a"
    return f"{float(count) / elapsed_seconds:.1f} samples/s"


def estimate_remaining_seconds(*, loop_started_at: float, completed_steps: int, total_steps: int, now: float) -> float | None:
    if total_steps <= completed_steps:
        return 0.0
    if completed_steps <= 0:
        return None
    elapsed = max(now - loop_started_at, 0.0)
    average_step = elapsed / completed_steps
    return average_step * (total_steps - completed_steps)


def format_prefix(parts: Sequence[str]) -> str:
    return "".join(f"[{part}]" for part in parts if part)


class ProgressReporter:
    def __init__(
        self,
        *,
        time_fn: Callable[[], float] = monotonic,
        print_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._time_fn = time_fn
        self._print_fn = print_fn or self._default_print

    @staticmethod
    def _default_print(line: str) -> None:
        print(line, flush=True)

    def now(self) -> float:
        return float(self._time_fn())

    def log(self, prefix: Sequence[str], message: str) -> None:
        line_prefix = format_prefix(prefix)
        if line_prefix:
            self._print_fn(f"{line_prefix} {message}")
            return
        self._print_fn(message)

    def log_step(
        self,
        prefix: Sequence[str],
        *,
        label: str,
        index: int,
        total: int,
        loop_started_at: float,
        step_started_at: float,
        extra: str | None = None,
    ) -> None:
        now = self.now()
        message = (
            f"{label} {index}/{total} completed | step={format_duration(now - step_started_at)} "
            f"| elapsed={format_duration(now - loop_started_at)} "
            f"| eta={format_duration(estimate_remaining_seconds(loop_started_at=loop_started_at, completed_steps=index, total_steps=total, now=now))}"
        )
        if extra:
            message = f"{message} | {extra}"
        self.log(prefix, message)
