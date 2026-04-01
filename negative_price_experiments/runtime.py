from __future__ import annotations

import os


CPU_WORKERS_ENV = "NEGPRICE_CPU_WORKERS"


def get_cpu_worker_count(*, max_workers: int | None = None, reserve_cores: int = 1) -> int:
    override = os.getenv(CPU_WORKERS_ENV)
    if override is not None:
        try:
            parsed = int(override)
            if parsed > 0:
                return min(parsed, max_workers) if max_workers is not None else parsed
        except ValueError:
            pass

    cpu_total = os.cpu_count() or 1
    workers = max(cpu_total - reserve_cores, 1)
    if max_workers is not None:
        workers = min(workers, max_workers)
    return max(workers, 1)


def get_parallel_worker_count(
    item_count: int,
    *,
    max_workers: int | None = None,
    min_items_per_worker: int = 256,
) -> int:
    if item_count <= 0:
        return 1

    workers = get_cpu_worker_count(max_workers=max_workers)
    workers = min(workers, item_count)
    if item_count < min_items_per_worker:
        return 1

    capacity = max(item_count // min_items_per_worker, 1)
    return max(1, min(workers, capacity))
