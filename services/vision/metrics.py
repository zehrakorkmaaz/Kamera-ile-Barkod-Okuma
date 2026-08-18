"""Thread-safe counters and latency statistics for the scanner.

These exist so that when the system is tested against real products it is
possible to answer *where* it failed -- no candidate found, decode failed,
checksum rejected, or product missing from the catalogue -- instead of only
knowing that nothing happened.
"""
from collections import deque
from dataclasses import dataclass, field
import threading

COUNTERS = (
    "total_frames", "processed_frames", "dropped_frames", "barcode_candidates",
    "decode_attempts", "decode_success", "decode_failure", "checksum_failure",
    "decoder_conflicts", "confirmations", "product_found", "product_not_found",
    "duplicate_scan_count", "camera_reconnects", "pipeline_errors",
)


@dataclass
class ScanMetrics:
    window: int = 120
    _counters: dict[str, int] = field(default_factory=lambda: {name: 0 for name in COUNTERS})
    _decode_latencies: deque = field(default_factory=deque)
    _confirmation_times: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        self._decode_latencies = deque(maxlen=self.window)
        self._confirmation_times = deque(maxlen=self.window)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def observe_decode_latency(self, milliseconds: float) -> None:
        with self._lock:
            self._decode_latencies.append(float(milliseconds))

    def observe_confirmation_time(self, milliseconds: float) -> None:
        with self._lock:
            self._confirmation_times.append(float(milliseconds))

    def snapshot(self) -> dict:
        with self._lock:
            counters = dict(self._counters)
            latencies = sorted(self._decode_latencies)
            confirmations = list(self._confirmation_times)
        attempts = counters["decode_attempts"]
        return {
            **counters,
            "decode_success_rate": round(counters["decode_success"] / attempts, 3) if attempts else 0.0,
            "average_decode_latency_ms": _mean(latencies),
            "p95_decode_latency_ms": _percentile(latencies, 0.95),
            "average_confirmation_time_ms": _mean(confirmations),
        }

    def reset(self) -> None:
        with self._lock:
            self._counters = {name: 0 for name in COUNTERS}
            self._decode_latencies.clear()
            self._confirmation_times.clear()


def _mean(values) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


def _percentile(sorted_values, fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return round(sorted_values[index], 1)
