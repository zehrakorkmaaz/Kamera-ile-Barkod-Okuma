"""Temporal confirmation, duplicate suppression and the scan state machine.

A camera can read the same barcode thirty times a second; a shopping cart must
record one product.  Equally, a single frame is thin evidence, so a read is
carried across frames and only promoted to an event once it has earned it:

    IDLE -> DETECTING -> CANDIDATE -> CONFIRMED -> PRODUCT_FOUND -> COOLDOWN -> READY

Suppression is per barcode value, not global: showing a *different* product
immediately after one another is a normal checkout action and must never be
delayed.  A repeat of the *same* code is only accepted once it has left the
frame and its cooldown has passed, so a product resting in view cannot be
counted twice, and a genuinely re-scanned product is never locked out forever.
"""
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import time

from services.config import DEFAULT_CONFIG, ScannerConfig
from services.vision.confidence import ConfidenceLevel
from services.vision.quality import QualityHint


class ScanState(str, Enum):
    IDLE = "IDLE"
    DETECTING = "DETECTING"
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    PRODUCT_FOUND = "PRODUCT_FOUND"
    PRODUCT_UNKNOWN = "PRODUCT_UNKNOWN"
    RESULT_CONFLICT = "RESULT_CONFLICT"
    COOLDOWN = "COOLDOWN"
    READY = "READY"


STATE_MESSAGES = {
    ScanState.IDLE: "Barkodu kameraya gösterin",
    ScanState.DETECTING: "Barkod algılandı, okunuyor…",
    ScanState.CANDIDATE: "Barkod doğrulanıyor…",
    ScanState.CONFIRMED: "Barkod okundu",
    ScanState.PRODUCT_FOUND: "Ürün doğrulandı",
    ScanState.PRODUCT_UNKNOWN: "Bu barkod ürün kataloğunda bulunamadı",
    ScanState.RESULT_CONFLICT: "Barkod net okunamadı, ürünü yeniden gösterin",
    ScanState.COOLDOWN: "Sonraki ürünü gösterebilirsiniz",
    ScanState.READY: "Barkodu kameraya gösterin",
}

TOO_FAR_MESSAGE = "Barkodu kameraya yaklaştırın"


@dataclass
class TrackedBarcode:
    """Everything known about one barcode value across recent frames."""
    value: str
    format: str = ""
    first_seen: float = 0.0
    last_seen: float = 0.0
    frame_count: int = 0
    successful_decode_count: int = 0
    confidence: float = 0.0
    best_confidence: float = 0.0
    _hits: deque = field(default_factory=lambda: deque(maxlen=32), repr=False)

    def observe(self, confidence: float, now: float, decoded: bool = True) -> None:
        if not self.first_seen:
            self.first_seen = now
        self.last_seen = now
        self.frame_count += 1
        if decoded:
            self.successful_decode_count += 1
            self._hits.append(now)
        self.confidence = confidence
        self.best_confidence = max(self.best_confidence, confidence)

    def recent_hits(self, now: float, window: float) -> int:
        """Decodes of this value inside the confirmation window."""
        return sum(1 for at in self._hits if now - at <= window)

    @property
    def stability(self) -> float:
        """Share of observed frames in which this value actually decoded."""
        return self.successful_decode_count / self.frame_count if self.frame_count else 0.0

    @property
    def age_ms(self) -> float:
        return (self.last_seen - self.first_seen) * 1000

    def as_dict(self) -> dict:
        return {"value": self.value, "format": self.format, "first_seen": self.first_seen,
                "last_seen": self.last_seen, "frame_count": self.frame_count,
                "successful_decode_count": self.successful_decode_count,
                "confidence": round(self.confidence, 1),
                "stability": round(self.stability, 2)}


@dataclass
class ScanEvent:
    """A confirmed scan, ready for product lookup."""
    value: str
    format: str
    confidence: float
    level: ConfidenceLevel
    decoders: tuple[str, ...]
    frames: int
    confirmation_ms: float
    at: float

    def as_dict(self) -> dict:
        return {"barcode": self.value, "format": self.format,
                "confidence": round(self.confidence, 1), "level": self.level.value,
                "decoders": list(self.decoders), "frames": self.frames,
                "confirmation_ms": round(self.confirmation_ms, 1), "at": self.at}


class ScanTracker:
    """Consumes per-frame pipeline results and emits confirmed scan events."""

    def __init__(self, config: ScannerConfig = DEFAULT_CONFIG, metrics=None):
        self.config = config
        self.metrics = metrics
        self.state = ScanState.IDLE
        self.tracks: dict[str, TrackedBarcode] = {}
        self.active_value: str | None = None
        self.last_event: ScanEvent | None = None
        self._last_event_at: dict[str, float] = {}
        self._misses = 0
        self._raw_hint = ScanState.IDLE
        self._hint_message = STATE_MESSAGES[ScanState.IDLE]
        self._pending_message = self._hint_message
        self._pending_since = 0.0

    # -- main entry point --------------------------------------------------

    def update(self, result, now: float | None = None) -> ScanEvent | None:
        """Fold one frame result into the tracker; returns an event when confirmed."""
        now = time.monotonic() if now is None else now
        self._prune(now)

        if result.conflict:
            # Two decoders disagreed: never guess, ask for another look.
            self._transition(ScanState.RESULT_CONFLICT, now, result)
            return None

        if not result.value:
            return self._observe_miss(now, result)

        track = self.tracks.get(result.value)
        if track is None:
            track = TrackedBarcode(result.value, result.format)
            self.tracks[result.value] = track
        track.format = result.format or track.format
        track.observe(result.confidence, now)
        self._misses = 0

        if result.value == self.active_value:
            # Same product still sitting in front of the camera -- one scan only.
            self._count("duplicate_scan_count")
            self._transition(self._post_scan_state(), now, result)
            return None

        if not self._cooldown_elapsed(track.value, now):
            # Left the frame and came straight back; wait out the cooldown so a
            # wobble cannot register the same item twice.
            self._count("duplicate_scan_count")
            self._transition(ScanState.COOLDOWN, now, result)
            return None

        if not self._is_confirmed(track, result, now):
            self._transition(ScanState.CANDIDATE, now, result)
            return None

        return self._emit(track, result, now)

    # -- state helpers -----------------------------------------------------

    def _observe_miss(self, now: float, result) -> None:
        self._misses += 1
        if self._misses >= self.config.miss_tolerance:
            # The code left the frame: the same product may be scanned again.
            self.active_value = None
        if self.active_value or self._holding_result(now):
            self._transition(self._post_scan_state(), now, result)
        elif result.candidates:
            self._transition(ScanState.DETECTING, now, result)
        elif self._in_cooldown(now):
            self._transition(ScanState.COOLDOWN, now, result)
        else:
            self._transition(ScanState.IDLE, now, result)
        return None

    def _is_confirmed(self, track: TrackedBarcode, result, now: float) -> bool:
        """Prefer immediate acceptance of a checksum-valid decoder result.

        ZXing only reaches this method with a valid decoded barcode. Waiting for
        a second frame when the confidence is already at the normal threshold
        adds visible scan latency without improving the barcode value. Very
        weak reads still require temporal agreement as a safety fallback.
        """
        if result.confidence >= self.config.confidence_threshold:
            return True
        return track.recent_hits(now, self.config.confirmation_window_seconds) >= \
            self.config.confirmation_frames

    def _emit(self, track: TrackedBarcode, result, now: float) -> ScanEvent:
        event = ScanEvent(track.value, track.format or result.format, result.confidence,
                          result.level, tuple(result.decoders), track.successful_decode_count,
                          track.age_ms, now)
        self.active_value = track.value
        self._last_event_at[track.value] = now
        self.last_event = event
        self._transition(ScanState.CONFIRMED, now, result)
        self._count("confirmations")
        if self.metrics is not None:
            self.metrics.observe_confirmation_time(track.age_ms)
        return event

    def note_product(self, found: bool, now: float | None = None) -> None:
        """Record the catalogue outcome for the most recent confirmed scan."""
        now = time.monotonic() if now is None else now
        self.state = ScanState.PRODUCT_FOUND if found else ScanState.PRODUCT_UNKNOWN
        self._set_message(STATE_MESSAGES[self.state], now, immediate=True)
        self._count("product_found" if found else "product_not_found")

    def _post_scan_state(self) -> ScanState:
        """Keep showing the outcome of the scan that is still in view."""
        if self.state in (ScanState.PRODUCT_FOUND, ScanState.PRODUCT_UNKNOWN,
                          ScanState.RESULT_CONFLICT):
            return self.state
        return ScanState.CONFIRMED if self.active_value else ScanState.COOLDOWN

    def _holding_result(self, now: float) -> bool:
        """A finished scan stays on screen long enough to actually be read."""
        if self.last_event is None:
            return False
        return (now - self.last_event.at) <= self.config.result_hold_seconds

    #  Advisory states describe the scene and can flap frame to frame, so their
    #  messages are held briefly.  Everything else is a real outcome the user is
    #  waiting on and must appear at once.
    _ADVISORY_STATES = (ScanState.IDLE, ScanState.DETECTING)

    def _transition(self, state: ScanState, now: float, result) -> None:
        previous, self.state = self.state, state
        # Smoothing applies *between* advisory messages.  Leaving a real outcome
        # must clear it at once, or the UI would still claim success while the
        # user is already holding up the next product.
        immediate = (state not in self._ADVISORY_STATES
                     or previous not in self._ADVISORY_STATES)
        self._set_message(self._message_for(state, result), now, immediate=immediate)

    #  Blur and contrast only describe *something we are looking at*; reported
    #  for an empty scene they would tell the user to steady a product that is
    #  not there.  Lighting problems are worth saying either way.
    _NEEDS_CANDIDATE = (QualityHint.BLURRY, QualityHint.LOW_CONTRAST)

    def _message_for(self, state: ScanState, result) -> str:
        """Prefer an actionable, image-derived reason over a generic status."""
        if state in (ScanState.IDLE, ScanState.DETECTING) and result is not None:
            if state is ScanState.DETECTING and result.too_far:
                return TOO_FAR_MESSAGE
            quality = result.quality
            if quality is not None and quality.hint is not QualityHint.OK:
                if quality.hint not in self._NEEDS_CANDIDATE or result.candidates:
                    return quality.message
        return STATE_MESSAGES[state]

    def _set_message(self, message: str, now: float, immediate: bool = False) -> None:
        """Hold a new message briefly before showing it, so the UI cannot flicker."""
        if message == self._hint_message:
            self._pending_message, self._pending_since = message, now
            return
        if immediate:
            self._hint_message = self._pending_message = message
            self._pending_since = now
            return
        if message != self._pending_message:
            self._pending_message, self._pending_since = message, now
            return
        if now - self._pending_since >= self.config.hint_hold_seconds:
            self._hint_message = message

    # -- cooldown / housekeeping -------------------------------------------

    def _cooldown_elapsed(self, value: str, now: float) -> bool:
        last = self._last_event_at.get(value)
        return last is None or (now - last) >= self.config.scan_cooldown_seconds

    def _in_cooldown(self, now: float) -> bool:
        if not self._last_event_at:
            return False
        return (now - max(self._last_event_at.values())) < self.config.scan_cooldown_seconds

    def _prune(self, now: float) -> None:
        """Forget stale tracks so confirmation counts reflect the current product."""
        window = max(self.config.confirmation_window_seconds * 2, 3.0)
        for value, track in list(self.tracks.items()):
            if now - track.last_seen > window:
                del self.tracks[value]
        for value, at in list(self._last_event_at.items()):
            if now - at > max(60.0, self.config.scan_cooldown_seconds * 10):
                del self._last_event_at[value]

    def _count(self, name: str) -> None:
        if self.metrics is not None:
            self.metrics.increment(name)

    # -- reporting ---------------------------------------------------------

    @property
    def message(self) -> str:
        return self._hint_message

    def status(self) -> dict:
        return {"state": self.state.value, "message": self.message,
                "active_barcode": self.active_value,
                "tracks": [track.as_dict() for track in
                           sorted(self.tracks.values(), key=lambda t: t.last_seen, reverse=True)[:3]]}
