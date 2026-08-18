"""Turn the evidence collected for a read into a single 0-100 score.

Every factor is evidence a human would also use: did it decode, does the check
digit hold, did a second decoder agree, was the code big enough, was the image
sharp, was it held straight, and did it repeat.  Weights live in
`ScannerConfig`, so a deployment can retune them against real products without
touching this logic.
"""
from dataclasses import dataclass
from enum import Enum

from services.config import DEFAULT_CONFIG, ScannerConfig


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class ConfidenceBreakdown:
    score: float
    level: ConfidenceLevel
    factors: dict[str, float]

    def as_dict(self) -> dict:
        return {"score": round(self.score, 1), "level": self.level.value,
                "factors": {name: round(value, 1) for name, value in self.factors.items()}}


def score_confidence(*, decoded: bool, checksum_valid: bool, decoder_agreement: bool,
                     size_ratio: float, sharpness: float, perspective: float,
                     agreements: int, config: ScannerConfig = DEFAULT_CONFIG) -> ConfidenceBreakdown:
    """Score one read.

    `size_ratio` is the code's long edge over the scan area width, `sharpness`
    the raw variance-of-Laplacian, `perspective` the 0-1 rectangularity of the
    detected quad and `agreements` how many frames have produced this value.
    """
    if not decoded:
        return ConfidenceBreakdown(0.0, ConfidenceLevel.LOW, {})

    factors = {
        "decode": config.weight_decode,
        "checksum": config.weight_checksum if checksum_valid else 0.0,
        "decoder_agreement": config.weight_decoder_agreement if decoder_agreement else 0.0,
        # 25% of the scan width is a comfortably readable code; more adds nothing.
        "size": config.weight_size * _ramp(size_ratio, 0.25),
        # Three times the blur threshold is "clearly sharp".
        "sharpness": config.weight_sharpness * _ramp(sharpness, config.blur_threshold * 3),
        "perspective": config.weight_perspective * max(0.0, min(1.0, perspective)),
        "temporal": config.weight_temporal * _ramp(agreements, max(1, config.confirmation_frames)),
    }
    score = max(0.0, min(100.0, sum(factors.values())))
    return ConfidenceBreakdown(score, level_for(score, config), factors)


def level_for(score: float, config: ScannerConfig = DEFAULT_CONFIG) -> ConfidenceLevel:
    if score >= config.confidence_high:
        return ConfidenceLevel.HIGH
    if score >= config.confidence_medium:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def _ramp(value: float, full: float) -> float:
    """Linear 0..1 ramp that saturates at `full`."""
    if full <= 0:
        return 1.0
    return max(0.0, min(1.0, float(value) / float(full)))
