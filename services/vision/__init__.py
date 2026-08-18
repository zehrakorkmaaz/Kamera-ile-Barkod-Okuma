"""Camera vision pipeline for SmartCart.

The modules are ordered the way a frame travels through them:

    quality      -> is this frame worth spending CPU on, and why not?
    detection    -> where in the frame could a 1D barcode be?
    preprocess   -> rectify that region and produce decoding inputs
    decoding     -> run the decoders, normalise and checksum the result
    confidence   -> how much do we trust this read?
    tracking     -> temporal confirmation + scan state machine
    verification -> combine vision with (future) load-cell evidence
    pipeline     -> orchestrates all of the above for one frame
    metrics      -> counters and latencies for the whole scanner

Nothing in here imports Flask, sqlite3 or the camera device, so every stage can
be unit-tested from a plain numpy image.
"""
from services.vision.confidence import ConfidenceBreakdown, ConfidenceLevel, score_confidence
from services.vision.decoding import (
    DecodeResult, decode_image, decode_with_opencv, is_valid_barcode,
    lookup_aliases, normalize_barcode,
)
from services.vision.detection import Candidate, detect_candidates
from services.vision.metrics import ScanMetrics
from services.vision.pipeline import BarcodePipeline, FrameResult
from services.vision.quality import FrameQuality, QualityHint, analyse_quality
from services.vision.tracking import ScanEvent, ScanState, ScanTracker, TrackedBarcode
from services.vision.verification import (
    FinalConfidence, NullWeightSource, VisionSignal, WeightSignal, combine_confidence,
)

__all__ = [
    "BarcodePipeline", "Candidate", "ConfidenceBreakdown", "ConfidenceLevel",
    "DecodeResult", "FinalConfidence", "FrameQuality", "FrameResult", "NullWeightSource",
    "QualityHint", "ScanEvent", "ScanMetrics", "ScanState", "ScanTracker",
    "TrackedBarcode", "VisionSignal", "WeightSignal", "analyse_quality",
    "combine_confidence", "decode_image", "decode_with_opencv", "detect_candidates",
    "is_valid_barcode", "lookup_aliases", "normalize_barcode", "score_confidence",
]
