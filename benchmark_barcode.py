#!/usr/bin/env python3
"""Measure decode success rate and latency without a camera.

    python benchmark_barcode.py                    # built-in synthetic suite
    python benchmark_barcode.py test_images/       # your own pictures
    python benchmark_barcode.py test_images/ --expect 8690530046269

With a folder, every image is decoded once and the expected value can either be
given with --expect or taken from the file name (e.g. `8690530046269_dark.jpg`).
The synthetic suite sweeps distance, angle, lighting and blur, so a change to
the pipeline can be judged against the same cases every time.
"""
import argparse
import json
from pathlib import Path
import sys
import time

from services.config import ScannerConfig
from services.test_mode import load_images
from services.vision import simulate as sim
from services.vision.pipeline import BarcodePipeline

VALUE = "8690530046269"


def synthetic_suite() -> list[tuple[str, "object", str]]:
    """Cases chosen to mirror how products are really presented to a cart."""
    cases = []
    for centimetres in (10, 15, 20, 25, 30, 40, 50):
        cases.append((f"distance-{centimetres}cm", sim.at_distance(VALUE, centimetres), VALUE))
    base = sim.at_distance(VALUE, 20)
    for degrees in (10, 20, 30, 45, 60, 90):
        cases.append((f"angle-{degrees}deg", sim.rotate(base, degrees), VALUE))
    for amount in (0.15, 0.25, 0.35):
        cases.append((f"perspective-{amount}", sim.perspective(base, amount), VALUE))
    for factor in (0.5, 0.3, 0.18):
        cases.append((f"dark-{factor}", sim.brightness(base, factor), VALUE))
    for sigma in (1.0, 1.8, 2.5):
        cases.append((f"blur-{sigma}", sim.blur(base, sigma), VALUE))
    for length in (5, 9, 15):
        cases.append((f"motion-{length}px", sim.motion_blur(base, length), VALUE))
    cases.append(("glare", sim.glare(base), VALUE))
    cases.append(("noise", sim.noise(base, 14), VALUE))
    cases.append(("combined-30cm-25deg", sim.rotate(sim.at_distance(VALUE, 30), 25), VALUE))
    cases.append(("combined-dark-tilted", sim.brightness(sim.rotate(base, 20), 0.4), VALUE))
    cases.append(("ean8", sim.scene("96385074", "EAN-8"), "96385074"))
    cases.append(("upca", sim.scene("036000291452", "UPC-A"), "036000291452"))
    return cases


def folder_suite(folder: str, expected: str | None) -> list[tuple[str, "object", str]]:
    cases = []
    for name, image in load_images(folder):
        # `8690530046269_angle.jpg` -> expected value taken from the file name.
        stem = Path(name).stem.split("_")[0].split("-")[0]
        cases.append((name, image, expected or (stem if stem.isdigit() else "")))
    return cases


def run(cases, config: ScannerConfig, verbose: bool) -> dict:
    pipeline = BarcodePipeline(config)
    if cases:
        pipeline.process(cases[0][1])           # warm up caches before timing
    results, latencies = [], []
    for name, image, expected in cases:
        started = time.perf_counter()
        result = pipeline.process(image)
        elapsed = (time.perf_counter() - started) * 1000
        latencies.append(elapsed)
        ok = bool(result.value) and (not expected or result.value == expected)
        results.append({"case": name, "expected": expected, "value": result.value,
                        "ok": ok, "wrong": bool(result.value) and bool(expected)
                        and result.value != expected, "confidence": round(result.confidence, 1),
                        "level": result.level.value, "ms": round(elapsed, 1),
                        "too_far": result.too_far, "quality": result.quality.hint.value})
        if verbose:
            mark = "OK  " if ok else ("WRONG" if results[-1]["wrong"] else "FAIL")
            print(f"  {mark} {name:24} {str(result.value):>14} "
                  f"conf={result.confidence:5.1f} {elapsed:6.1f}ms  {result.quality.hint.value}")
    return summarise(results, latencies)


def summarise(results, latencies) -> dict:
    total = len(results)
    successful = sum(1 for item in results if item["ok"])
    wrong = sum(1 for item in results if item["wrong"])
    ordered = sorted(latencies)
    return {"total": total, "successful": successful, "failed": total - successful,
            "wrong_reads": wrong,
            "success_rate": round(100 * successful / total, 1) if total else 0.0,
            "average_latency_ms": round(sum(ordered) / total, 1) if total else 0.0,
            "p95_latency_ms": round(ordered[min(total - 1, int(0.95 * (total - 1)))], 1) if total else 0.0,
            "max_latency_ms": round(ordered[-1], 1) if total else 0.0,
            "results": results}


def render(summary: dict) -> str:
    failed = [item for item in summary["results"] if not item["ok"]]
    lines = [
        f"Total images: {summary['total']}",
        f"Successful: {summary['successful']}",
        f"Failed: {summary['failed']}",
        f"Wrong reads: {summary['wrong_reads']}",
        f"Success rate: {summary['success_rate']}%",
        f"Average latency: {summary['average_latency_ms']} ms",
        f"P95 latency: {summary['p95_latency_ms']} ms",
        f"Max latency: {summary['max_latency_ms']} ms",
    ]
    if failed:
        lines.append("\nOkunamayanlar:")
        lines.extend(f"  {item['case']:24} ({item['quality']}"
                     f"{', çok uzak' if item['too_far'] else ''})" for item in failed)
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SmartCart barkod hattı ölçümü")
    parser.add_argument("folder", nargs="?", help="görüntü klasörü (yoksa sentetik test seti)")
    parser.add_argument("--expect", help="tüm görüntüler için beklenen barkod")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = ScannerConfig.from_env()
    if args.folder:
        cases = folder_suite(args.folder, args.expect)
        if not cases:
            print(f"'{args.folder}' içinde görüntü bulunamadı.")
            return 1
    else:
        cases = synthetic_suite()

    if not args.as_json and not args.quiet:
        print(f"{len(cases)} görüntü işleniyor…\n")
    summary = run(cases, config, verbose=not args.as_json and not args.quiet)
    print(json.dumps(summary, indent=2, ensure_ascii=False) if args.as_json
          else "\n" + render(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
