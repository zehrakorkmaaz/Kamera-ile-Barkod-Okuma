#!/usr/bin/env python3
"""Report what the attached camera can actually do.

    python camera_diagnostics.py [--index 0] [--measure] [--json]

Every line is verified by reading real frames, because `VideoCapture.set()`
returning True means very little on most drivers.  Use this before blaming the
software for a bad read: if the camera only grants 640x480, no amount of image
processing will read an EAN-13 from a metre away.
"""
import argparse
import json
import sys

from services.camera_device import CameraDevice
from services.config import ScannerConfig
from services.vision.simulate import barcode_pixel_width


def collect(index: int, measure: bool, config: ScannerConfig) -> dict:
    device = CameraDevice(config=config, index=index)
    if not device.open():
        return {"error": device.error or "Kamera açılamadı.", "index": index}
    try:
        report = {"index": index, "profile": device.profile.as_dict(),
                  "capabilities": device.capabilities,
                  "supported_modes": device.probe_resolutions()}
        if measure:
            report["measured_fps"] = device.measure_fps(2.0)
        report["estimated_range"] = estimate_range(device.profile.width)
        return report
    finally:
        device.close()


def estimate_range(frame_width: int) -> dict:
    """Distance at which an EAN-13 still has enough pixels per module.

    An EAN-13 carries 95 modules.  Both decoders need roughly 1.2-1.5 pixels per
    module, which is a property of the optics and the sensor, not of the code
    that reads them -- so this is the honest ceiling for a given resolution.
    """
    limits = {}
    for label, pixels_per_module in (("comfortable", 2.0), ("maximum", 1.2)):
        best = 0
        for centimetres in range(5, 201):
            if barcode_pixel_width(centimetres, frame_width) / 95 >= pixels_per_module:
                best = centimetres
        limits[f"{label}_cm"] = best
    return limits


def render(report: dict) -> str:
    if "error" in report:
        return f"Kamera {report['index']}: {report['error']}"
    profile, lines = report["profile"], []
    lines.append("Camera:")
    lines.append(f"  index {report['index']} · backend {profile['backend'] or 'bilinmiyor'}")
    lines.append("")
    lines.append(f"Resolution:\n  {profile['resolution']} (istenen {profile['requested'][0]}x{profile['requested'][1]})")
    lines.append(f"\nFPS:\n  {profile['fps']} (bildirilen)")
    if "measured_fps" in report:
        lines.append(f"  {report['measured_fps']} (ölçülen)")
    lines.append(f"\nPixel Format:\n  {profile['fourcc'] or 'bilinmiyor'}")
    autofocus = {True: "Supported", False: "Not supported", None: "Unknown"}[profile["autofocus"]]
    lines.append(f"\nAutofocus:\n  {autofocus}")

    lines.append("\nControls:")
    for name, info in sorted(report["capabilities"].items()):
        state = "Supported" if info["supported"] else "Not supported"
        lines.append(f"  {name:<20} {state:<15} {info['value'] if info['supported'] else ''}")

    lines.append("\nSupported modes (doğrulanmış):")
    for mode in report["supported_modes"]:
        exact = "" if mode["exact"] else f"  (istenen {mode['requested']})"
        lines.append(f"  {mode['width']}x{mode['height']} @ {mode['fps'] or '?'} fps"
                     f" [{mode['fourcc']}]{exact}")

    limits = report["estimated_range"]
    lines.append("\nEAN-13 okuma menzili (bu çözünürlükte, fiziksel tahmin):")
    lines.append(f"  rahat okuma  : ~{limits['comfortable_cm']} cm'e kadar")
    lines.append(f"  teorik sınır : ~{limits['maximum_cm']} cm")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SmartCart kamera tanılama")
    parser.add_argument("--index", type=int, default=None, help="kamera indeksi")
    parser.add_argument("--measure", action="store_true", help="gerçek FPS'i ölç (2 sn)")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    config = ScannerConfig.from_env()
    index = args.index if args.index is not None else config.camera_index
    report = collect(index, args.measure, config)
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.as_json else render(report))
    return 1 if "error" in report else 0


if __name__ == "__main__":
    sys.exit(main())
