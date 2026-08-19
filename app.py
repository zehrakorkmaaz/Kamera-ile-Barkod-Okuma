import logging
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, Response, send_from_directory

from services.camera import CameraService
from services.config import ScannerConfig
from services.products import ProductRepository, ProductValidationError, ProductConflictError
from services.vision.decoding import MAX_BARCODE_LENGTH, normalize_barcode


BASE_DIR = Path(__file__).resolve().parent


def _frame_source(config):
    """Replay a folder of images instead of opening a camera, when asked to.

        SMARTCART_TEST_IMAGE_DIR=test_images python app.py
    """
    if not config.test_image_dir:
        return None
    from services.test_mode import FolderFrameSource
    logging.getLogger("smartcart.camera").info("TEST_MODE folder=%s", config.test_image_dir)
    return FolderFrameSource(folder=config.test_image_dir, config=config)


def create_app(test_config=None):
    app = Flask(__name__)
    scanner_config = ScannerConfig.from_env()
    app.config.from_mapping(
        DATABASE=str(BASE_DIR / "smartcart.db"),
        UPLOAD_FOLDER=str(BASE_DIR / "uploads"),
        CAMERA_INDEX=scanner_config.camera_index,
        SCANNER_CONFIG=scanner_config,
    )
    if test_config:
        app.config.update(test_config)

    logging.basicConfig(
        level=os.environ.get("SMARTCART_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    products = ProductRepository(app.config["DATABASE"], app.config["UPLOAD_FOLDER"])
    products.migrate()
    # The camera looks products up itself so a confirmed scan arrives at the UI
    # complete, instead of costing an extra round trip.
    camera = CameraService(app.config["CAMERA_INDEX"], config=app.config["SCANNER_CONFIG"],
                           lookup=products.find, device=_frame_source(app.config["SCANNER_CONFIG"]))
    app.extensions["products"] = products
    app.extensions["camera"] = camera

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/products")
    def list_products():
        return jsonify(products.list(request.args.get("q", "")))

    @app.get("/api/products/<barcode>")
    def get_product(barcode):
        product = products.find(barcode)
        if not product:
            return jsonify({"error": "Bu barkoda ait ürün bulunamadı."}), 404
        return jsonify(product)

    @app.post("/api/products")
    def create_product():
        try:
            return jsonify(products.create(_product_payload())), 201
        except ProductConflictError:
            return jsonify({"error": "Bu barkoda ait ürün zaten kayıtlı."}), 409
        except ProductValidationError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.put("/api/products/<barcode>")
    def update_product(barcode):
        try:
            product = products.update(barcode, _product_payload())
        except ProductValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        if not product:
            return jsonify({"error": "Bu barkoda ait ürün bulunamadı."}), 404
        return jsonify(product)

    @app.post("/api/scan")
    def scan_lookup():
        data = request.get_json(silent=True) or {}
        barcode = normalize_barcode(str(data.get("barcode", "")))
        if not barcode:
            return jsonify({"error": "Barkod gerekli."}), 400
        if len(barcode) > MAX_BARCODE_LENGTH or not barcode.isprintable():
            return jsonify({"error": "Geçersiz barkod."}), 400
        product = products.find(barcode)
        return jsonify({"found": product is not None, "barcode": barcode, "product": product})

    @app.post("/api/camera/start")
    def start_camera():
        if camera.start():
            return jsonify({"started": True})
        return jsonify({"error": camera.error or f"Camera {camera.index} could not be opened."}), 503

    @app.post("/api/camera/stop")
    def stop_camera():
        camera.stop()
        return jsonify({"stopped": True})

    @app.get("/api/camera/status")
    def camera_status():
        return jsonify(camera.status())

    @app.get("/api/camera/debug")
    def camera_debug():
        return jsonify(camera.debug_status())

    @app.get("/api/camera/metrics")
    def camera_metrics():
        return jsonify(camera.metrics.snapshot())

    @app.get("/api/camera/stream")
    def camera_stream():
        return Response(camera.mjpeg_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/uploads/<path:filename>")
    def uploaded_file(filename):
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

    def _product_payload():
        # Accept JSON and standard multipart form submissions (for a camera photo).
        data = request.form.to_dict() if request.form else (request.get_json(silent=True) or {})
        image = request.files.get("image")
        if image and image.filename:
            data["image"] = products.save_image(image)
        return data

    return app


if __name__ == "__main__":
    app = create_app()
    # AVFoundation needs to open the camera from Python's main thread in order to
    # trigger macOS's permission flow.  Later start requests remain harmless.
    app.extensions["camera"].start()
    # Port 5000 is used by macOS Control Center/AirPlay on some installations.
    # Use a dedicated local port so requests cannot intermittently reach that service.
    # threaded=True is required: /api/camera/stream is a long-lived MJPEG
    # response, and Werkzeug's dev server is single-threaded by default, so
    # without this every other endpoint (status polling, /api/scan, product
    # CRUD) would queue behind the open stream connection.
    app.run(host="127.0.0.1", port=5050, debug=True, use_reloader=False, threaded=True)
