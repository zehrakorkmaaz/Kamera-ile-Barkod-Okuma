import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from werkzeug.utils import secure_filename

from services.vision.decoding import lookup_aliases
from services.vision.verification import expected_weight_grams


class ProductValidationError(ValueError): pass
class ProductConflictError(ValueError): pass


class ProductRepository:
    def __init__(self, database, upload_folder):
        self.database, self.upload_folder = database, Path(upload_folder)

    def _connect(self):
        con = sqlite3.connect(self.database)
        con.row_factory = sqlite3.Row
        return con

    def migrate(self):
        with self._connect() as con:
            con.execute('''CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT, barcode TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL, brand TEXT, category TEXT, price REAL, weight TEXT,
                image TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)''')

    def _clean(self, data):
        barcode, name = str(data.get("barcode", "")).strip(), str(data.get("name", "")).strip()
        if not barcode or not name: raise ProductValidationError("Barkod ve ürün adı zorunludur.")
        try: price = float(data["price"]) if str(data.get("price", "")).strip() else None
        except (TypeError, ValueError): raise ProductValidationError("Fiyat sayı olmalıdır.")
        return {"barcode": barcode, "name": name, "brand": str(data.get("brand", "")).strip(),
                "category": str(data.get("category", "")).strip(), "price": price,
                "weight": str(data.get("weight", "")).strip(), "image": data.get("image")}

    def _row(self, row):
        if not row: return None
        product = dict(row)
        # Derived, not stored: lets the future load-cell check compare against a
        # number without migrating the hand-entered `weight` text column.
        product["expected_weight_g"] = expected_weight_grams(product.get("weight"))
        return product
    def get(self, barcode):
        with self._connect() as con: return self._row(con.execute("SELECT * FROM products WHERE barcode = ?", (barcode,)).fetchone())
    def find(self, barcode):
        """Look up a scanned code, accepting equivalent UPC-A/EAN-13 spellings."""
        for candidate in lookup_aliases(str(barcode or "")):
            product = self.get(candidate)
            if product: return product
        return None
    def list(self, query=""):
        with self._connect() as con:
            rows = con.execute("SELECT * FROM products WHERE barcode LIKE ? OR name LIKE ? ORDER BY name", (f"%{query}%", f"%{query}%")).fetchall()
        return [self._row(row) for row in rows]
    def create(self, data):
        data, now = self._clean(data), datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as con:
                con.execute('''INSERT INTO products (barcode,name,brand,category,price,weight,image,created_at,updated_at)
                VALUES (:barcode,:name,:brand,:category,:price,:weight,:image,:created_at,:updated_at)''', {**data, "created_at":now, "updated_at":now})
        except sqlite3.IntegrityError: raise ProductConflictError()
        return self.get(data["barcode"])
    def update(self, barcode, data):
        if not self.get(barcode): return None
        data = self._clean({**data, "barcode": barcode})
        if not data.get("image"):
            existing = self.get(barcode); data["image"] = existing["image"]
        with self._connect() as con:
            con.execute('''UPDATE products SET name=:name,brand=:brand,category=:category,price=:price,weight=:weight,image=:image,updated_at=:updated_at WHERE barcode=:barcode''',
                        {**data, "updated_at": datetime.now(timezone.utc).isoformat()})
        return self.get(barcode)
    def save_image(self, upload):
        ext = Path(secure_filename(upload.filename)).suffix.lower() or ".jpg"
        name = f"{uuid.uuid4().hex}{ext}"
        upload.save(self.upload_folder / name)
        return name
