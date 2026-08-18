# SmartCart Kamera Barkod Prototipi

Mac'in dahili kamerası (veya `CAMERA_INDEX` değiştirilerek USB kamera) üzerinden barkod okur; ürünleri SQLite veritabanına tanıtır, bulur ve günceller. Tek uygulama, tek `products` tablosu kullanır.

## Gereksinimler

- macOS, Apple Silicon
- Python 3.10+ (bu proje Python 3.14 ile de test edildi)
- Kamera erişimi

Kurulum:

```bash
cd "/Users/zehrakorkmaz/Desktop/kamera "
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`zxing-cpp`, EAN-13/EAN-8/UPC-A/UPC-E/Code 128 ve QR çözümü için kullanılır; `zbar` veya fiziksel barkod okuyucu gerektirmez.

## macOS kamera izni

İlk kamera denemesinde macOS izin penceresi görüntülenir. Reddedilmişse **Sistem Ayarları > Gizlilik ve Güvenlik > Kamera** içinden Terminal (veya uygulamayı başlattığınız IDE) için izni etkinleştirin; sonra uygulamayı yeniden başlatın. Kamera ekranında görüntü yoksa başka bir uygulamanın kamerayı kullanmadığından emin olun.

macOS AVFoundation izin penceresinin güvenilir biçimde açılabilmesi için kamera uygulama başlarken ana iş parçacığında hazırlanır; tarama ekranına girildiğinde canlı akış gösterilir.

## Başlatma

```bash
source .venv/bin/activate
python app.py
```

Tarayıcıdan [http://127.0.0.1:5050](http://127.0.0.1:5050) adresini açın. macOS'ta AirPlay Receiver bazen `5000` portunu kullandığından SmartCart çakışmayı önlemek için `5050` üzerinde çalışır.

- **Ürün Tanıt:** Barkod algılanınca alan otomatik dolar. Yeni ürünü kaydedin; kayıtlı barkodda form mevcut bilgiyle gelir ve güncelleme yapılır.
- **Barkod Oku:** Barkod kamerada bulunur, veritabanında aranır; kayıt yoksa **Bu Ürünü Tanıt** ile aynı barkod aktarılır.
- **Ürünler:** Barkoda veya ürün adına göre arama yapın.

Ürün fotoğrafı seçimi isteğe bağlıdır; uygulama `uploads/` içine kaydeder ve yolu `products.image` alanında tutar.

## API

- `GET /api/products?q=` — ürün listesi / arama
- `GET /api/products/{barcode}` — tek ürün
- `POST /api/products` — ürün oluşturma (`barcode` benzersiz)
- `PUT /api/products/{barcode}` — ürün güncelleme
- `POST /api/scan` — `{ "barcode": "8691234567890" }` ile ürün arama

## Testler

```bash
source .venv/bin/activate
pytest -q
```

Otomatik testler ürün oluşturma, benzersiz barkod kontrolü, arama, güncelleme, kayıtlı/kayıtsız barkod yanıtı ve debounce davranışını kontrol eder. Fiziksel kamera ve gerçek EAN-13 testi, bu bilgisayarda izin/kamera donanımı gerektirdiğinden uygulamayı başlattıktan sonra gerçek ürünle yapılmalıdır.

## Sorun giderme

- **Kamera açılamadı:** macOS kamera iznini ve başka uygulamanın kamerayı kullanıp kullanmadığını kontrol edin.
- **Barkod algılanamadı:** Barkodu düz, yakın ve iyi aydınlatılmış biçimde tarama alanına tutun; parlama oluşmasını engelleyin.
- **USB kamera:** `app.py` içindeki `CAMERA_INDEX=0` değerini, OpenCV'nin USB kameranıza atadığı numarayla değiştirin (çoğu zaman `1`).

Kamera 1280×720 hedef çözünürlükte çalışır. Görüntü yakalama (MJPEG önizleme) ve barkod tarama artık iki ayrı iş parçacığında çalışır: yakalama thread'i sadece kare okur ve önizlemeyi günceller; tarama thread'i her zaman **en güncel kareyi** işler (birikmiş eski kareleri sırayla işlemez), böylece tarama gecikmesi kare sayısına değil çözümleme süresine bağlıdır. Ekrandaki görünür tarama kutusu, `object-fit: contain` ve gerçek kamera en-boy oranı dikkate alınarak gerçek ROI'ye çevrilir. ROI içindeki olası barkod bölgeleri küçükse 2×/3×/4× büyütülmüş/iyileştirilmiş kopyalarla denenir; yakın ve büyük barkodlarda bu ağır fallback çalışmaz.

Checksum'dan geçen bir barkod **ilk okumada** anında olay üretir (market barkod okuyucusu davranışı) — zxing-cpp zaten sembolü doğruluyor, `is_valid_barcode` da EAN/UPC checksum'ını ayrıca kontrol ediyor, bu yüzden ikinci bir teyit okuması beklenmiyor. Aynı barkod ekranda kalmaya devam ettiği sürece tekrar olay üretilmez; barkod birkaç kare boyunca okunamayınca (üründen uzaklaşma/çıkarma) "aktif" durum sıfırlanır ve aynı ürün tekrar gösterildiğinde yeniden bir olay üretilir. Flask geliştirme sunucusu `threaded=True` ile çalışır; aksi halde açık kalan MJPEG bağlantısı diğer tüm API isteklerini (durum sorgulama, ürün CRUD) bloke edebilirdi. Frontend, kamera durumunu 100 ms'de bir sorgular (`/api/camera/status`).

## Tarama tanılama modu

Kamera açıkken aşağıdaki adres, gerçek kamera çözünürlüğünü, önizleme FPS'ini, tarama thread'inin saniyedeki tarama sayısını (`scan_fps`), son çözümleme süresini (`last_decode_ms`), tarama alanının koordinatlarını, aday bölge sayısını, son decoder sonucunu ve kullanılan ön işleme yolunu gösterir:

```text
http://127.0.0.1:5050/api/camera/debug
```

Tarama kutusu gerçek bir ROI'dir. Önce bu alanın ham, gri tonlu ve CLAHE sürümleri; ardından burada bulunan en fazla iki barkod adayının 2×/3×/4× büyütülmüş, keskinleştirilmiş ve adaptif eşikli sürümleri denenir. Kullanıcıya gösterilen görüntü üzerinde yatay çevirme uygulanmaz.
