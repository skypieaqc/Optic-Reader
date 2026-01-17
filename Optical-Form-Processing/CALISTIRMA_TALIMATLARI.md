# OMR Projesi Çalıştırma Talimatları

## Hızlı Başlangıç

### 1. Projeyi Derleme

```bash
cd /Users/meryemdurdu/Downloads/OMR_PROJECT/Optical-Form-Processing
mkdir -p build_macos
cd build_macos
cmake ..
make
```

### 2. Projeyi Çalıştırma

Varsayılan kamera (kamera 0) ile çalıştırma:
```bash
./omr
```

Farklı bir kamera ile çalıştırma (örneğin kamera 1):
```bash
./omr 1
```

## Klavye Kısayolları

Program çalışırken kullanabileceğiniz tuşlar:

- **ESC**: Programdan çıkış
- **d / D**: Debug görünümünü aç/kapat
- **a / A**: Detaylı analiz görünümünü aç/kapat
- **b / B**: Bubble detection debug modunu aç/kapat
- **s / S**: Warped görüntüyü kaydet
- **+ / =**: Doluluk eşiğini artır (0.05 adımlarla)
- **- / _**: Doluluk eşiğini azalt (0.05 adımlarla)
- **t / T**: Doluluk eşiğini manuel olarak ayarla

## Gereksinimler

- **OpenCV 4.x**: `brew install opencv` ile yüklenebilir
- **CMake 3.16+**: `brew install cmake` ile yüklenebilir
- **C++17 uyumlu derleyici**: macOS'ta varsayılan olarak yüklüdür

## Sorun Giderme

### Kamera açılmıyorsa:
- Farklı bir kamera indeksi deneyin: `./omr 1` veya `./omr 2`
- Kamera izinlerini kontrol edin (Sistem Ayarları > Gizlilik ve Güvenlik > Kamera)

### Derleme hatası alıyorsanız:
- OpenCV'nin doğru yüklendiğinden emin olun: `pkg-config --modversion opencv4`
- Build klasörünü temizleyip yeniden deneyin:
  ```bash
  rm -rf build_macos
  mkdir build_macos && cd build_macos
  cmake .. && make
  ```

## Notlar

- Program, optik formu kamera ile görüntüleyerek canlı puanlama yapar
- Cevap anahtarı `main.cpp` dosyasında hardcoded olarak tanımlanmıştır
- Form boyutu: 1600x2200 piksel olarak ayarlanmıştır

## Yeni Özellikler (Güncel)

### Geliştirilmiş Bubble Detection

**1. Basitleştirilmiş Algılama**
- Otsu threshold ile otomatik eşik belirleme
- Sadece siyah (dolu) pikselleri sayma
- Daha az yanlış pozitif

**2. Temporal Smoothing (Kararlı Skor)**
- Son 7 frame'den majority voting
- %70 consensus gereksinimi (daha stabil)
- Score'un sürekli değişmesi engellendi
- Gerçek skoru daha kolay yakalarsınız

**3. Optimize Edilmiş Parametreler**
- fillThreshold: 0.20 (varsayılan)
- minSeparation: 0.20 (yanlış algılamayı azaltır)
- historySize: 7 frame (daha uzun stabilizasyon)

### Ayarlama İpuçları

**Bubble'lar algılanmıyorsa:**
- `-` tuşu ile threshold'u düşürün (örn: 0.15)
- `b` tuşu ile bubble debug'u açın ve doluluk değerlerini görün

**Yanlış bubble'lar algılanıyorsa:**
- `+` tuşu ile threshold'u yükseltin (örn: 0.25)
- Score stabilizasyonu zaten açık (temporal smoothing)

**Score hala titreşiyorsa:**
- Temporal smoothing artık varsayılan olarak AÇIK
- Daha uzun beklemeniz yeterli (7 frame consensus)

