#include "ROIDetector.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <set>
#include <sstream>

using namespace cv;
using namespace std;

namespace {

cv::Rect rectPct(const cv::Mat& img, float x, float y, float w, float h) {
    int X = static_cast<int>(x * img.cols);
    int Y = static_cast<int>(y * img.rows);
    int W = static_cast<int>(w * img.cols);
    int H = static_cast<int>(h * img.rows);
    return cv::Rect(X, Y, W, H);
}

static cv::Mat preprocessForFill(const cv::Mat& roiGray) {
    cv::Mat blurImg, thr;
    cv::GaussianBlur(roiGray, blurImg, cv::Size(5, 5), 0);

    // Daha agresif adaptive threshold
    cv::adaptiveThreshold(
        blurImg, thr, 255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        15, 3
    );

    // Küçük gürültüleri temizle (morfolojik opening)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
    cv::morphologyEx(thr, thr, cv::MORPH_OPEN, kernel);

    return thr;
}

static double cellFillRatio(const cv::Mat& bin, const cv::Rect& cell) {
    cv::Rect c = cell & cv::Rect(0, 0, bin.cols, bin.rows);
    if (c.width <= 0 || c.height <= 0) return 0.0;

    cv::Mat sub = bin(c);

    // ✅ Grid çizgilerinin etkisini daha fazla azalt (kenarlardan %20 kırp)
    int marginX = std::max(2, c.width / 5);
    int marginY = std::max(2, c.height / 5);
    cv::Rect inner(
        marginX, marginY,
        std::max(1, c.width - 2 * marginX),
        std::max(1, c.height - 2 * marginY)
    );
    inner &= cv::Rect(0, 0, sub.cols, sub.rows);

    cv::Mat innerCell = sub(inner);
    return static_cast<double>(cv::countNonZero(innerCell)) /
           static_cast<double>(innerCell.total());
}

/*
  ✅ TC / ÖğrenciNo gibi sayısal alanlar:
  - Her sütun = 1 hane
  - İlk satır (r=0) yazı alanı => ATLANIR
  - r=1 => 0, r=2 => 1, ... r=10 => 9
  rows burada fiziksel satır sayısıdır (11 olmalı).
*/
static std::string detectGridByColumns_AsDigits_SkipFirstRow(
    const cv::Mat& roiGray,
    int rows, int cols,
    double fillThreshold
) {
    cv::Mat thr = preprocessForFill(roiGray);

    int cellH = roiGray.rows / rows;
    int cellW = roiGray.cols / cols;

    std::string out;
    out.reserve(cols);

    for (int c = 0; c < cols; ++c) {
        int bestRow = -1;
        double bestVal = 0.0;
        double secondVal = 0.0;

        // 🔴 İLK SATIRDAN BAŞLA (r=0), çünkü formda başlık satırı grid'in dışında olabilir
        for (int r = 0; r < rows; ++r) {
            cv::Rect cell(c * cellW, r * cellH, cellW, cellH);
            double filled = cellFillRatio(thr, cell);

            if (filled > bestVal) {
                secondVal = bestVal;
                bestVal = filled;
                bestRow = r;
            } else if (filled > secondVal) {
                secondVal = filled;
            }
        }

        // ✅ Daha sıkı threshold kontrolü
        if (bestVal < fillThreshold || bestRow < 0) {
            out.push_back('-');
            continue;
        }

        // ✅ Çoklu işaret kontrolü
        if (secondVal >= fillThreshold * 0.6 && (bestVal - secondVal) < 0.08) {
            out.push_back('-');
            continue;
        }

        int digit = bestRow; // r=0 => 0, r=1 => 1, ...
        if (digit < 0 || digit > 9) {
            out.push_back('-');
            continue;
        }

        out.push_back(static_cast<char>('0' + digit));
    }

    return out;
}

/*
  ✅ Ad-Soyad:
  - Her sütun = 1 karakter
  - İlk satır (r=0) yazı alanı => ATLANIR
  - r=1 => alphabet[0], r=2 => alphabet[1], ...
  rows burada fiziksel satır sayısıdır (30 olmalı: 1 yazı + 29 harf).
*/
static std::string detectGridByColumns_AsLetters_SkipFirstRow(
    const cv::Mat& roiGray,
    int rows, int cols,
    double fillThreshold,
    const std::string& alphabet
) {
    cv::Mat thr = preprocessForFill(roiGray);

    int cellH = roiGray.rows / rows;
    int cellW = roiGray.cols / cols;

    std::string out;
    out.reserve(cols);

    for (int c = 0; c < cols; ++c) {
        int bestRow = -1;
        double bestVal = 0.0;
        double secondVal = 0.0;

        // 🔴 İLK SATIRDAN BAŞLA (r=0), çünkü formda başlık satırı grid'in dışında olabilir
        for (int r = 0; r < rows; ++r) {
            cv::Rect cell(c * cellW, r * cellH, cellW, cellH);
            double filled = cellFillRatio(thr, cell);

            if (filled > bestVal) {
                secondVal = bestVal;
                bestVal = filled;
                bestRow = r;
            } else if (filled > secondVal) {
                secondVal = filled;
            }
        }

        // ✅ Daha sıkı threshold kontrolü
        if (bestVal < fillThreshold || bestRow < 0) {
            out.push_back('-');
            continue;
        }

        // ✅ Çoklu işaret kontrolü
        if (secondVal >= fillThreshold * 0.6 && (bestVal - secondVal) < 0.08) {
            out.push_back('?');
            continue;
        }

        int alphaIndex = bestRow; // r=0 => alphabet[0] (A)
        if (alphaIndex >= 0 && alphaIndex < (int)alphabet.size())
            out.push_back(alphabet[alphaIndex]);
        else
            out.push_back('-');
    }

    return out;
}

// Tek sütunlu alanlar (oturum vb.)
static std::string detectSingleColumn(const cv::Mat& roiGray,
                                      int rows,
                                      double fillThreshold) {
    cv::Mat thr = preprocessForFill(roiGray);

    int cellH = roiGray.rows / rows;
    int cellW = roiGray.cols;

    int bestIdx = -1;
    double bestVal = 0.0;

    for (int r = 0; r < rows; ++r) {
        cv::Rect cell(0, r * cellH, cellW, cellH);
        double filled = cellFillRatio(thr, cell);
        if (filled > bestVal) {
            bestVal = filled;
            bestIdx = r;
        }
    }

    if (bestVal < fillThreshold || bestIdx < 0) return "-";
    return std::to_string(bestIdx);
}

} // namespace

ROIDetector::ROIDetector()
    : fillThreshold_(0.15),
      bubbleDetector_(fillThreshold_),
      debugMode_(false) {

    // ✅ TC: 10 satır (0..9) x 11 sütun (11 hane)
    regions_.push_back({
        "tc_kimlik",
        {0.0065f, 0.283f, 0.271f, 0.172f},
        10, 11, GRID
    });

    // ✅ Öğrenci No: 10 satır (0..9) x 10 sütun
    regions_.push_back({
        "ogrenci_no",
        {0.287f, 0.283f, 0.123f, 0.172f},
        10, 5, GRID
    });

    // ✅ Ad Soyad: 29 satır (A-Z) x 21 sütun
    regions_.push_back({
        "adi_soyadi",
        {0.000f, 0.490f, 0.520f, 0.495f},
        29, 21, GRID
    });


    
    regions_.push_back({ "turkce",    {0.525f, 0.263f, 0.125f, 0.345f}, 20, 4, GRID });
    regions_.push_back({ "sosyal",    {0.640f, 0.263f, 0.125f, 0.345f}, 20, 4, GRID });
    regions_.push_back({ "din",       {0.755f, 0.263f, 0.125f, 0.345f}, 20, 4, GRID });
    regions_.push_back({ "ingilizce", {0.874f, 0.263f, 0.125f, 0.345f}, 20, 4, GRID });

    // Matematik ve Fen (Aynı hizayı korur)
    regions_.push_back({ "matematik", {0.641f, 0.640f, 0.125f, 0.345f}, 20, 4, GRID });
    regions_.push_back({ "fen",       {0.757f, 0.640f, 0.125f, 0.345f}, 20, 4, GRID });
}

void ROIDetector::setFillThreshold(double threshold) {
    fillThreshold_ = threshold;
    bubbleDetector_.setFillThreshold(threshold);
}

void ROIDetector::setDebugMode(bool enabled) {
    debugMode_ = enabled;
}

cv::Mat ROIDetector::getLastDebugVisualization() const {
    if (lastDebugVis_.empty()) return cv::Mat();
    return lastDebugVis_.clone();
}

bool ROIDetector::isSubjectRegion(const std::string& name) const {
    static const std::set<std::string> subjects = {
        "turkce", "sosyal", "din", "ingilizce", "matematik", "fen"
    };
    return subjects.find(name) != subjects.end();
}

std::string ROIDetector::bubblesToAnswerString(const std::vector<BubbleResult>& results) const {
    std::ostringstream oss;
    
    // --- GÜNCELLEME BURADA ---
    // İstediğin gibi eşik değerini 60.0 yaptık.
    // Artık 60 puanın altındaki her şey "BOŞ" (-) sayılacak.
    const double CONFIDENCE_THRESHOLD = 60.0; 

    for (size_t i = 0; i < results.size(); ++i) {
        char mark = '-'; 
        const auto& br = results[i];

        if (!br.isValid) {
            mark = 'X'; // Okuma hatası
        } 
        else {
            // MANTIK KONTROLLERİ:
            
            // 1. Kural: En yüksek puan (confidence) 60'tan küçükse -> BOŞ (-)
            // Bu kural, "bir sorudaki bütün bubblelar 60'tan düşükse boş kabul edilsin" isteğini tam karşılar.
            // Çünkü en yüksek puanlı (seçilen) baloncuk bile 60 etmiyorsa, diğerleri zaten etmiyordur.
            if (br.confidence < CONFIDENCE_THRESHOLD) {
                mark = '-'; 
            }
            // 2. Kural: İkinci en yüksek şık da 60'ı geçtiyse -> GEÇERSİZ (-)
            // (Çift işaretleme durumu: Hem A hem B karalanmışsa)
            else if (br.secondConfidence > CONFIDENCE_THRESHOLD) {
                 mark = '-'; // İkisi de 60 üzeri -> İptal/Boş say
            }
            // 3. Kural: Puan 60'tan büyükse ve tek işaretleme varsa -> HARFİ AL
            else if (!br.markedAnswer.empty()) {
                mark = br.markedAnswer[0];
            }
        }

        if (i > 0) oss << ",";
        oss << mark;
    }
    return oss.str();
}
std::map<std::string, std::string>
ROIDetector::process(const cv::Mat& warped, cv::Mat& debugOut) {
    CV_Assert(!warped.empty());

    cv::Mat gray;
    if (warped.channels() == 3)
        cv::cvtColor(warped, gray, cv::COLOR_BGR2GRAY);
    else
        gray = warped.clone();

    if (warped.channels() == 3)
        lastDebugVis_ = warped.clone();
    else
        cv::cvtColor(warped, lastDebugVis_, cv::COLOR_GRAY2BGR);

    std::map<std::string, std::string> out;

    // ✅ ID alanları için ayrı threshold (cevap bubble'ından bağımsız)
    // Daha yüksek threshold kullan ki gürültü kabul edilmesin
    double idThr = std::clamp(fillThreshold_ * 1.2, 0.25, 0.45);

    for (const auto& reg : regions_) {
        cv::Rect roi = rectPct(gray, reg.rectPct[0], reg.rectPct[1],
                               reg.rectPct[2], reg.rectPct[3]);

        // GRID bölgelerinde border gürültüsünü azalt
        if (reg.type == GRID) {
            int dx = std::max(1, static_cast<int>(roi.width * 0.02));
            int dy = std::max(1, static_cast<int>(roi.height * 0.02));
            roi = cv::Rect(roi.x + dx, roi.y + dy,
                           std::max(1, roi.width - 2 * dx),
                           std::max(1, roi.height - 2 * dy));
        }

        if (isSubjectRegion(reg.name)) {
            // TÜM DERSLER İÇİN TEK KURAL:
            // 1. Koordinatlar (0.525 vb.) soru numarasının başladığı yerdir.
            // 2. Biz bu alanın solundan %17'lik kısmı (numaraları) atlayacağız.
            // 3. Sağ taraftan da taşma olmaması için genişliği limitleyeceğiz.

            double questionOffsetRatio = 0.19; // Soru numaralarını atlama payı
            double widthScaleFactor = 0.95;    // Baloncuk alanını netleştirmek için sağdan kırpma

            int originalW = roi.width;
            
            // Soru numaralarını geçmek için X ekseninde öteleme
            int offsetX = static_cast<int>(originalW * questionOffsetRatio);
            
            // Yeni genişlik hesabı (Ofseti düş, sağdan da biraz daralt)
            int targetTotalW = static_cast<int>(originalW * widthScaleFactor);
            int newWidth = std::max(1, targetTotalW - offsetX);

            // ROI'yi güncelle
            roi.x += offsetX;
            roi.width = newWidth;
        }

        roi &= cv::Rect(0, 0, gray.cols, gray.rows);
        if (roi.width <= 0 || roi.height <= 0) continue;

        cv::Mat sub = gray(roi).clone();
        std::string val;

        // ✅ Ders alanları: contour tabanlı bubble detector (sarı/yellow seçim)
        if (reg.type == GRID && isSubjectRegion(reg.name)) {
            
            auto bubbles = bubbleDetector_.detectBubblesWithContours(sub, reg.rows, reg.cols, 1, 'A');
            val = bubblesToAnswerString(bubbles);

            if (debugMode_) {
                bubbleDetector_.drawBubbleDebug(lastDebugVis_, roi, bubbles, reg.rows, reg.cols, reg.name);
            }
        }
        // ✅ TC: sütun bazlı digit, ilk satır atla
        else if (reg.name == "tc_kimlik") { // SADECE TC_KIMLIK İÇİN
            
            std::string resultString = "";
            int rows = reg.rows; // 10 (0-9)
            int cols = reg.cols; // 11 (Haneler)
            
            int cellW = sub.cols / cols;
            int cellH = sub.rows / rows;

            cv::Mat workingImg = sub.clone();

            // 1. Yumuşatma
            cv::GaussianBlur(workingImg, workingImg, cv::Size(5, 5), 0);
            
            // 2. YÖNTEM A: ADAPTIVE (Detaycı)
            cv::Mat adaptiveBin;
            cv::adaptiveThreshold(workingImg, adaptiveBin, 255,
                                  cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv::THRESH_BINARY_INV,
                                  21, 15); 

            // 3. YÖNTEM B: GLOBAL MASK (Kesin Filtre)
            cv::Mat globalBin;
            cv::threshold(workingImg, globalBin, 160, 255, cv::THRESH_BINARY_INV);

            // 4. KESİŞİM (AND) - GÜRÜLTÜYÜ SİL
            cv::Mat finalBin;
            cv::bitwise_and(adaptiveBin, globalBin, finalBin);
            
            // 5. TEMİZLİK (Çizgileri kopar)
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::morphologyEx(finalBin, finalBin, cv::MORPH_OPEN, kernel);

            // --- SÜTUNLARI GEZ ---
            for (int c = 0; c < cols; ++c) {
                
                double bestVal = 0.0;
                int bestRow = -1;

                // --- SATIRLARI GEZ ---
                for (int r = 0; r < rows; ++r) {
                    
                    // Kenar paylarını biraz artırdık (%30)
                    int marginX = static_cast<int>(cellW * 0.30);
                    int marginY = static_cast<int>(cellH * 0.30);
                    
                    cv::Rect cell(c * cellW + marginX, r * cellH + marginY, 
                                  cellW - 2*marginX, cellH - 2*marginY);
                    
                    cell &= cv::Rect(0, 0, finalBin.cols, finalBin.rows);
                    if (cell.width <= 0 || cell.height <= 0) continue;

                    cv::Mat binCell = finalBin(cell);
                    
                    double ratio = (double)cv::countNonZero(binCell) / (cell.width * cell.height);

                    if (ratio > bestVal) {
                        bestVal = ratio;
                        bestRow = r;
                    }

                    // --- DEBUG: GRİ YUVARLAKLAR + YEŞİL PUAN ---
                    if (debugMode_) {
                        int centerX = roi.x + (c * cellW) + (cellW / 2);
                        int centerY = roi.y + (r * cellH) + (cellH / 2);
                        int radius = std::min(cellW, cellH) * 0.35;

                        // Gri Daire
                        cv::circle(lastDebugVis_, cv::Point(centerX, centerY), radius, 
                                   cv::Scalar(100, 100, 100), 1, cv::LINE_AA);
                        
                        // YEŞİL/BÜYÜK PUAN
                        if (ratio > 0.05) {
                            std::string scoreTxt = std::to_string((int)(ratio * 100));
                            cv::putText(lastDebugVis_, scoreTxt,
                                       cv::Point(centerX - 10, centerY + 5), 
                                       cv::FONT_HERSHEY_DUPLEX, 0.40, cv::Scalar(0, 255, 0), 1);
                        }
                    }
                }

                // --- KARAR ---
                // Temizlenmiş resimde güvenli eşik: 0.30.
                double THRESHOLD = 0.30; 
                char detectedChar = '-'; 
                
                if (bestVal > THRESHOLD && bestRow != -1) {
                    detectedChar = '0' + bestRow;

                    if (debugMode_) {
                        cv::Rect finalCell(roi.x + c * cellW, roi.y + bestRow * cellH, cellW, cellH);
                        
                        // Seçilen Kare
                        cv::rectangle(lastDebugVis_, finalCell, cv::Scalar(0, 255, 0), 2);
                        
                        // Rakamı Yaz
                        std::string charTxt(1, detectedChar);
                        cv::putText(lastDebugVis_, charTxt,
                                   cv::Point(finalCell.x + 5, finalCell.y + finalCell.height - 5),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(0, 255, 0), 2);
                    }
                }
                
                resultString += detectedChar;
            }
            
            val = resultString;
        }
        else if (reg.name == "ogrenci_no") { 
            
            std::string resultString = "";
            int rows = reg.rows; // 10 (0-9)
            int cols = reg.cols; // 5 (Haneler)
            
            int cellW = sub.cols / cols;
            int cellH = sub.rows / rows;

            cv::Mat workingImg = sub.clone();

            // 1. Yumuşatma (Gürültüyü azalt)
            cv::GaussianBlur(workingImg, workingImg, cv::Size(7, 7), 0);
            
            // 2. YÖNTEM A: ADAPTIVE (Daha seçici)
            cv::Mat adaptiveBin;
            cv::adaptiveThreshold(workingImg, adaptiveBin, 255,
                                  cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv::THRESH_BINARY_INV,
                                  25, 20); // Kritik ayarlar

            // 3. YÖNTEM B: GLOBAL MASK (Katı Filtre)
            cv::Mat globalBin;
            cv::threshold(workingImg, globalBin, 180, 255, cv::THRESH_BINARY_INV);

            // 4. KESİŞİM (AND) - GÜRÜLTÜYÜ SİL
            cv::Mat finalBin;
            cv::bitwise_and(adaptiveBin, globalBin, finalBin);
            
            // 5. TEMİZLİK (Çizgileri kopar - Morphological Open)
            cv::Mat kernel_morph = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::morphologyEx(finalBin, finalBin, cv::MORPH_OPEN, kernel_morph);

            // 6. TEKRAR EROSION (Ek Aşındırma)
            cv::Mat kernel_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
            cv::erode(finalBin, finalBin, kernel_erode, cv::Point(-1, -1), 1);

            // --- SÜTUNLARI GEZ ---
            for (int c = 0; c < cols; ++c) {
                
                double bestVal = 0.0;
                int bestRow = -1;

                // --- SATIRLARI GEZ ---
                for (int r = 0; r < rows; ++r) {
                    
                    // Kenar paylarını artırdık (%30)
                    int marginX = static_cast<int>(cellW * 0.30);
                    int marginY = static_cast<int>(cellH * 0.30);
                    
                    cv::Rect cell(c * cellW + marginX, r * cellH + marginY, 
                                  cellW - 2*marginX, cellH - 2*marginY);
                    
                    cell &= cv::Rect(0, 0, finalBin.cols, finalBin.rows);
                    if (cell.width <= 0 || cell.height <= 0) continue;

                    cv::Mat binCell = finalBin(cell);
                    
                    double ratio = (double)cv::countNonZero(binCell) / (cell.width * cell.height);

                    if (ratio > bestVal) {
                        bestVal = ratio;
                        bestRow = r;
                    }

                    // --- DEBUG: GRİ YUVARLAKLAR + YEŞİL PUAN ---
                    if (debugMode_) {
                        int centerX = roi.x + (c * cellW) + (cellW / 2);
                        int centerY = roi.y + (r * cellH) + (cellH / 2);
                        int radius = std::min(cellW, cellH) * 0.35;

                        // Gri Daire
                        cv::circle(lastDebugVis_, cv::Point(centerX, centerY), radius, 
                                   cv::Scalar(100, 100, 100), 1, cv::LINE_AA);
                        
                        // YEŞİL/BÜYÜK PUAN (Sadece kayda değer doluluk varsa yaz)
                        if (ratio > 0.05) {
                            std::string scoreTxt = std::to_string((int)(ratio * 100));
                            cv::putText(lastDebugVis_, scoreTxt,
                                       cv::Point(centerX - 10, centerY + 5), 
                                       cv::FONT_HERSHEY_DUPLEX, 0.40, cv::Scalar(0, 255, 0), 1);
                        }
                    }
                }

                // --- KARAR ---
                // Eşik değeri 0.20 olarak kalıyor. Bu, temizlenmiş resimde güvenilir bir değerdir.
                double THRESHOLD = 0.10; 
                char detectedChar = '-'; 
                
                if (bestVal > THRESHOLD && bestRow != -1) {
                    detectedChar = '0' + bestRow;

                    if (debugMode_) {
                        cv::Rect finalCell(roi.x + c * cellW, roi.y + bestRow * cellH, cellW, cellH);
                        
                        // Seçilen Kare
                        cv::rectangle(lastDebugVis_, finalCell, cv::Scalar(0, 255, 0), 2);
                        
                        // Rakamı Yaz
                        std::string charTxt(1, detectedChar);
                        cv::putText(lastDebugVis_, charTxt,
                                   cv::Point(finalCell.x + 5, finalCell.y + finalCell.height - 5),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(0, 255, 0), 2);
                    }
                }
                
                resultString += detectedChar;
            }
            
            val = resultString;
        }

        else if (reg.name == "adi_soyadi") {
            
            std::vector<std::string> TR_CHARS = {
                "A","B","C","C","D","E","F","G","G","H","I","I","J","K","L","M",
                "N","O","O","P","R","S","S","T","U","U","V","Y","Z"
            };

            std::string resultString = "";
            int rows = reg.rows; 
            int cols = reg.cols; 
            
            int cellW = sub.cols / cols;
            int cellH = sub.rows / rows;

            cv::Mat workingImg = sub.clone();
            
            // 1. Yumuşatma
            cv::GaussianBlur(workingImg, workingImg, cv::Size(5, 5), 0);

            // 2. YÖNTEM A: ADAPTIVE (Detaycı)
            // G satırını ve silik işaretleri yakalar.
            // C değerini 25 yaptık (Daha seçici olsun diye).
            cv::Mat adaptiveBin;
            cv::adaptiveThreshold(workingImg, adaptiveBin, 255,
                                  cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv::THRESH_BINARY_INV,
                                  31, 25);

            // 3. YÖNTEM B: GLOBAL MASK (Filtre)
            // Kağıdın boş yerlerini (beyaz/açık gri) kesinlikle eler.
            // 160 değeri: 255(Beyaz) ile 0(Siyah) arasında orta-açık gri bir sınırdır.
            // Bunun üzerindeki (daha beyaz) her şeyi yok sayar.
            cv::Mat globalBin;
            cv::threshold(workingImg, globalBin, 160, 255, cv::THRESH_BINARY_INV);

            // 4. KESİŞİM (AND) - SİHİRLİ DOKUNUŞ
            // Bir pikselin işaret sayılması için HEM Adaptive (çevresinden koyu)
            // HEM DE Global (gerçekten koyu) olması gerekir.
            // Bu işlem boş sütunlardaki gürültüyü %100 temizler.
            cv::Mat finalBin;
            cv::bitwise_and(adaptiveBin, globalBin, finalBin);

            // 5. TEMİZLİK (Çizgileri kopar)
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::morphologyEx(finalBin, finalBin, cv::MORPH_OPEN, kernel);

            // --- SÜTUNLARI GEZ ---
            for (int c = 0; c < cols; ++c) {
                
                double bestVal = 0.0;
                int bestRow = -1;

                // --- SATIRLARI GEZ ---
                for (int r = 0; r < rows; ++r) {
                    
                    int marginX = static_cast<int>(cellW * 0.30);
                    int marginY = static_cast<int>(cellH * 0.30);
                    
                    cv::Rect cell(c * cellW + marginX, r * cellH + marginY, 
                                  cellW - 2*marginX, cellH - 2*marginY);
                    
                    cell &= cv::Rect(0, 0, finalBin.cols, finalBin.rows);
                    if (cell.width <= 0 || cell.height <= 0) continue;

                    cv::Mat binCell = finalBin(cell);
                    
                    double ratio = (double)cv::countNonZero(binCell) / (cell.width * cell.height);

                    if (ratio > bestVal) {
                        bestVal = ratio;
                        bestRow = r;
                    }

                    // DEBUG: Gri Daire + Yeşil Puan
                    if (debugMode_) {
                        int centerX = roi.x + (c * cellW) + (cellW / 2);
                        int centerY = roi.y + (r * cellH) + (cellH / 2);
                        int radius = std::min(cellW, cellH) * 0.35;

                        cv::circle(lastDebugVis_, cv::Point(centerX, centerY), radius, 
                                   cv::Scalar(100, 100, 100), 1, cv::LINE_AA);
                        
                        if (ratio > 0.05) { // Sadece %5 üstü doluluk varsa yaz
                            std::string scoreTxt = std::to_string((int)(ratio * 100));
                            cv::putText(lastDebugVis_, scoreTxt,
                                       cv::Point(centerX - 10, centerY + 5), 
                                       cv::FONT_HERSHEY_DUPLEX, 0.40, cv::Scalar(0, 255, 0), 1);
                        }
                    }
                }

                // --- KARAR ---
                // Melez yöntem sayesinde boş yerler tertemiz (0 puan) çıkar.
                // Gerçek işaretler ise 30-70 arası çıkar.
                // Eşiği 0.20 (%20) yapmak çok güvenlidir.
                double THRESHOLD = 0.40; 
                std::string detectedChar = " "; 
                
                if (bestVal > THRESHOLD && bestRow != -1 && bestRow < (int)TR_CHARS.size()) {
                    detectedChar = TR_CHARS[bestRow];

                    if (debugMode_) {
                        cv::Rect finalCell(roi.x + c * cellW, roi.y + bestRow * cellH, cellW, cellH);
                        cv::rectangle(lastDebugVis_, finalCell, cv::Scalar(0, 255, 0), 2);
                        cv::putText(lastDebugVis_, detectedChar,
                                   cv::Point(finalCell.x + 5, finalCell.y + finalCell.height - 5),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(0, 255, 0), 2);
                    }
                }
                
                resultString += detectedChar;
            }
            
            // Sağdaki boşlukları (trailing spaces) temizle
            // Örneğin "AHMET YUNUS      " -> "AHMET YUNUS"
            size_t lastChar = resultString.find_last_not_of(' ');
            if (lastChar != std::string::npos) {
                resultString = resultString.substr(0, lastChar + 1);
            } else {
                resultString = ""; // Tamamen boşsa
            }

            val = resultString;
        }
        else {
            val = detectSingleColumn(sub, reg.rows, idThr);
        }

        out[reg.name] = val;

        cv::rectangle(lastDebugVis_, roi, cv::Scalar(0, 255, 0), 2);
        cv::putText(lastDebugVis_, reg.name, roi.tl() + cv::Point(4, 16),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
    }

    debugOut = lastDebugVis_.clone();
    return out;
}
