#include "core/BubbleDetector.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>

using namespace cv;
using namespace std;

BubbleDetector::BubbleDetector(double fillThreshold)
    : fillThreshold_(fillThreshold),
      minSeparation_(10.0),
      temporalSmoothingEnabled_(false),
      historySize_(5)
{
}

double BubbleDetector::calculateFillRatio(const cv::Mat& bubbleImg) {
    if (bubbleImg.empty()) return 0.0;
    int total = bubbleImg.total();
    if (total == 0) return 0.0;
    int nonZero = cv::countNonZero(bubbleImg);
    return (double)nonZero / total;
}

cv::Rect BubbleDetector::refineBubbleRect(const cv::Mat& cellPatch, const cv::Rect& initialRect) {
    // Basit bir merkezleme (center of mass) yapılabilir
    // Şimdilik olduğu gibi döndürüyoruz, eski kodunda böyleydi muhtemelen
    return initialRect;
}

std::vector<BubbleResult> BubbleDetector::detectBubblesGridCore(
    const cv::Mat& roiGray,
    int rows,
    int cols,
    int startQuestionNumber,
    char firstLabel,
    bool applySmoothing,
    std::vector<std::vector<double>>* cellFillRatios)
{
    // 1. Thresholding (Adaptive)
    cv::Mat blurImg, thr;
    cv::GaussianBlur(roiGray, blurImg, cv::Size(5, 5), 0);
    cv::adaptiveThreshold(blurImg, thr, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 15, 3);

    // Gürültü temizliği
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(thr, thr, cv::MORPH_OPEN, kernel);

    std::vector<BubbleResult> results;
    int cellW = roiGray.cols / cols;
    int cellH = roiGray.rows / rows;

    for (int r = 0; r < rows; ++r) {
        double bestVal = 0.0;
        double secondVal = 0.0;
        int bestIdx = -1;

        for (int c = 0; c < cols; ++c) {
            // Kenar payları
            int marginX = static_cast<int>(cellW * 0.15);
            int marginY = static_cast<int>(cellH * 0.15);
            cv::Rect cell(c * cellW + marginX, r * cellH + marginY, cellW - 2*marginX, cellH - 2*marginY);
            
            cell &= cv::Rect(0, 0, thr.cols, thr.rows);
            if (cell.width <= 0 || cell.height <= 0) continue;

            double ratio = calculateFillRatio(thr(cell));

            if (ratio > bestVal) {
                secondVal = bestVal;
                bestVal = ratio;
                bestIdx = c;
            } else if (ratio > secondVal) {
                secondVal = ratio;
            }
        }

        BubbleResult res;
        res.questionNumber = startQuestionNumber + r;
        res.confidence = bestVal * 100.0;
        res.secondConfidence = secondVal * 100.0;

        if (bestIdx != -1) {
            res.markedAnswer = std::string(1, firstLabel + bestIdx);
            res.isValid = true;
        } else {
            res.markedAnswer = "-";
            res.isValid = false;
        }
        results.push_back(res);
    }
    return results;
}

std::vector<BubbleResult> BubbleDetector::detectBubbles(
    const cv::Mat& roiGray,
    int rows,
    int cols,
    int startQuestionNumber,
    char firstLabel)
{
    return detectBubblesGridCore(roiGray, rows, cols, startQuestionNumber, firstLabel, false, nullptr);
}

std::vector<BubbleResult> BubbleDetector::detectBubblesWithContours(
    const cv::Mat& roiGray,
    int rows,
    int cols,
    int startQuestionNumber,
    char firstLabel,
    cv::Mat* debugVis)
{
    // Grid Core fonksiyonunu çağırıyoruz
    auto results = detectBubblesGridCore(roiGray, rows, cols, startQuestionNumber, firstLabel, false, nullptr);

    if (debugVis) {
        if (debugVis->empty() || debugVis->size() != roiGray.size()) {
            if (roiGray.channels() == 1) cv::cvtColor(roiGray, *debugVis, cv::COLOR_GRAY2BGR);
            else roiGray.copyTo(*debugVis);
        }
        drawBubbleDebug(*debugVis, cv::Rect(0,0,roiGray.cols, roiGray.rows), results, rows, cols, "");
    }
    return results;
}

void BubbleDetector::drawBubbleDebug(
    cv::Mat& debugImg,
    const cv::Rect& roi,
    const std::vector<BubbleResult>& results,
    int rows,
    int cols,
    const std::string& label)
{
    cv::Rect safeRoi = roi & cv::Rect(0, 0, debugImg.cols, debugImg.rows);
    if (safeRoi.area() <= 0) return;

    cv::Mat debugSub = debugImg(safeRoi);
    int cellW = safeRoi.width / cols;
    int cellH = safeRoi.height / rows;

    for (int r = 0; r < rows; ++r) {
        
        const BubbleResult* res = (r < (int)results.size()) ? &results[r] : nullptr;
        int selectedIdx = -1;
        
        if (res && res->isValid && !res->markedAnswer.empty()) {
            if (res->markedAnswer[0] >= 'A') selectedIdx = res->markedAnswer[0] - 'A';
            else selectedIdx = res->markedAnswer[0] - '0';
        }

        for (int c = 0; c < cols; ++c) {
            
            // Merkez ve Boyutlar
            int centerX = (c * cellW) + (cellW / 2);
            int centerY = (r * cellH) + (cellH / 2);
            int radius = std::min(cellW, cellH) * 0.35;

            // --- TÜM PUANLARI YAZDIRMA MANTIĞI ---
            // Bu kısım biraz tricky çünkü BubbleResult sadece "kazanan" puanı tutuyor.
            // Diğer şıkların puanlarını burada tekrar hesaplamak performans kaybı yaratabilir.
            // Ancak görsel tutarlılık için şimdilik sadece "Kazananın Puanı"nı yazıp,
            // diğerlerine sadece yuvarlak çizebiliriz. 
            // VEYA BubbleResult yapısını değiştirip tüm puanları tutmamız gerekir.
            // Mevcut yapıda en temiz çözüm:
            
            if (c == selectedIdx) {
                // SEÇİLEN: YEŞİL KARE
                cv::Rect cellRect(c * cellW, r * cellH, cellW, cellH);
                cv::rectangle(debugSub, cellRect, cv::Scalar(0, 255, 0), 2);
                
                // Puanı yaz
                if (res) {
                    std::string scoreTxt = std::to_string((int)res->confidence);
                    cv::putText(debugSub, scoreTxt, 
                               cv::Point(centerX - 10, centerY + 5),
                               cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(0, 255, 0), 2);
                }
            } else {
                // SEÇİLMEYEN: GRİ YUVARLAK
                cv::circle(debugSub, cv::Point(centerX, centerY), radius, cv::Scalar(100, 100, 100), 1, cv::LINE_AA);
                
                // (Not: Dersler kısmında her şıkkın puanı şu an elimizde yok, 
                // sadece en yüksek olan var. O yüzden buraya puan yazamıyoruz 
                // ama yuvarlak çizerek ızgarayı gösteriyoruz.)
            }
        }
    }
}
std::vector<BubbleResult> BubbleDetector::detectBubblesByColumn(
    const cv::Mat& roiGray,
    int rows,
    int cols,
    cv::Mat* debugVis)
{
    // 1. Görüntü Ön İşleme (Adaptive Threshold)
    cv::Mat blurImg, thr;
    cv::GaussianBlur(roiGray, blurImg, cv::Size(5, 5), 0);
    
    // ID alanları genelde daha koyu/kalın işaretlenir, o yüzden parametreleri sabit tutuyoruz
    cv::adaptiveThreshold(
        blurImg, thr, 255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        21, 5
    );

    // Gürültü temizliği
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(thr, thr, cv::MORPH_OPEN, kernel);

    std::vector<BubbleResult> results;
    int cellW = roiGray.cols / cols;
    int cellH = roiGray.rows / rows;

    // SÜTUNLARI GEZ (Soldan Sağa: 1. hane, 2. hane...)
    for (int c = 0; c < cols; ++c) {
        
        double bestVal = 0.0;
        int bestRow = -1;

        // SATIRLARI GEZ (Yukarıdan Aşağıya: 0, 1...9)
        for (int r = 0; r < rows; ++r) {
            // Kenar payları (%15)
            int marginX = static_cast<int>(cellW * 0.15);
            int marginY = static_cast<int>(cellH * 0.15);
            
            cv::Rect cell(c * cellW + marginX, r * cellH + marginY, 
                          cellW - 2*marginX, cellH - 2*marginY);
            
            // Güvenlik
            cell &= cv::Rect(0, 0, thr.cols, thr.rows);
            if (cell.width <= 0 || cell.height <= 0) continue;

            // Doluluk Oranı
            int totalPixels = cell.area();
            int nonZero = cv::countNonZero(thr(cell));
            double ratio = (totalPixels > 0) ? (double)nonZero / totalPixels : 0.0;

            if (ratio > bestVal) {
                bestVal = ratio;
                bestRow = r;
            }
        }

        // Sonuç Oluşturma
        BubbleResult res;
        res.questionNumber = c; // Sütun numarasını soru gibi kaydediyoruz
        
        // Eşik değer kontrolü (Örn: %40)
        // ID alanları için biraz daha toleranslı olabiliriz
        double localThr = 0.40; 

        if (bestVal > localThr && bestRow != -1) {
            // Çizim fonksiyonu 'A', 'B' beklediği için rakamı harfe çevirip saklıyoruz.
            // Örn: 0 -> 'A', 1 -> 'B'. ROIDetector tarafında bunu tekrar rakama çevireceğiz.
            res.markedAnswer = std::string(1, 'A' + bestRow);
            res.confidence = bestVal * 100.0;
            res.isValid = true;
        } else {
            res.markedAnswer = "-";
            res.confidence = 0.0;
            res.isValid = false;
        }
        
        results.push_back(res);
    }

    // Debug Çizimi (Eğer istenmişse)
    // TERS PARAMETRE: Sütunları soru, satırları şık gibi çizmesi için (cols, rows) veriyoruz.
    if (debugVis) {
        drawBubbleDebug(*debugVis, cv::Rect(0,0,roiGray.cols, roiGray.rows), results, cols, rows, "");
    }

    return results;
}