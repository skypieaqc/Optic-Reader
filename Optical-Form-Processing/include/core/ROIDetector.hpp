#ifndef ROI_DETECTOR_HPP
#define ROI_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>
#include "BubbleDetector.hpp"

class ROIDetector {
public:
    enum RegionType {
        GRID,
        COLUMN
    };
    
    struct RegionDef {
        std::string name;
        float rectPct[4];  // {x, y, w, h} - yüzdelik koordinatlar
        int rows;
        int cols;
        RegionType type;
    };
    
    // Tek bir soru için detaylı bilgi
    struct QuestionDetail {
        int questionNumber;
        char markedAnswer;     // İşaretlenen cevap (A,B,C,D,E veya -)
        char correctAnswer;    // Doğru cevap
        bool isCorrect;        // Doğru mu?
        double fillRatio;      // Doluluk oranı (0.0 - 1.0)
    };
    
    ROIDetector();
    
    // Temel işleme
    std::map<std::string, std::string> process(const cv::Mat& warped, cv::Mat& debugOut);
    
    // Detaylı analiz - her sorunun durumunu döndürür
    std::map<std::string, std::vector<QuestionDetail>> processWithDetails(
        const cv::Mat& warped, 
        cv::Mat& debugOut,
        const std::map<std::string, std::map<int, char>>& answerKey
    );
    
    // Minimum doluluk oranını ayarla (varsayılan: 0.20)
    void setFillThreshold(double threshold);
    
    double getFillThreshold() const { return fillThreshold_; }
    
    // Debug ÇõŽñktŽñsŽñ iÇõin
    void setDebugMode(bool enabled);
    cv::Mat getLastDebugVisualization() const;

private:
    std::vector<RegionDef> regions_;
    double fillThreshold_;
    BubbleDetector bubbleDetector_;
    bool debugMode_;
    cv::Mat lastDebugVis_;
    
    // Helper fonksiyonlar
    std::vector<QuestionDetail> analyzeGridWithDetails(
        const cv::Mat& roiGray,
        int rows,
        int cols,
        const std::map<int, char>& correctAnswers,
        std::vector<BubbleResult>* bubbleResults = nullptr,
        char firstLabel = 'A'
    );
    
    bool isSubjectRegion(const std::string& name) const;
    std::string bubblesToAnswerString(const std::vector<BubbleResult>& results) const;
};

#endif // ROI_DETECTOR_HPP
