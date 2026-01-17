#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <deque>
#include <map>

struct BubbleResult {
    int questionNumber;
    std::string markedAnswer; 
    double confidence;        
    double secondConfidence; 
    bool isValid;            

    BubbleResult() 
      : questionNumber(0), markedAnswer("-"), 
        confidence(0.0), secondConfidence(0.0), isValid(false) {}
};

struct BubbleContour {
    cv::Rect boundingBox;
    cv::Point2f center;
    double area;
    double fillRatio;
    int row;
    int col;
};

class BubbleDetector {
public:
    BubbleDetector(double fillThreshold = 0.45);

    // Eski kodunun aradığı fonksiyonlar burada
    std::vector<BubbleResult> detectBubbles(
        const cv::Mat& roiGray,
        int rows,
        int cols,
        int startQuestionNumber,
        char firstLabel = 'A');
        
    std::vector<BubbleResult> detectBubblesByColumn(
        const cv::Mat& roiGray,
        int rows,
        int cols,
        cv::Mat* debugVis = nullptr);
        
    std::vector<BubbleResult> detectBubblesWithContours(
        const cv::Mat& roiGray,
        int rows,
        int cols,
        int startQuestionNumber,
        char firstLabel,
        cv::Mat* debugVis = nullptr);

    void drawBubbleDebug(
        cv::Mat& debugImg,
        const cv::Rect& roi,
        const std::vector<BubbleResult>& results,
        int rows,
        int cols,
        const std::string& label);

    void setFillThreshold(double t) { fillThreshold_ = t; }
    double getFillThreshold() const { return fillThreshold_; }
    void setDebugMode(bool b) { debugMode_ = b; }
    cv::Mat getLastDebugVisualization() const { return debugVis_; }

private:
    double fillThreshold_;
    
    // --- BU DEĞİŞKENLER EKSİK OLDUĞU İÇİN HATA ALIYORDUN ---
    double minSeparation_ = 10.0;
    bool temporalSmoothingEnabled_ = false;
    int historySize_ = 5;
    bool debugMode_ = false;
    
    cv::Mat debugVis_;
    std::map<int, std::deque<std::string>> answerHistory_;
    std::vector<BubbleContour> lastDetectedBubbles_;
    // -------------------------------------------------------

    double calculateFillRatio(const cv::Mat& bubbleImg);
    
    std::vector<BubbleResult> detectBubblesGridCore(
        const cv::Mat& roiGray,
        int rows,
        int cols,
        int startQuestionNumber,
        char firstLabel,
        bool applySmoothing,
        std::vector<std::vector<double>>* cellFillRatios);

    cv::Rect refineBubbleRect(const cv::Mat& cellPatch, const cv::Rect& initialRect);
};